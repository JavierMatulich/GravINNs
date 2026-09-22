"""Reference caches, benchmark databases and scoring on many unseen orbits.

The module also provides memory-safe loading and chunked scoring for large
reference caches, together with the pairwise heat-map analysis used in the
manuscript.
"""
import gc
import os

import numpy as np
import torch
try:
    from IPython.display import display as _ipy_display
except ImportError:  # IPython is optional outside notebooks.
    def _ipy_display(_obj):
        return None

from .orbits import C_QD, C_SQ, _pham_batch_local



# ── Learned error filter: theta -> predicted L2 of the PINN ──────────────
def rk4_orbits_batch(thetas, pn_order=2, n_orbits=3, n_pts=2000,
                     n_steps=12000, n_rec=4000, device="cpu"):
    """Integrate a BATCH of PN orbits in parallel with fixed-step RK4 in torch.

    This is the GPU path for generating many reference orbits at once: all B
    initial conditions are advanced together as (B,4) tensors, so 1000 orbits
    cost barely more wall-time than one. The RHS is obtained by autograd on the
    batched Hamiltonian (_pham_batch_local), which is exact.

    Each orbit gets its OWN step size h_i = n_orbits*T_i/n_steps (periods
    differ across the box), so all finish in the same number of steps.

    Returns dict of numpy arrays on a per-orbit uniform phi grid:
      PHI (B,n_pts), U (B,n_pts), PR (B,n_pts), span (B,), L0 (B,)
    """
    import numpy as _np
    TH = torch.as_tensor(_np.asarray(thetas, dtype=_np.float64),
                         dtype=torch.float64, device=device)
    B = TH.shape[0]
    p0f, r0km, pr0f, nu = TH[:, 0], TH[:, 1], TH[:, 2], TH[:, 3]
    q0 = r0km/200.0
    p0t = p0f/torch.sqrt(q0)
    L0 = (q0*p0t).cpu().numpy()
    E = 0.5*(pr0f**2 + p0t**2) - 1.0/q0
    if bool((E >= 0).any()):
        raise ValueError("rk4_orbits_batch: some thetas are unbound (E>=0)")
    a = -0.5/E
    T = 2.0*_np.pi*a**1.5
    h = (float(n_orbits)*T/float(n_steps)).view(B, 1)

    y = torch.stack([q0, torch.zeros_like(q0), pr0f, p0t], dim=1)  # (B,4)

    def rhs(yy):
        q = yy[:, :2].detach().requires_grad_(True)
        p = yy[:, 2:].detach().requires_grad_(True)
        with torch.enable_grad():
            H = _pham_batch_local(q, p, pn_order, nu, C_SQ, C_QD).sum()
            dq, dp = torch.autograd.grad(H, [q, p])
        return torch.cat([dp, -dq], dim=1)

    rec_every = max(1, n_steps//n_rec)
    rec = torch.empty(B, n_steps//rec_every + 1, 4, dtype=torch.float64,
                      device=device)
    rec[:, 0] = y
    ri = 1
    with torch.no_grad():
        for s in range(n_steps):
            k1 = rhs(y)
            k2 = rhs(y + 0.5*h*k1)
            k3 = rhs(y + 0.5*h*k2)
            k4 = rhs(y + h*k3)
            y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            if (s+1) % rec_every == 0 and ri < rec.shape[1]:
                rec[:, ri] = y
                ri += 1
    rec = rec[:, :ri].cpu().numpy()

    PHI = _np.empty((B, n_pts)); U = _np.empty((B, n_pts))
    PR = _np.empty((B, n_pts)); span = _np.empty(B)
    for i in range(B):
        qx, qy, px, py = rec[i, :, 0], rec[i, :, 1], rec[i, :, 2], rec[i, :, 3]
        r = _np.hypot(qx, qy)
        ph = _np.unwrap(_np.arctan2(qy, qx)); ph -= ph[0]
        u = 1.0/r; pr = (qx*px + qy*py)/r
        span[i] = ph[-1]
        gp = _np.linspace(0.0, ph[-1], n_pts)
        PHI[i] = gp
        U[i] = _np.interp(gp, ph, u)
        PR[i] = _np.interp(gp, ph, pr)
    return dict(PHI=PHI, U=U, PR=PR, span=span, L0=L0)


def build_error_dataset(model, param_ranges, n_samples=1000,
                        pde_pn_order=2, n_orbits=3, n_pts=1500,
                        n_steps=12000, max_ecc=0.95, seed=11,
                        batch_size=256, device="cpu", verbose=True):
    """TRUE L2 error of the trained PINN at `n_samples` random thetas.

    Thetas are sampled uniformly in the box (rejecting unbound / over-eccentric
    ones); references come from the batched GPU RK4 integrator, so ~1000 orbits
    take minutes, not hours. Returns (thetas (n,4), l2 (n,)) -- the supervised
    dataset for the error-filter net."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    keys = ("p0_factor", "r0_km", "pr0_factor", "nu")
    lo = _np.array([param_ranges[k][0] for k in keys])
    hi = _np.array([param_ranges[k][1] for k in keys])

    def _valid(th):
        p0f, r0km, pr0f, _ = th
        q0 = r0km/200.0; p0t = p0f/_np.sqrt(q0)
        E = 0.5*(pr0f**2 + p0t**2) - 1.0/q0
        if E >= 0: return False
        ecc = _np.sqrt(max(0.0, 1 + 2*E*(q0*p0t)**2))
        return ecc <= max_ecc

    TH = []
    while len(TH) < n_samples:
        c = lo + rng.random(4)*(hi - lo)
        if _valid(c): TH.append(c)
    TH = _np.array(TH)

    PM = float(model.phi_max)
    L2 = _np.empty(n_samples)
    for b0 in range(0, n_samples, batch_size):
        chunk = TH[b0:b0+batch_size]
        rk = rk4_orbits_batch(chunk, pde_pn_order, n_orbits, n_pts,
                              n_steps=n_steps, device=device)
        for j, th in enumerate(chunk):
            gp = _np.linspace(0.0, min(rk["span"][j], PM), n_pts)
            u_r = _np.interp(gp, rk["PHI"][j], rk["U"][j])
            pr_r = _np.interp(gp, rk["PHI"][j], rk["PR"][j])
            L0 = rk["L0"][j]
            qxr = _np.cos(gp)/u_r; qyr = _np.sin(gp)/u_r
            pxr = pr_r*_np.cos(gp) - (L0*u_r)*_np.sin(gp)
            pyr = pr_r*_np.sin(gp) + (L0*u_r)*_np.cos(gp)
            ref = _np.stack([qxr, qyr, pxr, pyr])
            with torch.no_grad():
                P = torch.tensor(gp, dtype=torch.float32, device=device).view(-1, 1)
                Tt = torch.tensor([list(th)], dtype=torch.float32,
                                  device=device).repeat(len(gp), 1)
                out = model(P, Tt).cpu().numpy()
            u_n, pr_n, L_n = out[:, 0], out[:, 1], out[:, 2]
            qx = _np.cos(gp)/u_n; qy = _np.sin(gp)/u_n
            px = pr_n*_np.cos(gp) - (L_n*u_n)*_np.sin(gp)
            py = pr_n*_np.sin(gp) + (L_n*u_n)*_np.cos(gp)
            net = _np.stack([qx, qy, px, py])
            L2[b0+j] = _np.linalg.norm(net-ref)/(_np.linalg.norm(ref)+1e-16)
        if verbose:
            print(f"  error dataset: {min(b0+batch_size, n_samples)}/{n_samples} "
                  f"orbits done", flush=True)
    return TH, L2


class ErrorFilterNet(torch.nn.Module):
    """theta (4) -> predicted log10 L2 of the parametric PINN at that theta."""
    def __init__(self, param_ranges, hidden=64):
        super().__init__()
        keys = ("p0_factor", "r0_km", "pr0_factor", "nu")
        lo = torch.tensor([param_ranges[k][0] for k in keys], dtype=torch.float32)
        hi = torch.tensor([param_ranges[k][1] for k in keys], dtype=torch.float32)
        self.register_buffer("lo", lo); self.register_buffer("hi", hi)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1))

    def forward(self, th):
        x = 2.0*(th - self.lo)/(self.hi - self.lo + 1e-12) - 1.0
        return self.net(x)[:, 0]

    def predict_l2(self, theta):
        """Predicted L2 (not log) for one theta tuple."""
        t = torch.as_tensor([list(theta)], dtype=torch.float32,
                            device=self.lo.device)
        with torch.no_grad():
            return float(10.0**self(t)[0])


def train_error_filter(thetas, l2, param_ranges, hidden=64,
                       epochs=4000, lr=3e-3, val_frac=0.2, seed=0,
                       device="cpu", verbose=True):
    """Fit ErrorFilterNet to (theta, L2) pairs in log10 space.

    Reports the validation median |predicted - true| in log10 units (0.3 means
    within a factor of 2) and the log-space correlation, so you know how much
    to trust the filter before wiring it into the chain."""
    import numpy as _np
    torch.manual_seed(seed)
    n = len(l2)
    perm = _np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(val_frac*n))
    iv, itr = perm[:n_val], perm[n_val:]
    TH = torch.tensor(_np.asarray(thetas), dtype=torch.float32, device=device)
    Y = torch.tensor(_np.log10(_np.maximum(l2, 1e-12)),
                     dtype=torch.float32, device=device)
    net = ErrorFilterNet(param_ranges, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                     eta_min=lr*1e-2)
    for ep in range(epochs):
        opt.zero_grad()
        loss = ((net(TH[itr]) - Y[itr])**2).mean()
        loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        pv = net(TH[iv]).cpu().numpy(); yv = Y[iv].cpu().numpy()
    med = float(_np.median(_np.abs(pv - yv)))
    corr = float(_np.corrcoef(pv, yv)[0, 1]) if n_val > 2 else float("nan")
    if verbose:
        print(f"  error filter: val median |dlog10 L2| = {med:.3f} "
              f"(factor {10**med:.2f}), log-corr = {corr:.3f} "
              f"on {n_val} held-out thetas")
    return net, dict(val_median_dlog=med, val_corr=corr)


# ── Benchmark database + pairwise error projections ──────────────────────
_PKEYS4 = ("p0_factor", "r0_km", "pr0_factor", "nu")


def build_benchmark_db(model, param_ranges, path="benchmark_db.npz",
                       n_samples=2000, anchor_thetas=None,
                       pde_pn_order=2, n_orbits=3, n_pts=1500,
                       n_steps=12000, max_ecc=0.95, seed=1234,
                       batch_size=256, device="cpu", verbose=True):
    """Build (and SAVE) a benchmark database: n_samples random thetas in the
    box, their GPU-batched RK reference orbits, and the trained model's
    relative L2 error against each.

    The thetas are drawn INDEPENDENTLY of the training anchors: evaluating on
    the anchors themselves would measure memorisation, not accuracy. If
    `anchor_thetas` is supplied, each sample also stores its normalised
    distance to the nearest anchor, so the same database answers both
    questions -- "how good is the model near training data?" and "how good is
    it far from it?" -- by slicing on that column.

    References come from rk4_orbits_batch: all initial conditions advance in
    parallel as one (B,4) state tensor on the GPU, so 2000 orbits cost a
    handful of batches rather than 2000 sequential integrations. Fixed-step
    RK4 at n_steps was validated against the adaptive scipy integrator to
    ~2e-5 -- far below the 1e-4..1e-2 errors being measured.

    Saves an .npz containing thetas, l2, ecc, energy, anchor_dist and the run
    metadata needed to interpret them. Load it with load_benchmark_db().

    Returns the same dict that load_benchmark_db() returns.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    lo = _np.array([param_ranges[k][0] for k in _PKEYS4], float)
    hi = _np.array([param_ranges[k][1] for k in _PKEYS4], float)

    def _geom(th):
        """(E, ecc) of the Newtonian IC; None if unbound."""
        p0f, r0km, pr0f, _nu = th
        q0 = r0km/200.0
        if q0 <= 0:
            return None
        p0t = p0f/_np.sqrt(q0)
        E = 0.5*(pr0f**2 + p0t**2) - 1.0/q0
        if not (E < 0):
            return None
        L0 = q0*p0t
        return E, float(_np.sqrt(max(0.0, 1.0 + 2.0*E*L0**2)))

    TH, ECC, EN = [], [], []
    tries = 0
    while len(TH) < n_samples and tries < 200*n_samples:
        tries += 1
        c = lo + rng.random(4)*(hi - lo)
        g = _geom(c)
        if g is None or g[1] > max_ecc:
            continue
        TH.append(c); EN.append(g[0]); ECC.append(g[1])
    if len(TH) < n_samples:
        raise RuntimeError(
            f"only {len(TH)}/{n_samples} valid samples after {tries} draws; "
            f"the box contains many unbound / ecc>{max_ecc} orbits.")
    TH = _np.array(TH); ECC = _np.array(ECC); EN = _np.array(EN)

    # normalised distance to the nearest TRAINING anchor (0 = on an anchor)
    if anchor_thetas is not None and len(anchor_thetas):
        span = _np.maximum(hi - lo, 1e-12)
        A = (_np.asarray([list(a) for a in anchor_thetas], float) - lo)/span
        C = (TH - lo)/span
        ADIST = _np.array([float(_np.min(_np.linalg.norm(A - c, axis=1)))
                           for c in C])
    else:
        ADIST = _np.full(len(TH), _np.nan)

    PM = float(model.phi_max)
    L2 = _np.empty(len(TH))
    for b0 in range(0, len(TH), batch_size):
        chunk = TH[b0:b0+batch_size]
        rk = rk4_orbits_batch(chunk, pde_pn_order, n_orbits, n_pts,
                              n_steps=n_steps, device=device)
        for j, th in enumerate(chunk):
            gp = _np.linspace(0.0, min(rk["span"][j], PM), n_pts)
            u_r = _np.interp(gp, rk["PHI"][j], rk["U"][j])
            pr_r = _np.interp(gp, rk["PHI"][j], rk["PR"][j])
            L0 = rk["L0"][j]
            qxr = _np.cos(gp)/u_r; qyr = _np.sin(gp)/u_r
            pxr = pr_r*_np.cos(gp) - (L0*u_r)*_np.sin(gp)
            pyr = pr_r*_np.sin(gp) + (L0*u_r)*_np.cos(gp)
            ref = _np.stack([qxr, qyr, pxr, pyr])
            with torch.no_grad():
                P = torch.tensor(gp, dtype=torch.float32, device=device).view(-1, 1)
                Tt = torch.tensor([list(th)], dtype=torch.float32,
                                  device=device).repeat(len(gp), 1)
                out = model(P, Tt).cpu().numpy()
            u_n, pr_n, L_n = out[:, 0], out[:, 1], out[:, 2]
            qx = _np.cos(gp)/u_n; qy = _np.sin(gp)/u_n
            px = pr_n*_np.cos(gp) - (L_n*u_n)*_np.sin(gp)
            py = pr_n*_np.sin(gp) + (L_n*u_n)*_np.cos(gp)
            net = _np.stack([qx, qy, px, py])
            L2[b0+j] = _np.linalg.norm(net-ref)/(_np.linalg.norm(ref)+1e-16)
        if verbose:
            print(f"  benchmark: {min(b0+batch_size, len(TH))}/{len(TH)} orbits",
                  flush=True)

    _np.savez_compressed(
        path, thetas=TH, l2=L2, ecc=ECC, energy=EN, anchor_dist=ADIST,
        param_ranges=_np.array([[param_ranges[k][0], param_ranges[k][1]]
                                for k in _PKEYS4]),
        keys=_np.array(_PKEYS4),
        meta=_np.array([pde_pn_order, n_orbits, n_pts, n_steps, PM, seed],
                       dtype=float))
    if verbose:
        print(f"saved -> {path}  ({len(TH)} orbits)")
        _report_db(dict(thetas=TH, l2=L2, ecc=ECC, anchor_dist=ADIST))
    return load_benchmark_db(path, verbose=False)


def load_benchmark_db(path="benchmark_db.npz", verbose=True):
    """Load a database written by build_benchmark_db().

    Returns dict with: thetas (n,4), l2 (n,), ecc (n,), energy (n,),
    anchor_dist (n,), param_ranges (dict), keys, and meta
    (pde_pn_order, n_orbits, n_pts, n_steps, phi_max, seed).
    """
    import numpy as _np
    d = _np.load(path, allow_pickle=False)
    keys = [str(k) for k in d["keys"]]
    pr = {k: (float(v[0]), float(v[1])) for k, v in zip(keys, d["param_ranges"])}
    m = d["meta"]
    out = dict(thetas=d["thetas"], l2=d["l2"], ecc=d["ecc"],
               energy=d["energy"], anchor_dist=d["anchor_dist"],
               param_ranges=pr, keys=keys,
               meta=dict(pde_pn_order=int(m[0]), n_orbits=int(m[1]),
                         n_pts=int(m[2]), n_steps=int(m[3]),
                         phi_max=float(m[4]), seed=int(m[5])))
    if verbose:
        print(f"loaded {path}: {len(out['l2'])} orbits, "
              f"pn={out['meta']['pde_pn_order']} n_orbits={out['meta']['n_orbits']}")
        _report_db(out)
    return out


def _report_db(db, tol=1e-3):
    import numpy as _np
    l2 = db["l2"]
    print(f"  L2: median {_np.median(l2):.3e}  mean {l2.mean():.3e}  "
          f"max {l2.max():.3e}")
    print(f"  fraction below {tol:.0e}: {100*float((l2 < tol).mean()):.1f}%  "
          f"({int((l2 < tol).sum())}/{len(l2)})")
    ad = db.get("anchor_dist")
    if ad is not None and _np.isfinite(ad).all():
        near = ad < _np.quantile(ad, 0.25)
        far = ad > _np.quantile(ad, 0.75)
        print(f"  nearest quartile to an anchor : median L2 {_np.median(l2[near]):.3e}"
              f"   below {tol:.0e}: {100*float((l2[near] < tol).mean()):.0f}%")
        print(f"  farthest quartile from anchors: median L2 {_np.median(l2[far]):.3e}"
              f"   below {tol:.0e}: {100*float((l2[far] < tol).mean()):.0f}%")
def db_pair_heatmaps(db, nbins=12, stat="median", tol=1e-3, log=True,
                     figsize=(13.5, 8.0), cmap="viridis", save_plots=False,
                     save_data=None):
    """All six pairwise (theta_i, theta_j) projections of the benchmark error.

    Each cell aggregates the scattered samples whose parameters fall in that
    2D bin, MARGINALISING over the other two parameters -- unlike the training
    monitor's heatmap, which fixes them at mid-box. Cells with no samples are
    left blank.

    stat : "median" -> median L2 in the bin (log10 colour if log=True)
           "frac"   -> fraction of the bin's orbits below `tol` (linear colour)
    save_plots: bool -> If True and stat=="median", saves each of the 6
                        heatmaps separately as a .png file.
    save_data : str, True, or None -> if given, saves an .npz with EVERYTHING
                        needed to redraw these heatmaps later without the
                        model or cache: the raw scatter (thetas, l2, keys,
                        param_ranges) -- which lets you rebin at any nbins/
                        stat/tol/log later -- AND the exact binned Z grid
                        for each of the 6 pairs at THIS call's settings, so
                        a straight reproduction needs no rebinning at all.
                        True -> "db_heatmap_{stat}.npz" in the current dir.
    """
    import os as _os
    import numpy as _np
    import matplotlib.pyplot as plt
    from itertools import combinations

    TH = db["thetas"]; l2 = db["l2"]; keys = db["keys"]; pr = db["param_ranges"]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    pairs = list(combinations(range(4), 2))
    ims = []
    _pair_Z = []          # binned grid actually drawn, one per pair (for save_data)
    _pair_extent = []     # [xi0, xi1, yj0, yj1], one per pair

    for ax, (i, j) in zip(axes.ravel(), pairs):
        xi = _np.linspace(pr[keys[i]][0], pr[keys[i]][1], nbins+1)
        yj = _np.linspace(pr[keys[j]][0], pr[keys[j]][1], nbins+1)
        Z = _np.full((nbins, nbins), _np.nan)
        ix = _np.clip(_np.digitize(TH[:, i], xi)-1, 0, nbins-1)
        iy = _np.clip(_np.digitize(TH[:, j], yj)-1, 0, nbins-1)

        for a in range(nbins):
            for b in range(nbins):
                sel = (ix == a) & (iy == b)
                if sel.sum() == 0:
                    continue
                Z[b, a] = (_np.median(l2[sel]) if stat == "median"
                           else float((l2[sel] < tol).mean()))

        if stat == "median" and log:
            Zp = _np.log10(Z); lab = r"$\log_{10} L^2$"
        else:
            Zp = Z; lab = (f"frac < {tol:.0e}" if stat == "frac" else "L2")

        _pair_Z.append(Zp.astype(_np.float64))
        _pair_extent.append([float(xi[0]), float(xi[-1]), float(yj[0]), float(yj[-1])])

        # --- 1. Plot on the main 2x3 grid ---
        im = ax.imshow(Zp, origin="lower", aspect="auto", cmap=cmap,
                       extent=[xi[0], xi[-1], yj[0], yj[-1]],
                       vmin=(0 if stat == "frac" else None),
                       vmax=(1 if stat == "frac" else None))
        ax.set_xlabel(keys[i], fontsize=8); ax.set_ylabel(keys[j], fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(lab, fontsize=7)
        ims.append(im)

        # --- 2. Save individual plot if requested ---
        if save_plots and stat == "median":
            # Create a separate, temporary figure for just this heatmap
            fig_single, ax_single = plt.subplots(figsize=(6, 5))
            im_single = ax_single.imshow(Zp, origin="lower", aspect="auto", cmap=cmap,
                                         extent=[xi[0], xi[-1], yj[0], yj[-1]])
            ax_single.set_xlabel(keys[i])
            ax_single.set_ylabel(keys[j])
            fig_single.colorbar(im_single, ax=ax_single).set_label(lab)

            # Clean up the variable names for the filename (removes spaces, $, and slashes)
            safe_key_i = str(keys[i]).replace('\\', '').replace('$', '').replace(' ', '_')
            safe_key_j = str(keys[j]).replace('\\', '').replace('$', '').replace(' ', '_')
            filename = f"heatmap_median_{safe_key_i}_vs_{safe_key_j}.png"

            fig_single.tight_layout()
            fig_single.savefig(filename, dpi=300, bbox_inches="tight")

            # Close the temporary figure so it doesn't pop up on your screen
            plt.close(fig_single)
            print(f"Saved individual file: {filename}")

    n_per_bin = len(l2)/(nbins**2)
    fig.suptitle(f"Benchmark: {len(l2)} random orbits, pairwise projections "
                 f"({stat}; ~{n_per_bin:.0f} orbits/bin, other 2 params "
                 f"marginalised)", fontsize=10)
    fig.tight_layout()

    # --- 3. Save the underlying data, if requested -------------------------
    if save_data:
        _path = save_data if isinstance(save_data, str) else f"db_heatmap_{stat}.npz"
        _os.makedirs(_os.path.dirname(_path) or ".", exist_ok=True)
        _tmp = _path + ".tmp"
        with open(_tmp, "wb") as _f:
            _np.savez_compressed(
                _f,
                # raw scatter -- enough to rebin at ANY nbins/stat/tol/log later
                thetas=_np.asarray(TH, dtype=_np.float64),
                l2=_np.asarray(l2, dtype=_np.float64),
                keys=_np.array(list(keys)),
                param_range_names=_np.array(list(keys)),
                param_range_values=_np.array([pr[k] for k in keys], dtype=_np.float64),
                # this call's settings + the exact grids it drew, for instant reproduction
                nbins=int(nbins), stat=str(stat), tol=float(tol), log=bool(log),
                pairs=_np.asarray(pairs, dtype=_np.int64),          # (6,2) index pairs into keys
                pair_Z=_np.stack(_pair_Z),                          # (6, nbins, nbins)
                pair_extent=_np.asarray(_pair_extent, dtype=_np.float64),  # (6,4)
            )
        _os.replace(_tmp, _path)
        print(f"  Heatmap data → {_path}")

    try:
        _ipy_display(fig)
    except Exception:
        pass
    # [repo fix] the figure was shown twice in Jupyter: once by the explicit
    # display above and again by the inline backend at the end of the cell,
    # because it was still open. Closing it keeps exactly one copy on screen;
    # the returned fig/axes stay usable (e.g. fig.savefig(...)).
    plt.close(fig)

    return fig, axes


# ── Reference-orbit cache: integrate once, score many models ─────────────
def build_reference_cache(param_ranges, path="refs_2000.npz",
                          n_samples=2000, anchor_thetas=None,
                          pde_pn_order=2, n_orbits=3, n_pts=1500,
                          n_steps=12000, max_ecc=0.95, seed=1234,
                          batch_size=256, device="cpu", verbose=True):
    """Integrate a fixed benchmark set ONCE and store the reference orbits.

    Unlike build_benchmark_db (which stores only the resulting L2 of one
    model), this stores the RK solutions themselves, so ANY number of models
    -- Adam-phase, refined, retrained, different capacity -- can afterwards be
    scored on the IDENTICAL orbits with pure network inference and no
    integration at all (eval_model_on_cache).

    Because every model is scored on the same thetas AND the same references,
    the resulting L2 arrays are paired sample-by-sample: you can report
    "improved on X% of the same 2000 orbits" and the median ratio, which are
    far stronger statistics than comparing two independent samples.

    NOTE the reference orbits are stored on each orbit's OWN uniform phi grid
    of n_pts points, spanning [0, span_i]. A model is later evaluated on
    [0, min(span_i, model.phi_max)] by interpolation, so a cache can serve
    models with different phi_max (e.g. trained with different n_orbits),
    provided the PN order matches.

    Storage: float32, ~2*n_samples*n_pts*4 bytes for (U, PR) plus the phi
    grids -> ~36 MB for 2000 x 1500. Set store_phi=False style compaction is
    unnecessary at this size.

    Returns the dict that load_reference_cache() returns.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    lo = _np.array([param_ranges[k][0] for k in _PKEYS4], float)
    hi = _np.array([param_ranges[k][1] for k in _PKEYS4], float)

    def _geom(th):
        p0f, r0km, pr0f, _nu = th
        q0 = r0km/200.0
        if q0 <= 0:
            return None
        p0t = p0f/_np.sqrt(q0)
        E = 0.5*(pr0f**2 + p0t**2) - 1.0/q0
        if not (E < 0):
            return None
        L0 = q0*p0t
        return E, float(_np.sqrt(max(0.0, 1.0 + 2.0*E*L0**2)))

    # ── identical sampling to build_benchmark_db: same seed -> same thetas ──
    TH, ECC, EN = [], [], []
    tries = 0
    while len(TH) < n_samples and tries < 200*n_samples:
        tries += 1
        c = lo + rng.random(4)*(hi - lo)
        g = _geom(c)
        if g is None or g[1] > max_ecc:
            continue
        TH.append(c); EN.append(g[0]); ECC.append(g[1])
    if len(TH) < n_samples:
        raise RuntimeError(f"only {len(TH)}/{n_samples} valid samples after "
                           f"{tries} draws; the box has many unbound/eccentric "
                           f"orbits.")
    TH = _np.array(TH); ECC = _np.array(ECC); EN = _np.array(EN)

    if anchor_thetas is not None and len(anchor_thetas):
        span_n = _np.maximum(hi - lo, 1e-12)
        A = (_np.asarray([list(a) for a in anchor_thetas], float) - lo)/span_n
        C = (TH - lo)/span_n
        ADIST = _np.array([float(_np.min(_np.linalg.norm(A - c, axis=1)))
                           for c in C])
    else:
        ADIST = _np.full(len(TH), _np.nan)

    # ── integrate once, on the GPU, in batches ─────────────────────────────
    PHI = _np.empty((len(TH), n_pts), dtype=_np.float32)
    U = _np.empty_like(PHI); PR = _np.empty_like(PHI)
    SPAN = _np.empty(len(TH)); L0A = _np.empty(len(TH))
    for b0 in range(0, len(TH), batch_size):
        chunk = TH[b0:b0+batch_size]
        rk = rk4_orbits_batch(chunk, pde_pn_order, n_orbits, n_pts,
                              n_steps=n_steps, device=device)
        sl = slice(b0, b0+len(chunk))
        PHI[sl] = rk["PHI"].astype(_np.float32)
        U[sl] = rk["U"].astype(_np.float32)
        PR[sl] = rk["PR"].astype(_np.float32)
        SPAN[sl] = rk["span"]; L0A[sl] = rk["L0"]
        if verbose:
            print(f"  reference cache: {min(b0+batch_size, len(TH))}/{len(TH)} "
                  f"orbits", flush=True)

    _np.savez_compressed(
        path, thetas=TH, ecc=ECC, energy=EN, anchor_dist=ADIST,
        PHI=PHI, U=U, PR=PR, span=SPAN, L0=L0A,
        param_ranges=_np.array([[param_ranges[k][0], param_ranges[k][1]]
                                for k in _PKEYS4]),
        keys=_np.array(_PKEYS4),
        meta=_np.array([pde_pn_order, n_orbits, n_pts, n_steps, seed],
                       dtype=float))
    if verbose:
        import os
        print(f"saved -> {path}  ({len(TH)} reference orbits, "
              f"{os.path.getsize(path)/1e6:.1f} MB)")
    return load_reference_cache(path, verbose=False)


def load_reference_cache(path="refs_2000.npz", verbose=True):
    """Load a reference cache written by build_reference_cache()."""
    import numpy as _np
    d = _np.load(path, allow_pickle=False)
    keys = [str(k) for k in d["keys"]]
    pr = {k: (float(v[0]), float(v[1])) for k, v in zip(keys, d["param_ranges"])}
    m = d["meta"]
    out = dict(thetas=d["thetas"], ecc=d["ecc"], energy=d["energy"],
               anchor_dist=d["anchor_dist"], PHI=d["PHI"], U=d["U"],
               PR=d["PR"], span=d["span"], L0=d["L0"],
               param_ranges=pr, keys=keys,
               meta=dict(pde_pn_order=int(m[0]), n_orbits=int(m[1]),
                         n_pts=int(m[2]), n_steps=int(m[3]), seed=int(m[4])))
    if verbose:
        print(f"loaded {path}: {len(out['thetas'])} reference orbits, "
              f"pn={out['meta']['pde_pn_order']} n_orbits={out['meta']['n_orbits']} "
              f"n_pts={out['meta']['n_pts']}")
    return out


def eval_model_on_cache(model, cache, device="cpu", batch_size=200,
                        verbose=True, tol=1e-3):
    """Score a model against a cached reference set. NO integration.

    All model evaluations for a batch of orbits are done in ONE forward pass
    (the network is fully vectorised over (phi, theta)), so 2000 orbits take
    seconds. Returns a db-shaped dict compatible with _report_db() and
    db_pair_heatmaps().
    """
    import numpy as _np
    TH = cache["thetas"]; n = len(TH)
    n_pts = cache["meta"]["n_pts"]
    PM = float(model.phi_max)
    if cache["meta"]["pde_pn_order"] != getattr(model, "_pn_order", cache["meta"]["pde_pn_order"]):
        pass  # PN order is a property of the cache; the caller must match it
    L2 = _np.empty(n)
    for b0 in range(0, n, batch_size):
        sl = slice(b0, min(b0+batch_size, n))
        thb = TH[sl]; B = len(thb)
        # per-orbit evaluation grid: [0, min(span_i, phi_max)]
        gp = _np.stack([_np.linspace(0.0, min(float(cache["span"][b0+j]), PM),
                                     n_pts) for j in range(B)])          # (B,n_pts)
        # interpolate the cached reference onto that grid
        u_r = _np.stack([_np.interp(gp[j], cache["PHI"][b0+j], cache["U"][b0+j])
                         for j in range(B)])
        pr_r = _np.stack([_np.interp(gp[j], cache["PHI"][b0+j], cache["PR"][b0+j])
                          for j in range(B)])
        L0 = cache["L0"][sl][:, None]
        ref = _np.stack([_np.cos(gp)/u_r, _np.sin(gp)/u_r,
                         pr_r*_np.cos(gp) - (L0*u_r)*_np.sin(gp),
                         pr_r*_np.sin(gp) + (L0*u_r)*_np.cos(gp)], axis=1)  # (B,4,n_pts)
        # ONE forward pass for the whole batch
        P = torch.tensor(gp.reshape(-1, 1), dtype=torch.float32, device=device)
        T = torch.tensor(_np.repeat(thb, n_pts, axis=0), dtype=torch.float32,
                         device=device)
        with torch.no_grad():
            o = model(P, T).cpu().numpy().reshape(B, n_pts, 3)
        u_n, pr_n, L_n = o[:, :, 0], o[:, :, 1], o[:, :, 2]
        net = _np.stack([_np.cos(gp)/u_n, _np.sin(gp)/u_n,
                         pr_n*_np.cos(gp) - (L_n*u_n)*_np.sin(gp),
                         pr_n*_np.sin(gp) + (L_n*u_n)*_np.cos(gp)], axis=1)
        num = _np.linalg.norm((net-ref).reshape(B, -1), axis=1)
        den = _np.linalg.norm(ref.reshape(B, -1), axis=1)
        L2[sl] = num/(den + 1e-16)
        #if verbose:
         #   print(f"  eval: {sl.stop}/{n}", flush=True)
        if verbose and (b0 % (batch_size * 50) == 0 or sl.stop == n):
            print(f"  eval: {sl.stop}/{n}", flush=True)
    db = dict(thetas=TH, l2=L2, ecc=cache["ecc"], energy=cache["energy"],
              anchor_dist=cache["anchor_dist"],
              param_ranges=cache["param_ranges"], keys=cache["keys"],
              meta=dict(cache["meta"], phi_max=PM))
    if verbose:
        _report_db(db, tol=tol)
    return db


def compare_models_on_cache(db_a, db_b, name_a="before", name_b="after",
                            tol=1e-3):
    """Paired comparison of two models scored on the SAME cache."""
    import numpy as _np
    a, b = db_a["l2"], db_b["l2"]
    if len(a) != len(b):
        raise ValueError("the two dbs have different lengths -- were they "
                         "scored on the same cache?")
    if not _np.allclose(db_a["thetas"], db_b["thetas"]):
        raise ValueError("theta sets differ -- the comparison would not be "
                         "paired. Score both models on the SAME cache.")
    print(f"{'':<10}{'median':>12}{'mean':>12}{'max':>12}{'<tol':>9}")
    for nm, x in ((name_a, a), (name_b, b)):
        print(f"{nm:<10}{_np.median(x):>12.3e}{x.mean():>12.3e}"
              f"{x.max():>12.3e}{100*float((x < tol).mean()):>8.1f}%")
    print(f"\npaired over the SAME {len(a)} orbits:")
    print(f"  {name_b} better on {100*float((b < a).mean()):.1f}% of them")
    print(f"  median ratio {name_b}/{name_a} = {_np.median(b/a):.3f}")
    ecc = db_a["ecc"]
    print(f"\n{'ecc bin':<14}{'n':>6}{name_a+' med':>14}{name_b+' med':>14}"
          f"{name_a+' <tol':>12}{name_b+' <tol':>12}")
    for x0, x1 in ((0.0, 0.30), (0.30, 0.50), (0.50, 0.65), (0.65, 0.80),
                   (0.80, 1.01)):
        s = (ecc >= x0) & (ecc < x1)
        if not s.sum():
            continue
        print(f"[{x0:.2f},{x1:.2f})  {int(s.sum()):>6}"
              f"{_np.median(a[s]):>14.2e}{_np.median(b[s]):>14.2e}"
              f"{100*float((a[s] < tol).mean()):>11.0f}%"
              f"{100*float((b[s] < tol).mean()):>11.0f}%")

def eval_chunked(model, cache, device, chunk=10000, batch_size=16):
    n = len(cache["thetas"]); L2 = np.empty(n)
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        sub = dict(cache)
        for k in ("thetas", "ecc", "energy", "anchor_dist", "span", "L0",
                  "PHI", "U", "PR"):
            sub[k] = cache[k][c0:c1]
        d = eval_model_on_cache(model, sub, device=device,
                                batch_size=batch_size, verbose=False)
        L2[c0:c1] = d["l2"]
        del sub, d
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {c1}/{n}  mean L2 = {L2[:c1].mean():.3e}", flush=True)
    out = dict(cache); out["l2"] = L2
    return out


# ============================================================================
#  Memory-safe cache loading + scoring helpers
# ============================================================================
_CACHE = {}   # keep exactly one cache alive at a time

def _free_cache(name=None):
    """Explicitly drop the previously loaded cache and force a GC."""
    global _CACHE
    if name is None or name in _CACHE:
        _CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_reference_cache_lazy(path="refs_20000_10orb_new.npz", verbose=True):
    """
    Same schema as load_reference_cache, but:
      * only ONE cache is kept in memory at a time,
      * the big (N, n_pts) arrays are float32 (halves RAM vs float64),
      * meta / thetas / ecc / span / L0 are cheap and loaded eagerly.
    """
    import numpy as _np
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    _free_cache()                      # <-- releases the previous 100k cache

    d = _np.load(path, allow_pickle=False)
    keys = [str(k) for k in d["keys"]]
    pr   = {k: (float(v[0]), float(v[1]))
            for k, v in zip(keys, d["param_ranges"])}
    m    = d["meta"]

    out = dict(
        thetas      = _np.asarray(d["thetas"],      dtype=_np.float32),
        ecc         = _np.asarray(d["ecc"],         dtype=_np.float32),
        energy      = _np.asarray(d["energy"],      dtype=_np.float32),
        anchor_dist = _np.asarray(d["anchor_dist"], dtype=_np.float32),
        span        = _np.asarray(d["span"],        dtype=_np.float32),
        L0          = _np.asarray(d["L0"],          dtype=_np.float32),
        PHI         = _np.asarray(d["PHI"],         dtype=_np.float32),  # (N, n_pts)
        U           = _np.asarray(d["U"],           dtype=_np.float32),
        PR          = _np.asarray(d["PR"],          dtype=_np.float32),
        param_ranges= pr, keys=keys,
        meta        = dict(pde_pn_order=int(m[0]), n_orbits=int(m[1]),
                           n_pts=int(m[2]), n_steps=int(m[3]),
                           seed=int(m[4])),
    )
    d.close()                          # release the npz handle immediately
    _CACHE[path] = out
    if verbose:
        n = len(out["thetas"])
        print(f"loaded {path}: {n} orbits, pn={out['meta']['pde_pn_order']} "
              f"n_orbits={out['meta']['n_orbits']} n_pts={out['meta']['n_pts']} "
              f"(~{out['PHI'].nbytes*3/1e6:.0f} MB)")
    return out


# ── memory-light replacement for eval_model_on_cache ────────────────────────
@torch.no_grad()
def score_on_cache(model, cache, batch_size=16, device="cpu", verbose=True,
                   print_every=5000):
    """Return (N,) float32 L2 array. One model at a time.

    print_every : [repo] progress line every this many ORBITS (was every
                  50 batches = 800 orbits at batch_size=16).
    """
    import numpy as _np

    TH   = cache["thetas"]                 # (N, 4)  float32
    PHI  = cache["PHI"]; U = cache["U"]; PR = cache["PR"]
    span = cache["span"]; L0 = cache["L0"]
    n    = len(TH); n_pts = cache["meta"]["n_pts"]
    PM   = float(model.phi_max)

    L2 = _np.empty(n, dtype=_np.float32)

    for b0 in range(0, n, batch_size):
        b1 = min(b0 + batch_size, n)
        B  = b1 - b0

        gp    = _np.empty((B, n_pts), dtype=_np.float32)
        u_r   = _np.empty_like(gp)
        pr_r  = _np.empty_like(gp)
        for j in range(B):
            hi = min(float(span[b0+j]), PM)
            g  = _np.linspace(0.0, hi, n_pts, dtype=_np.float32)
            gp[j]  = g
            u_r[j] = _np.interp(g, PHI[b0+j], U[b0+j]).astype(_np.float32)
            pr_r[j]= _np.interp(g, PHI[b0+j], PR[b0+j]).astype(_np.float32)

        L0b = L0[b0:b1].astype(_np.float32)[:, None]          # (B,1)
        cos_g = _np.cos(gp); sin_g = _np.sin(gp)

        ref = _np.stack([cos_g / u_r, sin_g / u_r,
                         pr_r*cos_g - (L0b*u_r)*sin_g,
                         pr_r*sin_g + (L0b*u_r)*cos_g], axis=1)  # (B,4,n_pts)

        P = torch.from_numpy(gp.reshape(-1, 1)).to(device)
        T = torch.from_numpy(_np.repeat(TH[b0:b1], n_pts, axis=0)).to(device)
        o = model(P, T).cpu().numpy().reshape(B, n_pts, 3)

        u_n, pr_n, L_n = o[:, :, 0], o[:, :, 1], o[:, :, 2]
        net = _np.stack([_np.cos(gp)/u_n, _np.sin(gp)/u_n,
                         pr_n*_np.cos(gp) - (L_n*u_n)*_np.sin(gp),
                         pr_n*_np.sin(gp) + (L_n*u_n)*_np.cos(gp)], axis=1)

        num = _np.linalg.norm((net - ref).reshape(B, -1), axis=1)
        den = _np.linalg.norm(ref.reshape(B, -1), axis=1)
        L2[b0:b1] = (num / (den + 1e-16)).astype(_np.float32)

        # ---- free the big temporaries NOW, before the next batch ----
        del gp, u_r, pr_r, ref, net, o, P, T
        del cos_g, sin_g, num, den
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose and (b1 == n or b1 // print_every > b0 // print_every):
            print(f"  score: {b1}/{n}", flush=True)

    return L2


def score_db(cache, l2):
    """[repo] Light-weight 'db' for ``db_pair_heatmaps`` from a per-orbit L2.

    Same keys as ``eval_model_on_cache`` returns (thetas, l2, ecc, energy,
    anchor_dist, param_ranges, keys, meta) but WITHOUT the big PHI / U / PR
    arrays, so it never keeps a second copy of a 100k-orbit database alive
    (which is what ``eval_chunked``'s ``dict(cache)`` did).
    """
    small = ("thetas", "ecc", "energy", "anchor_dist", "param_ranges", "keys", "meta")
    db = {k: cache[k] for k in small if k in cache}
    db["l2"] = np.asarray(l2)
    return db
