"""Trainer + monitoring for the parametric angle-PINN (Phase 1 supervised,
Phase 2 curriculum, Phase 3 PDE-only), anchor / probe selection.

Origin: GitHub_parametric.ipynb, cell 2 ('PART 3').  ``_save_panels_separately``
is the shared helper (verified identical); ``_PlotRecorder`` here is the
parametric version (different content from ``gravinns.recording.PlotRecorder``).
"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ..plotting.panels import live_display, save_panels_separately
from .model import ParametricAnglePINN
from .orbits import C_QD, C_SQ, _pham_batch_local, build_reference_set, integrate_orbit, integrate_orbit_cached, phi_residual_param



# ══════════════════════════════════════════════════════════════════════════ #
#  PART 3 — TRAINER + MONITORING for the parametric angle-PINN.                #
#                                                                              #
#  train_parametric_pinn(...) runs three phases:                              #
#    Phase 1  supervised pretrain : MSE(u,pr) to a small set of RK orbits      #
#                                   (data_pn_order, default 2PN) at K anchor   #
#                                   thetas, on a common phi grid.              #
#    Phase 2  curriculum          : loss = w_sup*alpha*L_sup                   #
#                                        + beta*(L_pde + w_energy*L_energy)    #
#                                   alpha: 1->0, beta: 0->1 over the window.   #
#    Phase 3  physics only        : chosen-PN PDE + energy over random theta   #
#                                   sampled fresh from the box each epoch.     #
#                                                                              #
#  Monitoring (live, updated every plot_every):                               #
#    (a) orbit plots for a few probe thetas (PINN vs RK)                       #
#    (b) loss-component curves                                                 #
#    (c) 2D L2-error heatmap over a choosable pair of parameters               #
# ══════════════════════════════════════════════════════════════════════════ #


_PKEYS = ("p0_factor", "r0_km", "pr0_factor", "nu")


# ── helpers ────────────────────────────────────────────────────────────────
def _sample_theta(param_ranges, n, device):
    """Uniformly sample n thetas from the box. Returns (n,4) raw tensor."""
    cols = []
    for k in _PKEYS:
        lo, hi = param_ranges[k]
        cols.append(torch.rand(n, device=device)*(hi-lo)+lo)
    return torch.stack(cols, dim=1)


def _grid_thetas(param_ranges, axes, n_side, device):
    """2D grid varying `axes` (two param names); others fixed at midpoint.
    Returns THETA (n_side*n_side,4), and the two 1-D axis arrays."""
    mids = {k: 0.5*(param_ranges[k][0]+param_ranges[k][1]) for k in _PKEYS}
    a0, a1 = axes
    v0 = np.linspace(param_ranges[a0][0], param_ranges[a0][1], n_side)
    v1 = np.linspace(param_ranges[a1][0], param_ranges[a1][1], n_side)
    TH = []
    for x1 in v1:            # rows
        for x0 in v0:        # cols
            d = dict(mids); d[a0] = x0; d[a1] = x1
            TH.append([d[k] for k in _PKEYS])
    return (torch.tensor(np.array(TH), dtype=torch.float32, device=device),
            v0, v1)



def _radial_period(orbit):
    """Angle between consecutive periapses (r minima) of an integrated orbit.

    This is the physical radial period Phi_r. The orbit is EXACTLY periodic in
    the radial phase, so 2*pi/Phi_r is the frequency the harmonic bank must sit
    at. (The precession lives in the slow phi(chi) drift, not here.)"""
    u = orbit["u"]; phi = orbit["phi"]
    r = 1.0/u
    peri = [i for i in range(1, len(r)-1) if r[i] < r[i-1] and r[i] < r[i+1]]
    if len(peri) >= 2:
        return float(np.mean(np.diff(phi[peri])))
    return 2.0*np.pi          # fallback: Newtonian


def _pretrain_omega_net(model, anchor_thetas, metas, dev,
                        epochs=12000, lr=5e-3, verbose=True):
    """Fit omega_net(theta) to the MEASURED radial frequency of each anchor.

    WHY THIS MATTERS: in the single-orbit code the harmonic bank's frequency
    was INITIALISED to the measured 2*pi/Phi_r of the reference orbit. That is
    the whole point of the bank -- harmonics locked to the true radial
    frequency represent every orbit with the same modes, so phase never
    accumulates. In the parametric version omega_net starts RANDOM (Softplus
    of small weights ~ 0.7), so the harmonics sit at the wrong frequency and
    the network must discover it from PDE gradients -- exactly the badly
    conditioned problem the bank exists to avoid. A ~20% frequency error slips
    a full radial period over ~4 revolutions, and the predicted orbit no
    longer tracks the reference.

    Fix: supervise omega_net directly on the anchors' measured frequencies
    (cheap: they are already integrated). It then interpolates across the box."""
    if model.n_harmonic <= 0:
        return
    omega_true = torch.tensor(
        [[2.0*np.pi/_radial_period(m)] for m in metas],
        dtype=torch.float32, device=dev)
    TH = torch.tensor(np.array([list(t) for t in anchor_thetas]),
                      dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.omega_net.parameters(), lr=lr)
    # Annealing matters here: at fixed lr=1e-2 the fit stalls (measured
    # 5.5e-2 max err on 96 big-box anchors at 2000 epochs; 7e-3 with
    # annealed 8000). The phase slip is fit_err * phi_max, so this error
    # multiplies directly into unseen-theta L2.
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs), eta_min=lr*1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model.omega_net(model.norm_theta(TH))
        loss = ((pred - omega_true)**2).mean()
        loss.backward(); opt.step(); sch.step()
    if verbose:
        with torch.no_grad():
            pred = model.omega_net(model.norm_theta(TH))
        err = (pred-omega_true).abs().max().item()
        print(f"  omega_net pretrained on {len(metas)} anchors: "
              f"freq range [{float(omega_true.min()):.4f},{float(omega_true.max()):.4f}], "
              f"max fit err {err:.2e}")



def make_probe_thetas(param_ranges, anchor_thetas, n_seen=2, n_unseen=2,
                      seed=1, min_dist=0.12, max_ecc=0.95, verbose=True):
    """Probe set for the live orbit plots: `n_seen` thetas taken FROM the
    supervised anchors, plus `n_unseen` thetas that are NOT anchors.

    Seeing both side by side is the point: the anchors show whether the fit is
    holding, the unseen thetas show whether the network actually GENERALISES
    across the box (which is the whole reason for a parametric PINN). A model
    can look perfect on anchors and be useless one step away from them.

    Unseen thetas are drawn from the box, validated (bound orbit, ecc<=max_ecc)
    and pushed at least `min_dist` away from every anchor in NORMALISED
    parameter space, so they are genuinely out-of-sample rather than a hair's
    breadth from a training point.

    Returns (probe_thetas, is_seen) where is_seen[i] is True for anchors.
    """
    import numpy as _np
    keys = ("p0_factor", "r0_km", "pr0_factor", "nu")
    lo = _np.array([param_ranges[k][0] for k in keys], float)
    hi = _np.array([param_ranges[k][1] for k in keys], float)
    rng = _np.random.default_rng(seed)
    span = _np.maximum(hi - lo, 1e-12)

    def _norm(t):
        return (_np.asarray(t, float) - lo)/span

    def _valid(th):
        p0f, r0km, pr0f, _nu = th
        q0 = r0km/200.0
        if q0 <= 0:
            return False
        p0t = p0f/_np.sqrt(q0)
        E = 0.5*(pr0f**2 + p0t**2) - 1.0/q0
        if not (E < 0):
            return False
        L0 = q0*p0t
        ecc = _np.sqrt(max(0.0, 1.0 + 2.0*E*L0**2))
        return ecc <= max_ecc

    n_seen = min(int(n_seen), len(anchor_thetas))
    seen = [tuple(float(x) for x in anchor_thetas[i]) for i in range(n_seen)]

    A = _np.array([_norm(a) for a in anchor_thetas]) if len(anchor_thetas) else None
    unseen = []
    for _ in range(20000):
        if len(unseen) >= int(n_unseen):
            break
        cand = lo + rng.random(4)*span
        th = tuple(float(x) for x in cand)
        if not _valid(th):
            continue
        c = _norm(th)
        if A is not None and _np.min(_np.linalg.norm(A - c, axis=1)) < min_dist:
            continue                      # too close to a training anchor
        if unseen and min(_np.linalg.norm(_norm(u) - c) for u in unseen) < min_dist:
            continue                      # too close to another probe
        unseen.append(th)

    if len(unseen) < int(n_unseen):
        raise RuntimeError(
            f"Only found {len(unseen)}/{n_unseen} unseen probes at least "
            f"min_dist={min_dist} from the {len(anchor_thetas)} anchors. Lower "
            f"min_dist, or use fewer anchors so the box is not saturated.")

    probes = seen + unseen
    is_seen = [True]*len(seen) + [False]*len(unseen)
    if verbose:
        print(f"Probe thetas: {len(seen)} SEEN (anchors) + {len(unseen)} UNSEEN")
        for th, s in zip(probes, is_seen):
            d = ("--" if s else
                 f"{_np.min(_np.linalg.norm(A - _norm(th), axis=1)):.3f}")
            print(f"  {'SEEN  ' if s else 'UNSEEN'} p0={th[0]:.3f} r0={th[1]:.1f} "
                  f"prr={th[2]:.3f} nu={th[3]:.3f}   dist-to-nearest-anchor {d}")
    return probes, is_seen


def _energy_target(theta, pn_order, c_sq, c_qd):
    """Per-sample target energy E*=H_pn(IC(theta)) at phi=0, differentiable.
    IC: q=(q0,0), p=(pr0, p0t) since at phi=0 the radial dir is x."""
    p0f = theta[:, 0]; r0km = theta[:, 1]; pr0f = theta[:, 2]; nu = theta[:, 3]
    q0 = r0km/200.0; p0t = p0f/torch.sqrt(q0); pr0 = pr0f
    q = torch.stack([q0, torch.zeros_like(q0)], dim=1)
    p = torch.stack([pr0, p0t], dim=1)
    return _pham_batch_local(q, p, pn_order, nu, c_sq, c_qd)


def _measure_l2(model, theta_row, pde_pn_order, n_orbits, n_pts,
                phi_max_global, device):
    """L2 shape error of the PINN vs an RK orbit at a single theta.
    Uses the SAME pn_order as the physics target (so we measure against the
    equation the network is solving)."""
    p0f, r0km, pr0f, nu = [float(x) for x in theta_row]
    o = integrate_orbit_cached(p0f, r0km, pr0f, nu, pde_pn_order, n_orbits, n_pts)
    phi_grid = np.linspace(0, min(o["phi_max"], phi_max_global), n_pts)
    u_ref = np.interp(phi_grid, o["phi"], o["u"])
    pr_ref = np.interp(phi_grid, o["phi"], o["pr"])
    L0 = o["L0"]
    # reference cartesian
    qx_r = np.cos(phi_grid)/u_ref; qy_r = np.sin(phi_grid)/u_ref
    px_r = pr_ref*np.cos(phi_grid) - (L0*u_ref)*np.sin(phi_grid)
    py_r = pr_ref*np.sin(phi_grid) + (L0*u_ref)*np.cos(phi_grid)
    ref = np.stack([qx_r, qy_r, px_r, py_r])
    with torch.no_grad():
        pg = torch.tensor(phi_grid, dtype=torch.float32, device=device).view(-1, 1)
        th = torch.tensor([[p0f, r0km, pr0f, nu]], dtype=torch.float32,
                          device=device).repeat(len(phi_grid), 1)
        out = model(pg, th).cpu().numpy()
    u_n = out[:, 0]; pr_n = out[:, 1]; L_n = out[:, 2]
    qx = np.cos(phi_grid)/u_n; qy = np.sin(phi_grid)/u_n
    px = pr_n*np.cos(phi_grid) - (L_n*u_n)*np.sin(phi_grid)
    py = pr_n*np.sin(phi_grid) + (L_n*u_n)*np.cos(phi_grid)
    net = np.stack([qx, qy, px, py])
    return float(np.linalg.norm(net-ref)/(np.linalg.norm(ref)+1e-16))


_PLOTREC = "plot_record.npz"   # <base_dir>/plot_record.npz -- everything the
                                # training figure draws, so it can be redrawn
                                # later WITHOUT retraining and WITHOUT torch.


class _PlotRecorder:
    """Stores everything needed to redraw the parametric-PINN training figure:
    the loss curves (sup/pde/energy/total), the 2D L2 heatmap, and the probe
    orbits (PINN vs RK reference) -- in ONE .npz file (no pickle).

    Key groups inside the file
      static_*   written once: probe thetas + SEEN/UNSEEN flags, each probe's
                 RK reference orbit (phi, u) on its own grid, heatmap axis
                 names/side, param_ranges, phase-boundary epochs, etc.
      full_*     the loss history (hist["ep"], hist["l_sup"], ...) -- an EXACT
                 copy of the very same arrays the live figure plots, kept in
                 sync every time `hist` is appended to. No independent
                 sampling, so the re-plotted loss curves are pixel-identical
                 to what training showed.
      heatmap_*  one row every time the heatmap is refreshed (every
                 `plot_every` epochs): epoch, the (side,side) log10 L2 grid,
                 and its axis values.
      orbit_*    one row every refresh: epoch, each probe's PINN prediction
                 (u) on its reference grid, and its L2 error (for the panel
                 title) -- so any probe's orbit panel can be redrawn at any
                 saved stage of training.
    """
    _GROUPS = ("heatmap", "orbit")

    def __init__(self, path):
        self.path   = path
        self.static = {}
        self.full   = {}
        self.rows   = {g: {} for g in self._GROUPS}

    def add(self, group, **vals):
        r = self.rows[group]
        for k, v in vals.items():
            r.setdefault(k, []).append(v)

    def save(self):
        out = {}
        for k, v in self.static.items():
            out["static_" + k] = np.asarray(v)
        for k, v in self.full.items():
            out["full_" + k] = np.asarray(v)
        for g in self._GROUPS:
            for k, v in self.rows[g].items():
                out[f"{g}_{k}"] = np.asarray(v)
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:                 # atomic: never a half-written file
            np.savez_compressed(f, **out)
        os.replace(tmp, self.path)



# identical (AST-checked) to the shared helper -> reuse it
_save_panels_separately = save_panels_separately


def train_parametric_pinn(
        param_ranges,                 # dict name->(lo,hi) for the 4 params
        anchor_thetas,                # list of (p0f,r0km,pr0f,nu): supervised set
        probe_thetas=None,            # thetas to plot each update
        probe_is_seen=None,           # bool per probe: True=training anchor,
                                      # False=held-out. Titles are labelled
                                      # SEEN/UNSEEN and show the live L2.
        ref_data=None,                # REMOVE
        heatmap_axes=("p0_factor", "r0_km"),  # two param names for the 2D heatmap
        pde_pn_order=2,               # 0/1/2/3 : PN order of the PHYSICS phase
        data_pn_order=2,              # PN order of the SUPERVISED orbits
        n_orbits=4,                   # 3-5 typical
        n_pts=2000,                   # points per orbit-set on the common grid
        # phase lengths (epochs)
        pretrain_epochs=20000,
        curriculum_epochs=20000,
        pde_epochs=60000,
        # loss weights
        w_sup=1.0,
        # w_energy=0 by DEFAULT. H(phi)=E* is already IMPLIED by the Hamilton
        # equations that R_u,R_pr enforce, so a separate energy loss
        # double-counts the same constraint and its gradient can fight the PDE.
        # The working 4-parameter trainer had no energy term at all. Set >0
        # only if you see the orbit drifting in radius.
        w_energy=0.0,
        # CAUSAL weighting of the PDE residual (per-theta, sorted in phi).
        # This is the ingredient that made the single-orbit trainer work:
        # a point at phi_i is only weighted once all EARLIER phi are solved,
        # which stops phase error from seeding in the outer revolutions.
        # use_causal=False by DEFAULT. Causal ordering helped the SINGLE-orbit
        # trainer over many revolutions; here each theta spans only ~3 orbits
        # and the working 4-parameter code used plain uniform phi sampling.
        # Turn it on only if you push n_orbits high.
        use_causal=False, causal_eps=3.0,
        # Supervised weight retained during the PDE phase. Default None resolves
        # by PN-order match:
        #   data_pn_order == pde_pn_order -> 1.0  (KEEP full supervision).
        #     The anchors are exact solutions of the very PDE being enforced;
        #     there is no conflict to release. And since the PDE residual is
        #     PHASE-BLIND (verified: a 0.1-rad shift leaves it unchanged), the
        #     anchors are the ONLY thing pinning the phase between them --
        #     decaying them to 5% was what created the SEEN/UNSEEN gap that the
        #     refinement step then had to undo.
        #   data_pn_order != pde_pn_order -> 0.05 (transfer case: the data is
        #     at a DIFFERENT order and genuinely conflicts with the PDE, so it
        #     must be released after warm-starting).
        alpha_floor=None,
        # sampling
        n_theta_batch=64,             # random thetas per PDE epoch
        n_colloc=1500,                # phi collocation pts per epoch
        n_sup_batch=None,             # supervised points per epoch. None =
                                      # full batch (all K*n_pts anchor points,
                                      # fine for K<~32). For MANY anchors set
                                      # e.g. 8000: a random subsample of the
                                      # anchor cloud each epoch (what the
                                      # working 4-param code did) keeps the
                                      # epoch cost flat as K grows.
        # network
        n_fourier=10, fourier_sigma=3.0,
        n_fourier_hi=16, fourier_sigma_hi=8.0,
        n_harmonic=8, hidden=128, depth=5, omega_hidden=32,
        pr_max=None,
        lr=3e-3,
        heatmap_side=11, heatmap_pn_order=None,
        plot_every=5000, log_every=1000,
        record_file=None,             # None -> <base_dir>/plot_record.npz
        base_dir="parametric_pinn", device="cpu"):

    os.makedirs(base_dir, exist_ok=True)
    dev = device
    if heatmap_pn_order is None:
        heatmap_pn_order = pde_pn_order
    if alpha_floor is None:
        alpha_floor = 1.0 if int(data_pn_order) == int(pde_pn_order) else 0.05
        print(f"  alpha_floor resolved to {alpha_floor} "
              f"(data order {'==' if alpha_floor==1.0 else '!='} PDE order)")

    # ── 1. build the supervised reference set (common phi grid) ─────────────
    if ref_data is not None:
        # Use precomputed reference tensors from the database
        print("Using precomputed reference data from database...")
        THETA = ref_data['THETA'].to(dev)
        U = ref_data['U'].to(dev)
        PR = ref_data['PR'].to(dev)
        PHI = ref_data['PHI'].to(dev)
        K = THETA.shape[0]
        phi_max_global = float(PHI[-1])   # common grid end

        # Compute u_lo, u_hi, pr_max from the precomputed data itself
        u_lo = float(U.min()) * 0.8
        u_hi = float(U.max()) * 1.2
        if pr_max is None:
            pr_max = 1.25 * float(PR.abs().max())
        # We will skip omega_net pretraining because we don't have 'metas'
        metas = None

        print(f"  {K} anchor orbits | phi_max_global={phi_max_global:.2f} rad "
              f"({phi_max_global/(2*np.pi):.2f} rev)")
        print(f"  u-band [{u_lo:.3f},{u_hi:.3f}]  pr_max={pr_max:.2f}")
    else:
        # Original code: build_reference_set by integrating
        print("Building reference set by integrating orbits...")
        metas = [integrate_orbit(*th, data_pn_order, n_orbits, n_pts)
                 for th in anchor_thetas]
        _spans = [m["phi_max"] for m in metas]
        phi_max_global = min(_spans)
        if max(_spans)/min(_spans) > 1.25:
            print(f"  [warn] anchor phi-spans vary {max(_spans)/min(_spans):.2f}x "
                  f"([{min(_spans):.1f},{max(_spans):.1f}] rad). The shared domain is "
                  f"clipped to the shortest, so wide-sweeping thetas train on fewer "
                  f"revolutions. Narrow param_ranges (esp. p0_factor/nu) if this matters.")
        THETA, U, PR, PHI, meta = build_reference_set(
            anchor_thetas, data_pn_order, n_orbits, n_pts, phi_max_global)
        THETA = THETA.to(dev)
        U = U.to(dev)
        PR = PR.to(dev)
        PHI = PHI.to(dev)
        K = THETA.shape[0]
        metas = meta
        u_lo = min(m["u_min"] for m in metas)*0.8
        u_hi = max(m["u_max"] for m in metas)*1.2
        if pr_max is None:
            pr_max = 1.25*max(m["pr_absmax"] for m in metas)
        print(f"  {K} anchor orbits | phi_max_global={phi_max_global:.2f} rad "
              f"({phi_max_global/(2*np.pi):.2f} rev)")
        print(f"  u-band [{u_lo:.3f},{u_hi:.3f}]  pr_max={pr_max:.2f}")
        
    # ── 2. model + optimiser ────────────────────────────────────────────────
    model = ParametricAnglePINN(
        phi_max=phi_max_global, param_ranges=param_ranges, u_lo=u_lo, u_hi=u_hi,
        n_fourier=n_fourier, fourier_sigma=fourier_sigma,
        n_fourier_hi=n_fourier_hi, fourier_sigma_hi=fourier_sigma_hi,
        n_harmonic=n_harmonic, hidden=hidden, depth=depth,
        pr_max=pr_max, omega_hidden=omega_hidden).to(dev)
    # Persist everything needed to RECONSTRUCT this model from a bare
    # state_dict in a fresh session. phi_max is a plain attribute (not a
    # buffer), so it is NOT inside the .pt files -- loading with the wrong
    # phi_max rescales the Fourier features and silently corrupts every
    # prediction. config.json makes loading automatic.
    import json as _json
    with open(os.path.join(base_dir, "config.json"), "w") as _f:
        _json.dump(dict(phi_max=float(phi_max_global),
                        param_ranges={k: list(v) for k, v in param_ranges.items()},
                        u_lo=float(u_lo), u_hi=float(u_hi), pr_max=float(pr_max),
                        n_fourier=int(n_fourier), fourier_sigma=float(fourier_sigma),
                        n_fourier_hi=int(n_fourier_hi),
                        fourier_sigma_hi=float(fourier_sigma_hi),
                        n_harmonic=int(n_harmonic), hidden=int(hidden),
                        depth=int(depth), omega_hidden=int(omega_hidden)), _f,
                   indent=1)

    # Lock the harmonic bank onto the TRUE radial frequency before any
    # training. Without this, omega_net starts random and the harmonics sit at
    # the wrong frequency -- the orbit then slips a full radial period over a
    # few revolutions and never tracks the reference.
    # More anchors need more omega-fit epochs: 2000 was tuned for ~8-32
    # anchors; with 128 on a wide box it left a 1.5e-2 frequency error
    # (= 0.27 rad phase slip over the domain).
    # Lock the harmonic bank onto the TRUE radial frequency.
    # If we are using precomputed ref_data, we skip this because we don't have 'metas'.
    if ref_data is None:
        _pretrain_omega_net(model, anchor_thetas, metas, dev,
                            epochs=max(3000, 60*len(anchor_thetas)))
    else:
        print("  omega_net pretraining skipped (using precomputed ref_data).")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine-annealed LR, exactly as the working 4-parameter trainer.
    # WITHOUT this, Adam at a constant lr=3e-3 bounces around the minimum for
    # the whole run: the loss "oscillates around a horizontal line" and never
    # settles. Annealing to lr*1e-3 lets it actually descend into the basin.
    total_epochs = pretrain_epochs + curriculum_epochs + pde_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_epochs), eta_min=lr*1e-3)

    if probe_thetas is None:
        # Default to 2 training anchors + 2 held-out thetas, so the plots show
        # fit AND generalisation rather than fit alone.
        try:
            probe_thetas, probe_is_seen = make_probe_thetas(
                param_ranges, anchor_thetas, n_seen=2, n_unseen=2, verbose=False)
        except Exception:
            probe_thetas = list(anchor_thetas[:min(4, len(anchor_thetas))])
            probe_is_seen = [True]*len(probe_thetas)
    if probe_is_seen is None:
        _aset = {tuple(round(float(x), 10) for x in a) for a in anchor_thetas}
        probe_is_seen = [tuple(round(float(x), 10) for x in t) in _aset
                         for t in probe_thetas]

    hist = dict(ep=[], l_sup=[], l_pde=[], l_energy=[], l_total=[])
    heat_hist = []

    # supervised phi grid expanded per-sample (K, n_pts) -> flat batches
    PHI_col = PHI.view(1, -1, 1).repeat(K, 1, 1)          # (K,n_pts,1)

    # ── PLOT RECORD: static part ─────────────────────────────────────────── #
    # Everything the live figure draws is written to <base_dir>/plot_record.npz
    # so it can be redrawn later without retraining. Recording only EVALUATES
    # the network / integrates cached reference orbits, so it never changes
    # the training trajectory.
    _rec_path = record_file or os.path.join(base_dir, _PLOTREC)
    _rec = _PlotRecorder(_rec_path)
    n_probe = len(probe_thetas)
    _probe_phi_ref = np.zeros((n_probe, n_pts))
    _probe_u_ref   = np.zeros((n_probe, n_pts))
    for _i, _th in enumerate(probe_thetas):
        _o = integrate_orbit_cached(*_th, pde_pn_order, n_orbits, n_pts)
        _pg = np.linspace(0, min(_o["phi_max"], phi_max_global), n_pts)
        _probe_phi_ref[_i] = _pg
        _probe_u_ref[_i]   = np.interp(_pg, _o["phi"], _o["u"])
    _rec.static.update(
        probe_thetas=np.asarray(probe_thetas, dtype=np.float64),   # (n_probe,4)
        probe_is_seen=np.asarray(probe_is_seen, dtype=bool),
        probe_phi_ref=_probe_phi_ref, probe_u_ref=_probe_u_ref,
        pde_pn_order=int(pde_pn_order), data_pn_order=int(data_pn_order),
        n_orbits=int(n_orbits), n_pts=int(n_pts),
        phi_max_global=float(phi_max_global),
        heatmap_axes=np.array(list(heatmap_axes)),
        heatmap_side=int(heatmap_side), heatmap_pn_order=int(heatmap_pn_order),
        param_range_names=np.array(list(param_ranges.keys())),
        param_range_values=np.array(list(param_ranges.values()), dtype=np.float64),
        pretrain_epochs=int(pretrain_epochs), curriculum_epochs=int(curriculum_epochs),
        pde_epochs=int(pde_epochs), total_epochs=int(total_epochs),
        w_sup=float(w_sup), w_energy=float(w_energy),
        plot_every=int(plot_every), log_every=int(log_every),
        base_dir=str(base_dir),
    )
    _rec.save()
    print(f"  Plot record → {_rec_path}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, max(n_probe, 3))
    ax_orbit = [fig.add_subplot(gs[0, i]) for i in range(n_probe)]
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])
    # dedicated, fixed axes for the heatmap colorbar (so it is never re-added)
    from mpl_toolkits.axes_grid1 import make_axes_locatable as _mal
    _cb = {"obj": None, "cax": _mal(ax_heat).append_axes("right", size="4%", pad=0.05)}
    disp = live_display(fig, None)   # [repo] live in Jupyter only

    def _refresh(ep):
        for a in ax_orbit: a.clear()
        ax_loss.clear(); ax_heat.clear()
        _orbit_u_pred = []; _orbit_l2 = []
        # orbits
        for a, th, _seen in zip(ax_orbit, probe_thetas, probe_is_seen):
            o = integrate_orbit_cached(*th, pde_pn_order, n_orbits, n_pts)
            pg = np.linspace(0, min(o["phi_max"], phi_max_global), n_pts)
            with torch.no_grad():
                P = torch.tensor(pg, dtype=torch.float32, device=dev).view(-1, 1)
                T = torch.tensor([list(th)], dtype=torch.float32,
                                 device=dev).repeat(len(pg), 1)
                out = model(P, T).cpu().numpy()
            ur = np.interp(pg, o["phi"], o["u"])
            a.plot(np.cos(pg)/ur, np.sin(pg)/ur, color="0.6", lw=1.2,
                   label=f"{pde_pn_order}PN RK")
            a.plot(np.cos(pg)/out[:, 0], np.sin(pg)/out[:, 0], "r", lw=0.9,
                   label="PINN")
            a.plot(0, 0, "k*", ms=8)
            a.set_aspect("equal")
            _l2 = _measure_l2(model, th, pde_pn_order, n_orbits, n_pts,
                              phi_max_global, dev)
            _orbit_u_pred.append(out[:, 0].astype(np.float32)); _orbit_l2.append(float(_l2))
            _tag = "SEEN (anchor)" if _seen else "UNSEEN (held-out)"
            a.set_title(f"{_tag}   L2={_l2:.2e}\n"
                        f"p0={th[0]:.2f} r0={th[1]:.0f} "
                        f"prr={th[2]:.2f} nu={th[3]:.2f}",
                        fontsize=7,
                        color=("black" if _seen else "darkred"))
            a.tick_params(labelsize=6)
        ax_orbit[0].legend(fontsize=6, loc="upper right")
        # loss
        if hist["ep"]:
            ax_loss.semilogy(hist["ep"], np.maximum(hist["l_sup"], 1e-12), label="sup")
            ax_loss.semilogy(hist["ep"], np.maximum(hist["l_pde"], 1e-12), label="pde")
            if w_energy > 0:      # inert when w_energy=0; plotting it misleads
                ax_loss.semilogy(hist["ep"], np.maximum(hist["l_energy"], 1e-12),
                                 label="energy")
            ax_loss.semilogy(hist["ep"], np.maximum(hist["l_total"], 1e-12), "k", label="total")
        ax_loss.axvline(pretrain_epochs, color="b", ls=":", lw=0.8)
        ax_loss.axvline(pretrain_epochs+curriculum_epochs, color="g", ls=":", lw=0.8)
        ax_loss.set_title("Loss", fontsize=8); ax_loss.legend(fontsize=6)
        ax_loss.set_xlabel("epoch", fontsize=7); ax_loss.tick_params(labelsize=6)
        # heatmap
        TH_g, v0, v1 = _grid_thetas(param_ranges, heatmap_axes, heatmap_side, dev)
        errs = np.array([
            _measure_l2(model, TH_g[i].cpu(), heatmap_pn_order, n_orbits,
                        n_pts, phi_max_global, dev)
            for i in range(TH_g.shape[0])]).reshape(heatmap_side, heatmap_side)
        im = ax_heat.imshow(np.log10(errs+1e-12), origin="lower", aspect="auto",
                            extent=[v0[0], v0[-1], v1[0], v1[-1]], cmap="viridis")
        ax_heat.set_xlabel(heatmap_axes[0], fontsize=7)
        ax_heat.set_ylabel(heatmap_axes[1], fontsize=7)
        ax_heat.set_title(f"log10 L2 @ ep{ep} (median {np.median(errs):.1e})",
                          fontsize=7)
        ax_heat.tick_params(labelsize=6)
        # Create the colorbar exactly ONCE. Calling fig.colorbar() on every
        # refresh appends a new colorbar axes each time, so after N redraws
        # the figure accumulates N stale bars beside the live one.
        if _cb["obj"] is None:
            _cb["obj"] = fig.colorbar(im, cax=_cb["cax"])
        else:
            _cb["obj"].update_normal(im)
        heat_hist.append((ep, errs, v0, v1))
        # ── PLOT RECORD: one row per refresh (heatmap + all probe orbits) ──
        _rec.add("heatmap", epoch=int(ep), errs=errs.astype(np.float64),
                 v0=np.asarray(v0, dtype=np.float64), v1=np.asarray(v1, dtype=np.float64))
        _rec.add("orbit", epoch=int(ep), u_pred=np.stack(_orbit_u_pred),
                 l2=np.asarray(_orbit_l2, dtype=np.float64))
        _rec.save()
        _med = float(np.median(errs))
        # phi_max is a plain python float on the model, not a buffer, so it is
        # NOT part of state_dict() and would otherwise be lost on reload.
        # Save it alongside every checkpoint.
        with open(os.path.join(base_dir, "phi_max.txt"), "w") as _f:
            _f.write(repr(float(phi_max_global)))   # plain python float, not np.float64
        torch.save(model.state_dict(), os.path.join(base_dir, "param_pinn_last.pt"))
        if _med < _best["med"]:
            _best["med"] = _med; _best["ep"] = ep
            torch.save(model.state_dict(),
                       os.path.join(base_dir, "param_pinn_best.pt"))
        fig.suptitle(f"Parametric PINN | ep {ep}/{total_epochs} | "
                     f"phases: pre {pretrain_epochs} / cur {curriculum_epochs} "
                     f"/ pde {pde_epochs}", fontsize=9)
        fig.tight_layout()
        disp.update(fig)

    # ── training loop ────────────────────────────────────────────────────────
    # Checkpointing: every heatmap refresh, save (a) the CURRENT weights to
    # param_pinn_last.pt (crash safety) and (b) the weights with the lowest
    # heatmap-median L2 so far to param_pinn_best.pt. The final weights are
    # still written to param_pinn.pt at the end, as before.
    _best = {"med": float("inf"), "ep": -1}
    pbar = tqdm(range(total_epochs), desc="param-PINN")
    for ep in pbar:
        opt.zero_grad()
        _ZERO = torch.zeros((), device=dev)
        l_sup = _ZERO
        l_pde = _ZERO
        l_energy = _ZERO

        # phase schedule
        if ep < pretrain_epochs:
            phase = "pre"; alpha = 1.0; beta = 0.0
        elif ep < pretrain_epochs+curriculum_epochs:
            phase = "cur"
            t = (ep-pretrain_epochs)/max(1, curriculum_epochs)
            alpha = (1.0-t) + t*alpha_floor      # decay 1 -> alpha_floor
            beta = t
        else:
            phase = "pde"; alpha = alpha_floor; beta = 1.0

        # supervised term (uses the precomputed anchor set)
        if alpha > 0:
            if n_sup_batch is None:
                phi_b = PHI_col.view(-1, 1)                   # (K*n_pts,1)
                th_b = THETA.view(K, 1, 4).repeat(1, PHI.shape[0], 1).view(-1, 4)
                out = model(phi_b, th_b)
                u_p = out[:, 0].view(K, -1); pr_p = out[:, 1].view(K, -1)
                l_sup = ((u_p-U)**2).mean() + ((pr_p-PR)**2).mean()
            else:
                # random subsample of the (anchor, point) cloud -- epoch cost
                # is O(n_sup_batch) regardless of how many anchors K there are.
                Npt = PHI.shape[0]
                idx = torch.randint(0, K*Npt, (int(n_sup_batch),), device=dev)
                ia = idx // Npt; ip = idx % Npt
                phi_b = PHI[ip].view(-1, 1)
                th_b = THETA[ia]
                out = model(phi_b, th_b)
                l_sup = (((out[:, 0]-U[ia, ip])**2).mean()
                         + ((out[:, 1]-PR[ia, ip])**2).mean())

        # physics term (random thetas, chosen PN order)
        if beta > 0:
            if use_causal and causal_eps > 0:
                # Causal ordering needs a phi-ladder PER theta, so group the
                # points. NOTE this costs phi-resolution: n_colloc/n_theta_batch
                # points per orbit. Only worth it for many revolutions.
                n_th = int(n_theta_batch)
                n_ph = max(2, int(n_colloc)//n_th)
                th_u = _sample_theta(param_ranges, n_th, dev)
                ph = torch.rand(n_th, n_ph, device=dev)*phi_max_global
                ph, _ = torch.sort(ph, dim=1)
                th_r = th_u.repeat_interleave(n_ph, dim=0)
                phi_r = ph.reshape(-1, 1).requires_grad_(True)
            else:
                # DEFAULT: plain Monte-Carlo over the whole (phi, theta) domain,
                # ONE INDEPENDENT theta per collocation point -- what the working
                # 4-parameter trainer did. Grouping 64 thetas x 23 phi (the old
                # behaviour) gave only ~5 phi-points per revolution, so the
                # gradient was dominated by sampling noise and the loss just
                # oscillated. Independent sampling measurably lowers the
                # gradient noise for the same n_colloc.
                n_th = int(n_colloc); n_ph = 1
                th_r = _sample_theta(param_ranges, int(n_colloc), dev)
                phi_r = (torch.rand(int(n_colloc), 1, device=dev)
                         * phi_max_global).requires_grad_(True)
            R_u, R_pr, Hcol = phi_residual_param(
                model, phi_r, th_r, pde_pn_order, C_SQ, C_QD)
            r_pt = (R_u**2 + R_pr**2).view(n_th, n_ph)   # (n_th,n_ph); (N,1) if iid
            # Only build the energy term when it is actually weighted. With
            # w_energy=0 (the default) H(phi)=E* is already implied by the
            # Hamilton equations R_u,R_pr, so computing it just wastes an
            # autograd graph -- and plotting it makes a term with ZERO gradient
            # look as though it were training the model.
            if w_energy > 0:
                E_star = _energy_target(th_r, pde_pn_order, C_SQ, C_QD)
                e_pt = ((Hcol - E_star)**2).view(n_th, n_ph)
            else:
                e_pt = None

            if use_causal and causal_eps > 0:
                # w_i = exp(-eps * cum_norm_i), cum_norm in [0,1] along phi.
                # Normalised by the per-theta total so eps is independent of
                # phi_max and of the residual scale (the fix that made the
                # single-orbit causal weighting work at many orbits).
                cum = torch.cumsum(r_pt.detach(), dim=1)
                cum = torch.cat([torch.zeros(n_th, 1, device=dev), cum[:, :-1]], 1)
                cum = cum/(cum[:, -1:] + 1e-12)
                w = torch.exp(-causal_eps*cum).detach()
                l_pde = (w*r_pt).mean()
                l_energy = (w*e_pt).mean() if e_pt is not None else _ZERO
            else:
                l_pde = r_pt.mean()
                l_energy = e_pt.mean() if e_pt is not None else _ZERO

        loss = w_sup*alpha*l_sup + beta*(l_pde + w_energy*l_energy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # working code used 1.0
        opt.step()
        sched.step()

        if ep % log_every == 0 or ep == total_epochs-1:
            hist["ep"].append(ep)
            hist["l_sup"].append(float(l_sup))
            hist["l_pde"].append(float(l_pde))
            hist["l_energy"].append(float(l_energy))
            hist["l_total"].append(float(loss))
            # exact copy of hist -- flushed to disk at the next refresh, so
            # the re-plotted loss curves match the live figure exactly
            _rec.full["loss_epoch"]  = np.asarray(hist["ep"], dtype=np.int64)
            _rec.full["loss_sup"]    = np.asarray(hist["l_sup"], dtype=np.float64)
            _rec.full["loss_pde"]    = np.asarray(hist["l_pde"], dtype=np.float64)
            _rec.full["loss_energy"] = np.asarray(hist["l_energy"], dtype=np.float64)
            _rec.full["loss_total"]  = np.asarray(hist["l_total"], dtype=np.float64)
            _pf = dict(phase=phase, sup=f"{float(l_sup):.1e}",
                       pde=f"{float(l_pde):.1e}",
                       lr=f"{opt.param_groups[0]['lr']:.1e}")
            if w_energy > 0:
                _pf["en"] = f"{float(l_energy):.1e}"
            pbar.set_postfix(**_pf)
        if ep % plot_every == 0 or ep == total_epochs-1:
            _refresh(ep)

    torch.save(model.state_dict(), os.path.join(base_dir, "param_pinn.pt"))
    # sidecar config: everything needed to reconstruct the model in a fresh
    # session (phi_max is NOT in the state_dict -- it is a plain float, and it
    # scales the Fourier features, so loading with the wrong value silently
    # distorts every prediction).
    import json as _json
    with open(os.path.join(base_dir, "param_pinn_config.json"), "w") as _f:
        _json.dump(dict(phi_max_global=float(phi_max_global),
                        param_ranges={k: list(v) for k, v in param_ranges.items()},
                        n_orbits=int(n_orbits), n_pts=int(n_pts),
                        pde_pn_order=int(pde_pn_order),
                        data_pn_order=int(data_pn_order)), _f, indent=2)
    fig.savefig(os.path.join(base_dir, "training_summary.png"), dpi=110)

    # ── convergence history, persisted ───────────────────────────────────
    # Without this, the only convergence evidence a run leaves behind is the
    # epoch of the best checkpoint. That cannot distinguish "the model had
    # converged" from "the cosine schedule ran out", which is the first thing
    # a comparison between two architectures has to establish: a ranking at a
    # fixed budget is only meaningful if it is stable over the trajectory.
    _hh = []
    for _e in heat_hist:
        _ep, _errs = _e[0], np.asarray(_e[1], dtype=float)
        _hh.append(dict(epoch=int(_ep),
                        median=float(np.median(_errs)),
                        mean=float(np.mean(_errs)),
                        p90=float(np.percentile(_errs, 90)),
                        max=float(np.max(_errs)),
                        frac_below_1e3=float((_errs < 1e-3).mean())))
    with open(os.path.join(base_dir, "heat_hist.json"), "w") as _f:
        _json.dump(_hh, _f, indent=2)

    # the per-epoch loss channels, as plotted in the summary figure
    _lh = {}
    if isinstance(hist, dict):
        for _k, _v in hist.items():
            try:
                _lh[_k] = [float(x) for x in np.asarray(_v).ravel()]
            except Exception:
                pass
    with open(os.path.join(base_dir, "loss_hist.json"), "w") as _f:
        _json.dump(_lh, _f)

    # the run's own settings, so a curve can be attributed later
    with open(os.path.join(base_dir, "run_settings.json"), "w") as _f:
        _json.dump(dict(total_epochs=int(total_epochs),
                        pretrain_epochs=int(pretrain_epochs),
                        curriculum_epochs=int(curriculum_epochs),
                        pde_epochs=int(pde_epochs),
                        alpha_floor=float(alpha_floor),
                        w_sup=float(w_sup), w_energy=float(w_energy),
                        n_anchors=int(K), n_sup_batch=int(n_sup_batch or 0),
                        n_colloc=int(n_colloc),
                        hidden=int(hidden), depth=int(depth),
                        n_harmonic=int(n_harmonic), lr=float(lr),
                        best_epoch=int(_best["ep"]),
                        best_median=float(_best["med"])), _f, indent=2)
    print(f"  heat_hist.json      convergence history "
          f"({len(_hh)} points)")
    print(f"\nDONE. Saved to {base_dir}/:")
    print(f"  param_pinn.pt       final weights (what `model` holds)")
    if _best["ep"] >= 0:
        print(f"  param_pinn_best.pt  lowest heatmap-median L2 = "
              f"{_best['med']:.3e} at ep {_best['ep']}")
        print(f"  param_pinn_last.pt  latest refresh checkpoint")
        if _best["ep"] < total_epochs - plot_every:
            print(f"  NOTE: best was NOT the final epoch -- consider loading "
                  f"param_pinn_best.pt")
    return model, hist, heat_hist


# ── Automatic anchor-theta generation ──────────────────────────────────────
def make_anchor_thetas(param_ranges, n_anchors=8, include_corners=False,
                       seed=0, max_ecc=0.95, verbose=True):
    """Automatically generate `n_anchors` anchor thetas covering `param_ranges`.

    Uses a Latin-Hypercube design (each parameter is stratified into n_anchors
    bins, one sample per bin) so the small anchor set spans every parameter's
    range instead of clumping the way uniform-random sampling does.

    Parameters
    ----------
    param_ranges   : dict name -> (lo, hi) for p0_factor, r0_km, pr0_factor, nu
    n_anchors      : how many anchor thetas to return (the choosable amount)
    include_corners: if True, prepend the 2^4=16 box corners (then fill the
                     remainder with LHS points). Corners are where a parametric
                     PINN extrapolates worst, so anchoring them helps — but it
                     costs 16 supervised orbits. Only worth it for n_anchors>=20.
    seed           : RNG seed for reproducibility
    max_ecc        : reject thetas whose Newtonian eccentricity exceeds this
                     (very eccentric orbits have a razor-sharp periapsis that
                     dominates the supervised loss)
    verbose        : print a coverage summary

    Every candidate is validated: the orbit must be BOUND (E<0) and have
    ecc<=max_ecc. Rejected points are resampled, so the returned list always
    contains n_anchors physically usable thetas.

    Returns
    -------
    list of (p0_factor, r0_km, pr0_factor, nu) tuples
    """
    import itertools
    import numpy as _np
    keys = ("p0_factor", "r0_km", "pr0_factor", "nu")
    for k in keys:
        if k not in param_ranges:
            raise KeyError(f"param_ranges missing '{k}'")
    lo = _np.array([param_ranges[k][0] for k in keys], dtype=float)
    hi = _np.array([param_ranges[k][1] for k in keys], dtype=float)
    if _np.any(hi < lo):
        raise ValueError("param_ranges has hi < lo for some parameter")
    rng = _np.random.default_rng(seed)

    def _valid(th):
        """Bound orbit with acceptable eccentricity?  th = (p0f, r0km, pr0f, nu)."""
        p0f, r0km, pr0f, _nu = th
        q0 = r0km/200.0
        if q0 <= 0:
            return False
        p0t = p0f/_np.sqrt(q0)
        E = 0.5*(pr0f**2 + p0t**2) - 1.0/q0
        if not (E < 0):                     # unbound -> RK reference is garbage
            return False
        L0 = q0*p0t
        ecc = _np.sqrt(max(0.0, 1.0 + 2.0*E*L0**2))
        return ecc <= max_ecc

    picked = []

    # ── optional: the 2^4 box corners ─────────────────────────────────────
    if include_corners:
        for c in itertools.product(*[(param_ranges[k][0], param_ranges[k][1])
                                     for k in keys]):
            if len(picked) >= n_anchors:
                break
            if _valid(c):
                picked.append(tuple(float(x) for x in c))

    # ── Latin-Hypercube fill for the remainder ────────────────────────────
    n_fill = max(0, n_anchors - len(picked))
    attempts = 0
    while n_fill > 0 and attempts < 50:
        attempts += 1
        # stratify each dim into n_fill bins, one draw per bin, shuffled per dim
        U = _np.empty((n_fill, 4))
        for j in range(4):
            perm = rng.permutation(n_fill)
            U[:, j] = (perm + rng.random(n_fill))/n_fill
        cand = lo + U*(hi - lo)
        for row in cand:
            if len(picked) >= n_anchors:
                break
            th = tuple(float(x) for x in row)
            if _valid(th):
                picked.append(th)
        n_fill = n_anchors - len(picked)

    if len(picked) < n_anchors:
        raise RuntimeError(
            f"Only found {len(picked)}/{n_anchors} valid anchors after {attempts} "
            f"LHS rounds. The box likely contains many unbound orbits (E>=0) or "
            f"orbits with ecc>{max_ecc}. Narrow param_ranges (lower p0_factor or "
            f"r0_km) or raise max_ecc.")

    picked = picked[:n_anchors]

    if verbose:
        arr = _np.array(picked)
        print(f"Generated {len(picked)} anchor thetas"
              f"{' (corners included)' if include_corners else ''}:")
        for k, j in zip(keys, range(4)):
            frac = (arr[:, j] - lo[j])/(hi[j] - lo[j] + 1e-12)
            s = _np.sort(frac)
            s = _np.concatenate([[0.0], s, [1.0]])
            print(f"  {k:<11} range [{arr[:,j].min():.4g}, {arr[:,j].max():.4g}]"
                  f"  max normalised gap {_np.diff(s).max():.3f}")
        eccs = []
        for (p0f, r0km, pr0f, _nu) in picked:
            q0 = r0km/200.0; p0t = p0f/_np.sqrt(q0)
            E = 0.5*(pr0f**2+p0t**2)-1.0/q0; L0 = q0*p0t
            eccs.append(_np.sqrt(max(0.0, 1+2*E*L0**2)))
        print(f"  eccentricity across anchors: "
              f"[{min(eccs):.3f}, {max(eccs):.3f}]  (all bound)")
    return picked
