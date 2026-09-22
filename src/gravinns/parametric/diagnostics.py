"""Disentangling tests: does eccentricity or periapsis control the error?
Memory-safe helpers for 100k-orbit caches.

Origin: GitHub_parametric.ipynb, cells 18, 19 and 35.  Cell 19's
``omega_accuracy`` replaced cell 18's in the notebook, so only it is kept.
"""
import gc
import os

import numpy as np



def periapsis_rg(thetas, ecc, R0=200.0, c_norm=6.954):
    q0  = thetas[:, 1] / R0
    p0t = thetas[:, 0] / np.sqrt(q0)
    E   = 0.5*(thetas[:, 2]**2 + p0t**2) - 1.0/q0
    return (-0.5/E) * (1.0 - ecc) * c_norm**2


# ---------------------------------------------------------------------------
# TEST 1: joint stratification. The decisive one.
# ---------------------------------------------------------------------------
def joint_table(l2, ecc, rp, tol=1e-3, min_n=150,
                e_edges=(0, 0.3, 0.5, 0.65, 0.8),
                r_edges=(0, 10, 20, 40, np.inf), label=""):
    """Rows = eccentricity, columns = periapsis.

    Read it twice:
      * along a ROW (fixed eccentricity, varying periapsis): if the error
        moves, periapsis matters at fixed eccentricity.
      * down a COLUMN (fixed periapsis, varying eccentricity): if the error
        moves, eccentricity matters at fixed periapsis.
    Whichever direction varies more is the controlling variable. Cells with
    fewer than `min_n` orbits are shown as "--" rather than quoted, since a
    percentage from a handful of orbits is not meaningful.
    """
    print("=" * 78)
    print(f"  JOINT STRATIFICATION{(' -- ' + label) if label else ''}"
          f"   (fraction below {tol:g}; n in parentheses)")
    print("=" * 78)
    hdr = f"{'e / r_p':>12}"
    for j in range(len(r_edges)-1):
        hi = r_edges[j+1]
        hdr += f"{f'[{r_edges[j]:g},{hi:g})' if np.isfinite(hi) else f'[{r_edges[j]:g},inf)':>16}"
    print(hdr)
    for i in range(len(e_edges)-1):
        row = f"{f'[{e_edges[i]},{e_edges[i+1]})':>12}"
        for j in range(len(r_edges)-1):
            m = ((ecc >= e_edges[i]) & (ecc < e_edges[i+1]) &
                 (rp >= r_edges[j]) & (rp < r_edges[j+1]))
            n = int(m.sum())
            row += f"{'--':>16}" if n < min_n else \
                   f"{f'{100*(l2[m]<tol).mean():.0f}% ({n})':>16}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# TEST 2: which predictor carries the variance
# ---------------------------------------------------------------------------
def variance_decomposition(l2, ecc, rp, nu=None):
    """Regress log10(L2) on the candidate predictors.

    Standardised coefficients are comparable to one another, so the largest
    in magnitude is the predictor the error actually follows. Reported with
    the partial correlation, which is the association of each predictor with
    the error once the others are held fixed -- exactly what the 1D tables
    cannot show.
    """
    y = np.log10(np.asarray(l2, float))
    cols = {"log10_rp": np.log10(np.asarray(rp, float)),
            "ecc": np.asarray(ecc, float)}
    if nu is not None:
        cols["nu"] = np.asarray(nu, float)
    good = np.isfinite(y) & np.all([np.isfinite(v) for v in cols.values()], axis=0)
    y = y[good]
    X = np.column_stack([v[good] for v in cols.values()])
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    beta, *_ = np.linalg.lstsq(np.column_stack([Xs, np.ones(len(ys))]), ys, rcond=None)
    print("=" * 78)
    print("  VARIANCE DECOMPOSITION of log10(L2)   (standardised coefficients)")
    print("=" * 78)
    for name, b in zip(cols, beta[:-1]):
        print(f"    {name:>12}: {b:+.3f}")
    pred = np.column_stack([Xs, np.ones(len(ys))]) @ beta
    print(f"    {'R^2':>12}: {1 - ((ys-pred)**2).sum()/((ys-ys.mean())**2).sum():.3f}")
    print("    the largest |coefficient| is the variable the error follows\n")


# ---------------------------------------------------------------------------
# TEST 3: is it the loss or the problem?
# ---------------------------------------------------------------------------
def compare_models(l2_pinn, l2_sup, ecc, rp, tol=1e-3, min_n=150):
    """Same joint cells, difference between the two models.

    A positive entry means the supervised model does better in that cell. The
    physics loss is implicated wherever the supervised model -- which never
    sees the residual -- is the more accurate of the two, since both were
    trained on the same references with the same architecture.
    """
    print("=" * 78)
    print("  SUPERVISED minus PINN, percentage points below tolerance")
    print("  (positive = supervised better = the residual is hurting there)")
    print("=" * 78)
    e_ed = (0, 0.3, 0.5, 0.65, 0.8); r_ed = (0, 10, 20, 40, np.inf)
    hdr = f"{'e / r_p':>12}"
    for j in range(4):
        hi = r_ed[j+1]
        hdr += f"{f'[{r_ed[j]:g},{hi:g})' if np.isfinite(hi) else f'[{r_ed[j]:g},inf)':>14}"
    print(hdr)
    for i in range(4):
        row = f"{f'[{e_ed[i]},{e_ed[i+1]})':>12}"
        for j in range(4):
            m = ((ecc >= e_ed[i]) & (ecc < e_ed[i+1]) &
                 (rp >= r_ed[j]) & (rp < r_ed[j+1]))
            n = int(m.sum())
            if n < min_n:
                row += f"{'--':>14}"
            else:
                d = 100*((l2_sup[m] < tol).mean() - (l2_pinn[m] < tol).mean())
                row += f"{d:+13.0f}"
        print(row)
    print()


def is_lazy(cache):
    """True if indexing this cache decompresses a whole array each time."""
    return isinstance(cache, np.lib.npyio.NpzFile)


def materialise(cache, keys=("PHI", "U", "PR")):
    """Pull the big arrays into memory ONCE.

    Returns a plain dict. Do this before any per-orbit loop; indexing the
    NpzFile itself inside a loop is what makes the kernel die.
    """
    out = {}
    for k in keys:
        if k in cache:
            out[k] = np.asarray(cache[k])
            print(f"  materialised {k}: {out[k].shape} "
                  f"{out[k].nbytes/1e9:.2f} GB")
    return out


def score_once(model, cache, name, device="cuda", out_dir="."):
    """Evaluate ONE model, save its per-orbit L2, and free everything.

    Avoids holding two models' evaluation intermediates alive at the same
    time, which is what the back-to-back `l2_pinn = ...; l2_sup = ...`
    pattern does.
    """
    path = os.path.join(out_dir, f"l2_{name}.npy")
    if os.path.exists(path):
        print(f"  {name}: cached -> {path}")
        return np.load(path)
    from .benchmark import eval_model_on_cache    # [repo] was: from __main__ import ...
    db = eval_model_on_cache(model, cache, device=device)
    l2 = np.asarray(db["l2"], dtype=np.float32)
    np.save(path, l2)
    del db
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"  {name}: {len(l2)} orbits -> {path}")
    return l2


def omega_accuracy(model, cache, rp, device="cuda", n_sample=2000, seed=0,
                   omega_attr=None, arrays=None, batch=256):
    """Relative error of the learned radial frequency, stratified by periapsis.

    Memory-safe rewrite: the reference arrays are materialised once and the
    network is evaluated in batches rather than one orbit at a time.

    The harmonic bank rests on omega(theta), and a fractional error eps in it
    produces a phase error growing as eps*phi_max. Strong-field orbits precess
    most, so their Phi_r departs furthest from 2*pi and the auxiliary network
    must extrapolate hardest there. If this error tracks the periapsis cliff,
    the strong-field degradation is a property of THIS architecture rather
    than of the post-Newtonian expansion -- and unlike the other explanations,
    it would be fixable.
    """
    import torch

    if arrays is None:
        if is_lazy(cache):
            print("  [materialising cache arrays first -- see module docstring]")
        arrays = materialise(cache, keys=("PHI", "U"))
    PHI, U = arrays["PHI"], arrays["U"]

    # locate the omega sub-network without guessing its name
    if omega_attr is None:
        for cand in ("omega_net", "omega_mlp", "net_omega", "omega"):
            if hasattr(model, cand):
                omega_attr = cand
                break
        else:
            names = [n for n, _ in model.named_children()]
            raise AttributeError(
                f"could not find the omega sub-network; children are {names}. "
                f"Pass omega_attr=... explicitly.")
    omega_net = getattr(model, omega_attr)
    print(f"  using model.{omega_attr}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(PHI), min(n_sample, len(PHI)), replace=False)

    # measured Phi_r per orbit, from periapsis spacing (vectorised per orbit,
    # but reading from the in-memory arrays, not the compressed archive)
    keep, Phi_true = [], []
    for i in idx:
        u = U[i]
        pk = np.where((u[1:-1] > u[:-2]) & (u[1:-1] > u[2:]))[0] + 1
        if len(pk) < 3:
            continue
        keep.append(i)
        Phi_true.append(float(np.polyfit(np.arange(len(pk)), PHI[i][pk], 1)[0]))
    keep = np.array(keep); Phi_true = np.array(Phi_true)

    th = np.asarray(cache["thetas"], dtype=np.float32)[keep]
    om_pred = []
    with torch.no_grad():
        for s in range(0, len(th), batch):
            t = torch.tensor(th[s:s+batch], device=device)
            om_pred.append(np.asarray(omega_net(t).squeeze(-1).cpu()).ravel())
    om_pred = np.concatenate(om_pred)
    om_true = 2*np.pi/Phi_true
    err = np.abs(om_pred - om_true)/om_true
    rp_k = np.asarray(rp)[keep]

    print("=" * 78)
    print("  LEARNED RADIAL FREQUENCY: relative error in omega(theta)")
    print("=" * 78)
    print(f"    {'r_p/r_g':>12}{'n':>7}{'median |domega/omega|':>24}"
          f"{'implied phase slip':>20}")
    phi_max = float(getattr(model, "phi_max", 55.46))
    for lo, hi, lab in ((0,10,'[0,10)'), (10,20,'[10,20)'),
                        (20,40,'[20,40)'), (40,np.inf,'[40,inf)')):
        m = (rp_k >= lo) & (rp_k < hi)
        if m.sum() < 20:
            continue
        e = float(np.median(err[m]))
        print(f"    {lab:>12}{int(m.sum()):7d}{e:24.2e}"
              f"{e*phi_max:19.2e} rad")
    print("\n    A sharp rise at small r_p would make the harmonic frequency")
    print("    map a distinct, fixable cause of the strong-field cliff.\n")
    return err, rp_k


def memory_report(tag=""):
    """Print resident memory, so a crash can be anticipated rather than hit."""
    try:
        import psutil
        rss = psutil.Process().memory_info().rss/1e9
        avail = psutil.virtual_memory().available/1e9
        print(f"  [{tag}] RSS {rss:.2f} GB | available {avail:.2f} GB")
    except ImportError:
        print(f"  [{tag}] (pip install psutil for memory reporting)")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [{tag}] CUDA allocated "
                  f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
    except Exception:
        pass


def subsample_cache(cache, n_keep=30000, seed=0, verbose=True):
    """A random subset of the benchmark, as a plain dict.

    Only entries that are per-orbit arrays are sliced. Metadata such as
    cache["meta"] is a dict and must be passed through untouched -- calling
    np.asarray on it produces a 0-d object array that can no longer be
    indexed by key, which breaks eval_model_on_cache downstream.
    """
    keys = cache.files if hasattr(cache, "files") else list(cache.keys())
    n = len(np.asarray(cache["thetas"]))
    n_keep = min(n_keep, n)
    idx = np.sort(np.random.default_rng(seed).choice(n, n_keep, replace=False))
    out, sliced = {}, []
    for k in keys:
        v = cache[k]
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[idx]; sliced.append(k)
        else:
            out[k] = v                      # dicts, scalars, metadata
    if verbose:
        big = sum(v.nbytes for v in out.values() if isinstance(v, np.ndarray))
        print(f"  subsampled {n} -> {n_keep} orbits ({big/1e9:.2f} GB); "
              f"sliced {sliced}")
    return out, idx
