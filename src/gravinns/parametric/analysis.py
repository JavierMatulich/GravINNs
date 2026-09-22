"""1-D L2 sweeps across one parameter of the box.

Origin: GitHub_parametric.ipynb, cell 2.
"""
import matplotlib.pyplot as plt
try:
    from IPython.display import display as _ipy_display
except ImportError:  # IPython is optional outside notebooks.
    def _ipy_display(_obj):
        return None

from .training import _measure_l2



# ── 1-D parameter sweep of the L2 error ──────────────────────────────────
def l2_sweep_1d(model, param_ranges, sweep_param,
                fixed=None,
                n_points=20,
                anchor_thetas=None,
                pde_pn_order=2, n_orbits=3, n_pts=2000,
                phi_max_global=None,
                min_anchor_dist=0.0,
                max_ecc=0.95,
                logy=True, show_ecc=True,
                device="cpu", ax=None, verbose=True):
    """L2 error (RK reference vs trained PINN) along ONE parameter axis.

    Sweeps `sweep_param` across its range in `param_ranges` while the other
    three parameters are held at fixed values. For each of the `n_points`
    thetas it integrates the true RK orbit at `pde_pn_order`, evaluates the
    PINN at the same thetas, and reports the relative 4D L2 error.

    Parameters
    ----------
    model          : the trained ParametricAnglePINN
    param_ranges   : dict name -> (lo, hi), the box the model was trained on
    sweep_param    : one of "p0_factor", "r0_km", "pr0_factor", "nu"
    fixed          : dict giving the value of the OTHER three parameters.
                     Any omitted parameter defaults to the midpoint of its
                     range. e.g. fixed=dict(r0_km=200.0, nu=0.20)
    n_points       : how many thetas along the sweep (default 20)
    anchor_thetas  : the TRAINING anchors. If given, every swept theta is
                     checked against them and the run reports how close the
                     nearest anchor is, so you can confirm the points are
                     genuinely unseen. With min_anchor_dist>0, points closer
                     than that (in normalised parameter space) are nudged off
                     the grid until they clear it.
    n_orbits,n_pts : MUST match the values used to train `model`.
    phi_max_global : shared angle domain. Defaults to model.phi_max, which is
                     what the model was trained on -- normally leave it None.
    max_ecc        : swept thetas whose Newtonian eccentricity exceeds this
                     (or which are unbound, E>=0) are skipped: their RK
                     reference would be meaningless.

    Returns
    -------
    dict with keys: values (the swept parameter), l2 (per-point error),
    thetas, mean, median, max, min, ecc, n_used
    """
    import numpy as _np
    keys = ("p0_factor", "r0_km", "pr0_factor", "nu")
    if sweep_param not in keys:
        raise ValueError(f"sweep_param must be one of {keys}, got {sweep_param!r}")
    for k in keys:
        if k not in param_ranges:
            raise KeyError(f"param_ranges missing '{k}'")

    if phi_max_global is None:
        phi_max_global = float(model.phi_max)

    # ── the three pinned parameters (midpoint unless the caller overrides) ──
    fixed = dict(fixed or {})
    if sweep_param in fixed:
        raise ValueError(f"'{sweep_param}' is the swept axis; do not pin it in `fixed`")
    pinned = {}
    for k in keys:
        if k == sweep_param:
            continue
        if k in fixed:
            v = float(fixed[k])
            lo, hi = param_ranges[k]
            if not (lo - 1e-12 <= v <= hi + 1e-12):
                raise ValueError(f"fixed['{k}']={v} lies outside param_ranges[{k}]={param_ranges[k]}")
            pinned[k] = v
        else:
            pinned[k] = 0.5*(param_ranges[k][0] + param_ranges[k][1])

    lo, hi = param_ranges[sweep_param]
    # Offset the grid off the exact endpoints/anchor lattice so the points are
    # interior samples rather than corners the anchors may already sit on.
    sweep_vals = _np.linspace(lo, hi, n_points + 2)[1:-1] if n_points > 1 \
        else _np.array([0.5*(lo + hi)])

    # ── normalised-space helper, to report distance to the nearest anchor ──
    _lo = _np.array([param_ranges[k][0] for k in keys], float)
    _hi = _np.array([param_ranges[k][1] for k in keys], float)
    _span = _np.maximum(_hi - _lo, 1e-12)
    A = None
    if anchor_thetas:
        A = _np.array([(_np.asarray(a, float) - _lo)/_span for a in anchor_thetas])

    def _nearest(th):
        if A is None:
            return _np.inf
        c = (_np.asarray(th, float) - _lo)/_span
        return float(_np.min(_np.linalg.norm(A - c, axis=1)))

    def _ecc_and_bound(th):
        p0f, r0km, pr0f, _nu = th
        q0 = r0km/200.0
        p0t = p0f/_np.sqrt(q0)
        E = 0.5*(pr0f**2 + p0t**2) - 1.0/q0
        if not (E < 0):
            return None
        L0 = q0*p0t
        return float(_np.sqrt(max(0.0, 1.0 + 2.0*E*L0**2)))

    vals, l2s, thetas, eccs, dists = [], [], [], [], []
    skipped = []
    for v in sweep_vals:
        d = dict(pinned); d[sweep_param] = float(v)
        th = tuple(d[k] for k in keys)
        e = _ecc_and_bound(th)
        if e is None or e > max_ecc:
            skipped.append((float(v), "unbound" if e is None else f"ecc={e:.3f}"))
            continue
        dist = _nearest(th)
        if min_anchor_dist > 0 and dist < min_anchor_dist:
            skipped.append((float(v), f"within {dist:.3f} of an anchor"))
            continue
        err = _measure_l2(model, th, pde_pn_order, n_orbits, n_pts,
                          phi_max_global, device)
        vals.append(float(v)); l2s.append(err); thetas.append(th)
        eccs.append(e); dists.append(dist)

    if not vals:
        raise RuntimeError("Every swept theta was skipped (unbound, too eccentric, "
                           "or too close to an anchor). Widen max_ecc or relax "
                           "min_anchor_dist.")

    vals = _np.array(vals); l2s = _np.array(l2s)
    eccs = _np.array(eccs); dists = _np.array(dists)
    mean_l2 = float(l2s.mean())

    # ── plot ───────────────────────────────────────────────────────────────
    _own_fig = ax is None
    if _own_fig:
        _fig, ax = plt.subplots(figsize=(8, 4.6))
    else:
        _fig = ax.figure
    ax.plot(vals, l2s, "o-", color="crimson", lw=1.4, ms=5, label="L² (RK vs PINN)")
    ax.axhline(mean_l2, color="navy", ls="--", lw=1.2,
               label=f"mean = {mean_l2:.3e}")
    ax.axhline(1e-3, color="0.5", ls=":", lw=1.2, label="target 1e-3")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("relative L² error")
    pin_txt = ",  ".join(f"{k}={pinned[k]:g}" for k in keys if k != sweep_param)
    ax.set_title(f"L² vs {sweep_param}   ({len(vals)} unseen orbits)\n"
                 f"fixed: {pin_txt}", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    if show_ecc:
        ax2 = ax.twinx()
        ax2.plot(vals, eccs, color="seagreen", lw=1.0, alpha=0.55)
        ax2.set_ylabel("eccentricity", color="seagreen", fontsize=8)
        ax2.tick_params(axis="y", colors="seagreen", labelsize=7)

    # This cell sets matplotlib.use("Agg") for the headless trainer, so a figure
    # created here will NOT auto-render in Jupyter. Display it explicitly.
    if _own_fig:
        _fig.tight_layout()
        try:
            _ipy_display(_fig)
        except Exception:
            pass

    if verbose:
        print(f"L2 sweep over '{sweep_param}'  ({len(vals)} orbits evaluated)")
        print(f"  fixed: {pin_txt}")
        if A is not None:
            print(f"  distance to nearest TRAINING anchor: "
                  f"min={dists.min():.3f}  median={_np.median(dists):.3f}  "
                  f"(normalised units; >0 means unseen)")
        print(f"  eccentricity range: [{eccs.min():.3f}, {eccs.max():.3f}]")
        print(f"  L2  mean   = {mean_l2:.4e}   <-- average over the set")
        print(f"      median = {_np.median(l2s):.4e}")
        print(f"      min    = {l2s.min():.4e}   at {sweep_param}={vals[l2s.argmin()]:.4g}")
        print(f"      max    = {l2s.max():.4e}   at {sweep_param}={vals[l2s.argmax()]:.4g}")
        print(f"      below 1e-3: {100*float((l2s < 1e-3).mean()):.0f}% of the sweep")
        if skipped:
            print(f"  skipped {len(skipped)} point(s): "
                  + ", ".join(f"{v:.4g} ({why})" for v, why in skipped[:5])
                  + (" ..." if len(skipped) > 5 else ""))

    return dict(values=vals, l2=l2s, thetas=thetas, ecc=eccs,
                anchor_dist=dists, mean=mean_l2, median=float(_np.median(l2s)),
                max=float(l2s.max()), min=float(l2s.min()), n_used=len(vals),
                fig=_fig, ax=ax)
