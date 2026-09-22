"""Offline analysis of finished runs (reads artefacts only, no GPU).

Origin: section B.3 of notebook cell 3, plus ``_wilson`` from cell 4.
"""
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
#  B.3  Post-run analysis of one trained model
# ---------------------------------------------------------------------------
#  Everything here is read back from the saved artefacts, so it can be re-run
#  offline on an old sweep without touching the GPU.
# ---------------------------------------------------------------------------

def summarize_run(ckpt_dir, l2_target=1e-3):
    """Extract convergence statistics from a completed run directory."""
    s = dict(l2_final=float("nan"), l2_best=float("nan"),
             epochs_to_target=None, converged=False,
             loss_final=float("nan"), l_pde_final=float("nan"),
             l_energy_final=float("nan"), l_pde_median_last10pct=float("nan"),
             n_epochs_run=0, l2_plateau_ratio=float("nan"),
             l2_monotone_frac=float("nan"))

    hp = os.path.join(ckpt_dir, "history.json")
    if os.path.exists(hp):
        with open(hp) as f:
            h = json.load(f)
        l2s = np.asarray(h.get("l2_shape", []), dtype=float)
        eps = np.asarray(h.get("l2_epochs", []), dtype=float)
        loss = np.asarray(h.get("loss", []), dtype=float)
        if l2s.size:
            s["l2_final"] = float(l2s[-1])
            s["l2_best"]  = float(np.nanmin(l2s))
            s["converged"] = bool(np.nanmin(l2s) < l2_target)
            hit = np.where(l2s < l2_target)[0]
            if hit.size and eps.size >= hit[0] + 1:
                s["epochs_to_target"] = int(eps[hit[0]])
            # Plateau ratio: how much of the total improvement happened in the
            # last half of training. ~0 means the run stalled early - the
            # signature of a local minimum rather than slow convergence.
            if l2s.size >= 4:
                mid = l2s.size // 2
                tot = l2s[0] - np.nanmin(l2s)
                late = l2s[mid] - np.nanmin(l2s)
                s["l2_plateau_ratio"] = float(late / (tot + 1e-30))
                d = np.diff(l2s)
                s["l2_monotone_frac"] = float(np.mean(d < 0))
        if loss.size:
            s["loss_final"] = float(loss[-1])
            s["n_epochs_run"] = int(loss.size)

    cp = os.path.join(ckpt_dir, "loss_components.npz")
    if os.path.exists(cp):
        z = np.load(cp)
        if "l_pde" in z:
            lp = z["l_pde"]
            s["l_pde_final"] = float(lp[-1])
            k = max(1, int(0.10 * lp.size))
            s["l_pde_median_last10pct"] = float(np.nanmedian(lp[-k:]))
        if "l_energy" in z:
            s["l_energy_final"] = float(z["l_energy"][-1])
    return s


def classify_failure(summary, l2_target=1e-3, residual_quantile=None):
    """Assign a mechanism label to a run.

    'ok'                 : reached target.
    'spurious_minimum'   : residual converged low but L2 stayed high. This is
                           the near-circular / phase-degeneracy signature -
                           the optimizer solved the equations and still got
                           the wrong orbit.
    'unconverged'        : residual itself never came down; the optimization
                           is still limited by stiffness (expected at high
                           eccentricity, where the periapsis residual
                           dominates).
    'stalled_early'      : L2 improvement all happened in the first half and
                           then flattened - consistent with a basin the
                           optimizer cannot leave.
    """
    if summary.get("converged"):
        return "ok"
    lpde = summary.get("l_pde_final", float("nan"))
    thr  = residual_quantile
    if thr is not None and np.isfinite(lpde):
        if lpde < thr:
            return "spurious_minimum"
        return "unconverged"
    # Fallback when no cross-run residual scale is available yet
    pr = summary.get("l2_plateau_ratio", float("nan"))
    if np.isfinite(pr) and pr < 0.10:
        return "stalled_early"
    return "unconverged"


def wilson_interval(k, n, z=1.96):
    """95% (z=1.96) Wilson score interval for k successes out of n."""
    if n == 0: return (np.nan, np.nan)
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    hw = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (max(0.0, c-hw), min(1.0, c+hw))


_wilson = wilson_interval   # notebook name
