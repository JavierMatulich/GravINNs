"""Statistics for the paper tables (Tables IV, V, VI): score a model once on a
large test cache, then stratify its L2 by eccentricity and periapsis.

The routines accept explicit cache and device arguments and contain the
fixed-edge joint stratification used for the manuscript tables.  The local
``periapsis_rg`` definition is the one used for those reported statistics.
"""
import gc
import os

import numpy as np
import torch

from .benchmark import score_on_cache
from .checkpoints import load_parametric_model

__all__ = ["score_once", "joint_table", "variance_decomposition",
           "compare_models", "joint_table_2d", "compare_models_joint",
           "periapsis_rg", "cache_strata"]


def cache_strata(cache):
    """ecc, thetas, periapsis (r_p / r_g) and nu of every orbit in a cache."""
    ecc = np.asarray(cache["ecc"])
    th  = np.asarray(cache["thetas"])
    rp  = periapsis_rg(th, ecc)
    nu  = th[:, 3]
    return ecc, th, rp, nu


# ── score each model in turn, then immediately free it ────────────────────
def score_once(model_path, which, tag, cache, device=None, batch_size=16,
               out_dir="."):
    """Load ONE model, score it on ``cache``, save ``<out_dir>/l2_<tag>.npy``,
    free the model, return the per-orbit L2."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_parametric_model(model_path, which=which, device=device)
    l2    = score_on_cache(model, cache, batch_size=batch_size, device=device,
                           verbose=True)
    np.save(os.path.join(out_dir, f"l2_{tag}.npy"), l2)   # persist so we never recompute
    # ---- CRITICAL: drop the model from GPU/CPU before loading the next one
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return l2


# ── joint tables, variance decomposition, comparison ───────────────────────
def joint_table(l2, ecc, rp, label, bins_ecc=(0,.3,.5,.65,.8,1.01),
                bins_rg=None, tol=1e-3):
    l2 = np.asarray(l2, float); ecc = np.asarray(ecc, float)
    rp = np.asarray(rp, float)
    if bins_rg is None:
        bins_rg = np.quantile(rp, [0, .25, .5, .75, 1.0])
        bins_rg[0] -= 1e-9; bins_rg[-1] += 1e-9
    print(f"\n--- {label} ---")
    print(f"{'ecc bin':>14}{'n':>7}{'median L2':>14}{'< 1e-3':>9}")
    for a, b in zip(bins_ecc[:-1], bins_ecc[1:]):
        s = (ecc >= a) & (ecc < b)
        if not s.sum(): continue
        print(f"[{a:.2f},{b:.2f}){int(s.sum()):>7}{np.median(l2[s]):>14.3e}"
              f"{100*(l2[s] < tol).mean():>8.1f}%")
    print(f"{'r_p/r_g':>14}{'n':>7}{'median L2':>14}{'< 1e-3':>9}")
    for a, b in zip(bins_rg[:-1], bins_rg[1:]):
        s = (rp >= a) & (rp < b)
        if not s.sum(): continue
        print(f"[{a:6.1f},{b:6.1f}){int(s.sum()):>7}"
              f"{np.median(l2[s]):>14.3e}{100*(l2[s] < tol).mean():>8.1f}%")


def variance_decomposition(l2, ecc, rp, nu):
    """How much of log10(L2) is explained by each parameter?"""
    y = np.log10(np.maximum(l2, 1e-12))
    for name, x in (("ecc", ecc), ("r_p/r_g", rp), ("nu", nu)):
        c = np.corrcoef(x, y)[0, 1]
        print(f"  corr(log10 L2, {name:8s}) = {c:+.3f}")


def compare_models(a, b, ecc, rp, name_a="A", name_b="B", tol=1e-3):
    a = np.asarray(a, float); b = np.asarray(b, float)
    assert len(a) == len(b)
    print(f"\n{'':<12}{'median':>12}{'mean':>12}{'max':>12}{'< 1e-3':>9}")
    for nm, x in ((name_a, a), (name_b, b)):
        print(f"{nm:<12}{np.median(x):>12.3e}{x.mean():>12.3e}"
              f"{x.max():>12.3e}{100*(x < tol).mean():>8.1f}%")
    print(f"{name_b} better on {100*(b < a).mean():.1f}% of the same orbits "
          f"| median ratio {name_b}/{name_a} = {np.median(b/a):.3f}")


# ============================================================================
#  Joint (eccentricity x periapsis) analysis
#
#  joint_table() bins ecc and r_p INDEPENDENTLY -- two separate tables, not one
#  joint grid -- so it cannot show whether a given cell's error comes from
#  eccentricity or from periapsis; the fixed-edge TWO-DIMENSIONAL grid below
#  is built to break exactly that confound.
# ============================================================================
def joint_table_2d(l2, ecc, rp, label="", tol=1e-3, min_n=100,
                   e_edges=(0, 0.3, 0.5, 0.65, 0.8),
                   r_edges=(0, 10, 20, 40, np.inf)):
    """ONE grid: rows = eccentricity, columns = periapsis, each cell is both
    at once. Read along a ROW at fixed eccentricity to see whether periapsis
    moves the error; read down a COLUMN at fixed periapsis to see whether
    eccentricity does. Cells below min_n are shown as "--".
    """
    l2, ecc, rp = np.asarray(l2, float), np.asarray(ecc, float), np.asarray(rp, float)
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


def compare_models_joint(l2_a, l2_b, ecc, rp, name_a="A", name_b="B",
                         tol=1e-3, min_n=100,
                         e_edges=(0, 0.3, 0.5, 0.65, 0.8),
                         r_edges=(0, 10, 20, 40, np.inf)):
    """name_b minus name_a, percentage points below tolerance, per joint cell.
    Positive = name_b better in that cell.
    """
    l2_a, l2_b = np.asarray(l2_a, float), np.asarray(l2_b, float)
    ecc, rp = np.asarray(ecc, float), np.asarray(rp, float)
    print("=" * 78)
    print(f"  {name_b} minus {name_a}, percentage points below {tol:g}")
    print("=" * 78)
    hdr = f"{'e / r_p':>12}"
    for j in range(len(r_edges)-1):
        hi = r_edges[j+1]
        hdr += f"{f'[{r_edges[j]:g},{hi:g})' if np.isfinite(hi) else f'[{r_edges[j]:g},inf)':>14}"
    print(hdr)
    for i in range(len(e_edges)-1):
        row = f"{f'[{e_edges[i]},{e_edges[i+1]})':>12}"
        for j in range(len(r_edges)-1):
            m = ((ecc >= e_edges[i]) & (ecc < e_edges[i+1]) &
                 (rp >= r_edges[j]) & (rp < r_edges[j+1]))
            n = int(m.sum())
            if n < min_n:
                row += f"{'--':>14}"
            else:
                d = 100*((l2_b[m] < tol).mean() - (l2_a[m] < tol).mean())
                row += f"{d:+13.0f}"
        print(row)
    print()


# ── helper: periapsis in gravitational radii ────────────────────────────────
def periapsis_rg(thetas, ecc):
    """r_p / r_g for a 1.4+1.4 M_sun system at each theta."""
    G = 6.67430e-11; c = 299792458.0; Msun = 1.989e30
    M = 2.8 * Msun
    r_g_km = G * M / c**2 / 1000.0
    p0f = thetas[:, 0]; r0km = thetas[:, 1]; pr0f = thetas[:, 2]
    q0  = r0km / 200.0
    p0t = p0f / np.sqrt(q0)
    E   = 0.5*(pr0f**2 + p0t**2) - 1.0/q0
    L0  = q0*p0t
    a   = -0.5/E
    rp  = a*(1.0 - np.asarray(ecc))            # in R0 units
    return (rp * 200.0) / r_g_km               # in r_g units
