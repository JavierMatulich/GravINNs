"""P.  Long-orbit runner -- accumulating results, selectable refinement.

Origin: cell 3 of GitHub_Longer_single_orbit.ipynb (and cells 4-8, which call
``run_ic``).  Uses the long-orbit trainer (``gravinns.training.long_orbit_trainer``).

Five initial conditions on a matched eccentricity ladder (e ~ 0.06 ... 0.64,
diverse nu / r0 / pr0), each trained over ``N_ORBITS`` orbits in two stages:

  Stage A  -- search (Stage-1 warm-start + causal/standard PINN)
  Stage B  -- refinement, resumed from Stage A's checkpoint:
                'adam'  low-LR Adam continuation (the validated default)
                'lbfgs' longer low-LR continuation (see STAGE_B_LB)
                'both'  Adam continuation, then STAGE_C_LB
                'none'  Stage A only

Results are MERGED into ``<sweep_root>/all_results.json`` keyed by IC name, so
one IC can be run at a time; re-running an IC replaces only its entry.

Usage (from the folder where ``longorbit/`` should live)::

    from gravinns.experiments.long_orbit import run_ic, show_results
    run_ic('e006', refine='adam')
    run_ic('e019', refine='adam')
    run_ic('e036', refine='adam')
    run_ic('e051', refine='adam')
    run_ic('e064', refine='adam')
    show_results()

or from the shell::

    gravinns-longorbit --ic e006 e019 e036 e051 e064 --refine adam
    gravinns-longorbit --show
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

from .. import constants as _const
from ..training.long_orbit_trainer import train_long_orbit

SWEEP_ROOT = "longorbit"
N_ORBITS   = 50
L2_TARGET  = 1e-3
SUMMARY    = os.path.join(SWEEP_ROOT, "all_results.json")

# ── the 5 ICs (matched eccentricity ladder, diverse nu / r0 / pr0) ────────
ALL_ICS = {
    "e006": dict(nu=0.24, p0_factor=0.972, r0_km=260.0, pr0_factor=0.02),
    "e019": dict(nu=0.12, p0_factor=0.902, r0_km=330.0, pr0_factor=-0.03),
    "e036": dict(nu=0.18, p0_factor=0.801, r0_km=210.0, pr0_factor=0.05),
    "e051": dict(nu=0.15, p0_factor=0.701, r0_km=300.0, pr0_factor=-0.04),
    "e064": dict(nu=0.22, p0_factor=0.600, r0_km=430.0, pr0_factor=0.03),
}

BASE_CFG = dict(
    target_pn_order    = 2,
    transfer_from_2pn  = True,
    warmstart_pn_order = '1pn',
    n_harmonic         = 10,
    n_march            = 1, march_fraction = 0.6,
    use_causal_training= True, causal_eps = 3.0, causal_fraction = 0.4,
    pretrain_epochs    = 30_000, pretrain_lr = 1e-3,
    curriculum_start   = 1_000,  curriculum_epochs = 15_000,
    w_anchor = 10.0, w_energy = 5.0,
    n_fourier = 10, fourier_sigma = 3.0,
    n_fourier_hi = 16, fourier_sigma_hi = 8.0,
    hidden = 128, depth = 5, n_colloc = 4000,
    convergence_window = 20, convergence_cv = 0.5,
    compute_l2_time = False,
    plot_every = 1000, log_every = 1_000, checkpoint_every = 10_000,
)

# stage definitions: (n_epochs, lr, lbfgs_rounds, lbfgs_cycles)
# NOTE: n_epochs is the TOTAL epoch count; Stage B resumes from Stage A's
# checkpoint, so it only trains (STAGE_B n_epochs - STAGE_A n_epochs) more,
# i.e. 150k - 100k = 50k extra epochs of low-LR Adam.
STAGE_A      = dict(n_epochs=100_000, lr=3e-3, lbfgs_rounds=0,  lbfgs_cycles=0)
STAGE_B_ADAM = dict(n_epochs=150_000, lr=1e-4, lbfgs_rounds=0,  lbfgs_cycles=0)
STAGE_B_LB   = dict(n_epochs=100_001, lr=1e-4, lbfgs_rounds=0, lbfgs_cycles=0)
STAGE_C_LB   = dict(n_epochs=150_001, lr=1e-4, lbfgs_rounds=0, lbfgs_cycles=0)

REFINE_MODES = ("adam", "lbfgs", "both", "none")


def _summary_path(sweep_root=SWEEP_ROOT):
    return SUMMARY if sweep_root == SWEEP_ROOT else os.path.join(sweep_root, "all_results.json")


def _merge_summary(new_records, path=SUMMARY):
    """Merge new results into the common file, keyed by IC name."""
    old = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                old = json.load(f)
        except Exception:
            old = []
    by_key = {r["ic_name"]: r for r in old}
    for r in new_records:
        by_key[r["ic_name"]] = r          # re-running an IC replaces its entry
    merged = sorted(by_key.values(), key=lambda r: r.get("ecc", 0.0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


def _print_table(records, l2_target=L2_TARGET, path=SUMMARY):
    print("\n" + "=" * 78)
    print(f"  ACCUMULATED LONG-ORBIT RESULTS  (N={N_ORBITS})")
    print("=" * 78)
    print(f"  {'IC':>6} {'ecc':>6} {'stage A':>10} {'refine':>10} "
          f"{'mode':>7} {'result':>9}")
    for r in records:
        a = r.get("l2_A", float('nan')); b = r.get("l2_B", float('nan'))
        print(f"  {r['ic_name']:>6} {r.get('ecc',float('nan')):6.3f} "
              f"{a:10.2e} {b:10.2e} {r.get('refine','-'):>7} "
              f"{'REACHED' if r.get('converged') else 'no':>9}")
    n_ok = sum(1 for r in records if r.get("converged"))
    print(f"\n  {n_ok}/{len(records)} reached {l2_target:g}   ->  {path}")


def run_ic(ic_name, refine="adam", sweep_root=SWEEP_ROOT, n_orbits=N_ORBITS,
           l2_target=L2_TARGET, dev=None, fresh_this_ic=True, live_plot=None):
    """Run ONE initial condition through Stage A + the chosen refinement.

    refine : 'adam' | 'lbfgs' | 'both' | 'none'
    fresh_this_ic : wipe ONLY this IC's checkpoint directory (never the others)
    live_plot : [repo] None = live figure in Jupyter only (see trainer)
    """
    if refine not in REFINE_MODES:
        raise ValueError(f"refine must be one of {REFINE_MODES}, got {refine!r}")
    ic  = ALL_ICS[ic_name]
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    run_base = os.path.join(sweep_root, f"N{n_orbits}")
    os.makedirs(run_base, exist_ok=True)
    summary = _summary_path(sweep_root)

    # the trainer's own folder name for this IC (rounded params)
    _prs = f"_pr{ic['pr0_factor']:+.2f}" if abs(ic['pr0_factor']) > 1e-6 else ""
    leaf = (f"nu{ic['nu']:.2f}_p{ic['p0_factor']:.2f}"
            f"_r{ic['r0_km']:.0f}km{_prs}")
    ic_dir = os.path.join(run_base, leaf)
    if fresh_this_ic and os.path.isdir(ic_dir):
        print(f"[fresh] removing only {ic_dir}")
        shutil.rmtree(ic_dir, ignore_errors=True)

    rec = dict(ic_name=ic_name, leaf=leaf, n_orbits=n_orbits,
               refine=refine, **ic)

    def _run(stage_name, stage):
        print(f"\n{'='*74}\n  {ic_name}  STAGE {stage_name}   "
              f"n_epochs={stage['n_epochs']:,}  lr={stage['lr']:.0e}  "
              f"lbfgs={stage['lbfgs_rounds']}x{stage['lbfgs_cycles']}\n{'='*74}")
        t0 = time.time()
        _, best, _ = train_long_orbit(
            nu=ic["nu"], p0_factor=ic["p0_factor"],
            r0_km=ic["r0_km"], pr0_factor=ic["pr0_factor"],
            n_orbits=n_orbits, l2_target=l2_target,
            base_dir=run_base, dev=dev, live_plot=live_plot,
            **BASE_CFG, **stage)
        b = float(best) if best is not None else float("nan")
        print(f"    stage {stage_name}: L2 = {b:.3e}   ({time.time()-t0:.0f}s)")
        return b

    try:
        rec["l2_A"] = _run("A-search", STAGE_A)
        if refine == "adam":
            rec["l2_B"] = _run("B-adam", STAGE_B_ADAM)
        elif refine == "lbfgs":
            rec["l2_B"] = _run("B-lbfgs", STAGE_B_LB)
        elif refine == "both":
            rec["l2_B_adam"] = _run("B-adam", STAGE_B_ADAM)
            rec["l2_B"]      = _run("C-lbfgs", STAGE_C_LB)
        else:
            rec["l2_B"] = rec["l2_A"]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec.setdefault("l2_A", float("nan"))
        rec["l2_B"] = float("nan")
        print(f"    !! {rec['error']}")

    # eccentricity for the table
    q0 = ic["r0_km"]/_const.R0_REF_KM; p0t = ic["p0_factor"]/np.sqrt(q0)
    L0 = q0*p0t; E = 0.5*(ic["pr0_factor"]**2 + p0t**2) - 1.0/q0
    rec["ecc"] = float(np.sqrt(max(0.0, 1 + 2*E*L0**2))) if E < 0 else float("nan")

    rec["l2_final"]  = rec["l2_B"]
    rec["converged"] = bool(np.isfinite(rec["l2_B"]) and rec["l2_B"] < l2_target)
    merged = _merge_summary([rec], path=summary)
    _print_table(merged, l2_target, path=summary)
    return rec


def show_results(sweep_root=SWEEP_ROOT):
    """Print the accumulated table without training anything."""
    summary = _summary_path(sweep_root)
    if not os.path.exists(summary):
        print("no results yet"); return []
    with open(summary) as f:
        recs = json.load(f)
    _print_table(recs, path=summary)
    return recs


# ── command line ──────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Long-orbit single-IC runner.")
    ap.add_argument("--ic", nargs="+", choices=list(ALL_ICS), default=None,
                    help="IC name(s) to run, in order")
    ap.add_argument("--refine", choices=REFINE_MODES, default="adam")
    ap.add_argument("--n-orbits", type=int, default=N_ORBITS)
    ap.add_argument("--sweep-root", default=SWEEP_ROOT)
    ap.add_argument("--keep", action="store_true",
                    help="do NOT wipe the IC's folder first (resume instead)")
    ap.add_argument("--show", action="store_true", help="print accumulated table")
    args = ap.parse_args(argv)
    if not args.ic and not args.show:
        ap.error("nothing to do: give --ic and/or --show")
    for name in args.ic or []:
        run_ic(name, refine=args.refine, sweep_root=args.sweep_root,
               n_orbits=args.n_orbits, fresh_this_ic=not args.keep)
    if args.show:
        show_results(args.sweep_root)


if __name__ == "__main__":
    main()
