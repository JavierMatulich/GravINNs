"""K-d.  Extend every stratum of the near-circular study to n = 30.

Origin: notebook cell 4 (and cells 5-7, which call ``run_to30`` / ``report30``).

The top-up brought each stratum to 18 runs (17 in cold ``high``, one lost).
This adds new initial conditions per stratum at 2 seeds per arm.

Why 30 is worth the compute, and where it is not
------------------------------------------------
The gain is concentrated in the strata with a NONZERO failure rate. At
e~0.02 the 95% Wilson interval on 8/18 = 44% is [24%,66%]; at 30 runs a
similar rate narrows it to about [28%,61%]. At e~0.26, 1/18 = 6% has
interval [1%,26%] -- very poorly determined, and the case most improved by
more samples. For the two strata that show 0/18, going to 30 only tightens
the upper bound on an unobserved rate (0/18 excludes rates above ~18%,
0/30 above ~11%); that is a real but modest gain.

New ICs are targeted at each stratum's POOLED MEDIAN eccentricity so the rows
keep describing narrow bands rather than widening as n grows.

Usage (from the directory that holds the ``sweep_nc_*`` folders)::

    from gravinns.experiments.near_circular_n30 import run_to30, report30
    run_to30(arm_order=("cold",))       # ~1.2 h
    run_to30(arm_order=("transfer",))   # ~1.2 h
    report30()

or from the shell::

    python scripts/run_near_circular_n30.py --arm cold
    python scripts/run_near_circular_n30.py --arm transfer
    python scripts/run_near_circular_n30.py --report

All folders are resolved relative to ``results_dir`` (default: the current
working directory, exactly like the notebook).
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from ..sweep.analysis import wilson_interval as _wilson
from ..sweep.ic_generation import generate_ics_stratified
from ..sweep.runner import run_ic_sweep
from .near_circular import ARMS_NC, COMMON_NC

PN_30       = "2"
IC_SEED_30  = 987                       # distinct from 53, 91, 137
ROOT_30     = "sweep_nc_n30_final_{pn}pn_{arm}"
# every folder that already holds runs for this experiment
ROOTS_PRIOR = ["sweep_nc_{pn}pn_{arm}", "sweep_nc_topup_{pn}pn_{arm}"]

# target each stratum's pooled median so the bands stay narrow
# STRATUM_TARGETS = {"quasi_circular": 0.027, "low": 0.107, "moderate": 0.256, "high": 0.541}
STRATUM_TARGETS = {"quasi_circular": 0.027}
N_NEW_ICS_PER_STRATUM = 50

REPORT_CSV = "nearcircular_n30.csv"
ARMS = ("cold", "transfer")


def _root(fmt, arm, results_dir=None):
    """Folder for one arm.  With results_dir None/'.' the path is the bare
    relative name, identical to the notebook (it ends up in the CSV)."""
    name = fmt.format(pn=PN_30, arm=arm)
    if results_dir in (None, "", "."):
        return name
    return os.path.join(results_dir, name)


def _key(d):
    return (round(d["nu"], 4), round(d["p0_factor"], 4),
            round(d["r0_km"], 1), round(d["pr0_factor"], 4))


def build_ics_30(results_dir=None, verbose=True):
    """Generate the new ICs, one stratum at a time, and drop every IC that was
    already trained in one of the ROOTS_PRIOR folders.

    Deterministic: same IC_SEED_30 and same prior folders -> same list.  The
    n = 30 folder itself is NOT in ROOTS_PRIOR, so re-running after the cold
    arm finished gives the same ICs for the transfer arm.
    """
    ics_30 = []
    for b, e_t in STRATUM_TARGETS.items():
        got, _ = generate_ics_stratified(
            n_per_bin   = N_NEW_ICS_PER_STRATUM,
            seed        = IC_SEED_30,
            nu_range    = (0.10, 0.25),
            r0_range    = (150.0, 350.0),
            pr0_range   = (-0.003, 0.003),
            ecc_targets = (e_t,),            # single target -> tight cluster
            min_rp_rg   = 20.0,
            verbose     = verbose,
        )
        keep = [d for d in got if d["ecc_bin"] == b]
        if len(keep) < len(got) and verbose:
            print(f"  [warn] {b}: {len(got)-len(keep)} IC(s) fell outside the bin, dropped")
        ics_30 += keep

    # drop anything already trained anywhere
    existing = set()
    for fmt in ROOTS_PRIOR:
        for arm in ARMS:
            p = os.path.join(_root(fmt, arm, results_dir), "sweep_index.csv")
            if os.path.exists(p):
                e = pd.read_csv(p)
                existing |= {(round(r.nu, 4), round(r.p0_factor, 4),
                              round(r.r0_km, 1), round(r.pr0_factor, 4))
                             for r in e.itertuples()}
    ics_30 = [d for d in ics_30 if _key(d) not in existing]

    if verbose:
        print(f"\n{len(ics_30)} new ICs")
        print("  bins:", dict(Counter(d["ecc_bin"] for d in ics_30)))
        for b in STRATUM_TARGETS:
            s = [d["ecc"] for d in ics_30 if d["ecc_bin"] == b]
            if s:
                print(f"   {b:15s} ecc {min(s):.4f}-{max(s):.4f}  (target {STRATUM_TARGETS[b]})")
        n_runs = len(ics_30)*2*2
        print(f"\n  -> {n_runs} runs, roughly {n_runs*6/60:.1f} h at ~6 min each\n")
    return ics_30


# The notebook built ics_30 once, when cell 4 ran; cache it per results_dir
# so repeated run_to30() calls in one session use the very same list.
_ICS_CACHE = {}


def get_ics_30(results_dir=None, refresh=False, verbose=True):
    key = os.path.abspath(results_dir or ".")
    if refresh or key not in _ICS_CACHE:
        _ICS_CACHE[key] = build_ics_30(results_dir, verbose=verbose)
    return _ICS_CACHE[key]


def run_to30(arm_order=ARMS, results_dir=None, ics=None,
             train_overrides=None, record_loss_components=False):
    """Train the n = 30 extension, one arm after the other.

    Run cold first: it carries the failure rate the extension is meant to
    resolve, so if the session is interrupted the informative arm is done.
    The sweep is resumable (completed runs are skipped).

    train_overrides : dict merged last into the trainer kwargs (e.g. a short
                      n_epochs for a smoke test).  Leave None for the paper runs.
    """
    if isinstance(arm_order, str):
        arm_order = (arm_order,)
    if ics is None:
        ics = get_ics_30(results_dir)
    for tag in arm_order:
        arm = ARMS_NC[tag]
        root = _root(ROOT_30, tag, results_dir)
        print(f"\n{'='*70}\n  n=30 ARM: {tag}  ->  {root}\n{'='*70}")
        run_ic_sweep(
            ics, sweep_root=root, n_seeds=2, l2_target=1e-3,
            use_adaptive=False, skip_completed=True,
            train_kwargs={**COMMON_NC, **arm, **(train_overrides or {})},
            record_loss_components=record_loss_components,
        )
    print("\nDone. Analyse with:  report30()")


# ── pooled analysis over ALL folders ──────────────────────────────────────
def _load_all(arm, results_dir=None):
    frames = []
    for fmt in ROOTS_PRIOR + [ROOT_30]:
        p = os.path.join(_root(fmt, arm, results_dir), "sweep_index.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames: return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).drop_duplicates("run_id", keep="last")
    return d[d.status == "complete"]


def report30(l2_target=1e-3, results_dir=None, out_csv=REPORT_CSV):
    """Pooled near-circular table over every NC folder (prior + n = 30)."""
    order = ["quasi_circular", "low", "moderate", "high"]
    cold, warm = _load_all("cold", results_dir), _load_all("transfer", results_dir)
    print("="*92)
    print("  POOLED near-circular table (all runs)")
    print("="*92)
    if cold.empty or warm.empty:
        print("  no completed runs found for "
              + " and ".join(a for a, d in (("cold", cold), ("transfer", warm)) if d.empty)
              + f" under {os.path.abspath(results_dir or '.')}")
        return pd.DataFrame()
    print(f"  {'ecc':>7} | {'cold fail':>18} {'l_pde':>10} {'L2':>10}"
          f" | {'warm fail':>14} {'l_pde':>10} {'L2':>10}")
    rows = []
    for b in order:
        c, w = cold[cold.ecc_bin == b], warm[warm.ecc_bin == b]
        if not len(c) or not len(w): continue
        def st(s):
            l2 = s.l2_best.astype(float).values; lp = s.l_pde_final.astype(float).values
            ok = l2 <= l2_target
            g = s.groupby(["nu", "p0_factor", "r0_km", "pr0_factor"]).l2_best
            spr = g.apply(lambda x: np.nanmax(x)/(np.nanmin(x)+1e-30))
            return dict(nfail=int((~ok).sum()), n=len(l2),
                        l2=float(np.nanmedian(l2[ok])) if ok.any() else np.nan,
                        lpde=float(np.nanmedian(lp[ok])) if ok.any() else np.nan,
                        lpde_f=float(np.nanmedian(lp[~ok])) if (~ok).any() else np.nan,
                        spread=float(np.nanmedian(spr)))
        C, W = st(c), st(w)
        lo, hi = _wilson(C["nfail"], C["n"])
        ecc = float(np.median(c.ecc))
        rows.append(dict(ecc_bin=b, ecc=ecc, **{f"cold_{k}": v for k, v in C.items()},
                         cold_ci_lo=lo, cold_ci_hi=hi,
                         **{f"warm_{k}": v for k, v in W.items()}))
        print(f"  {ecc:7.3f} | {100*C['nfail']/C['n']:4.0f}% ({C['nfail']:2d}/{C['n']:2d})"
              f" [{100*lo:2.0f},{100*hi:2.0f}]% {C['lpde']:10.2e} {C['l2']:10.2e}"
              f" | {100*W['nfail']/W['n']:4.0f}% ({W['nfail']:2d}/{W['n']:2d})"
              f" {W['lpde']:10.2e} {W['l2']:10.2e}")
    print("\n  median residual of the FAILED cold runs:")
    for r in rows:
        if r["cold_nfail"] > 0:
            print(f"    e={r['ecc']:.3f}: {r['cold_lpde_f']:.2e} "
                  f"(vs {r['cold_lpde']:.2e} converged, same bin)")
    t = pd.DataFrame(rows)
    out = out_csv if results_dir in (None, "", ".") else os.path.join(results_dir, out_csv)
    t.to_csv(out, index=False)
    print(f"\n  saved {out}")
    print("  Bracketed ranges are 95% Wilson intervals on the cold failure rate.")
    return t


# ── command line ──────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Near-circular study: extend every stratum to n = 30.")
    ap.add_argument("--arm", nargs="+", choices=ARMS, default=None,
                    help="arm(s) to train, in order (e.g. --arm cold)")
    ap.add_argument("--report", action="store_true",
                    help="print / save the pooled table (report30)")
    ap.add_argument("--results-dir", default=None,
                    help="folder holding the sweep_nc_* directories (default: cwd)")
    ap.add_argument("--l2-target", type=float, default=1e-3)
    ap.add_argument("--record-loss-components", action="store_true",
                    help="log per-epoch loss channels (fills l_pde in the report)")
    args = ap.parse_args(argv)
    if not args.arm and not args.report:
        ap.error("nothing to do: give --arm and/or --report")
    if args.arm:
        run_to30(arm_order=tuple(args.arm), results_dir=args.results_dir,
                 record_loss_components=args.record_loss_components)
    if args.report:
        report30(l2_target=args.l2_target, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
