"""Resumable sweep driver: one PINN per (initial condition, seed).

Origin: section B.4 of notebook cell 3.  See the module docstring of
``gravinns.sweep`` for the on-disk layout.
"""
import csv
import json
import os
import platform
import time

import numpy as np
import torch

from .. import constants as _const
from ..recording import loss_components as _loss_components
from ..training.near_circular_trainer import train_near_circular
from .analysis import summarize_run


# ---------------------------------------------------------------------------
#  B.4  The sweep driver
# ---------------------------------------------------------------------------

_SWEEP_CSV_FIELDS = [
    "run_id", "ic_index", "seed", "status",
    "nu", "p0_factor", "r0_km", "pr0_factor",
    "ecc", "ecc_bin", "r_peri", "rp_over_rg", "rp_bin", "x_peri",
    "a", "T_orb", "L0", "E_newt", "pn_class",
    "regime", "warmstart", "target_pn_order",
    "l2_final", "l2_best", "converged", "epochs_to_target",
    "loss_final", "l_pde_final", "l_energy_final",
    "l_pde_median_last10pct", "l2_plateau_ratio", "l2_monotone_frac",
    "n_epochs_run", "wall_time_s", "ckpt_dir",
]


def _run_id(d, seed):
    return (f"nu{d['nu']:.3f}_p{d['p0_factor']:.3f}"
            f"_r{d['r0_km']:.0f}_pr{d['pr0_factor']:+.3f}_s{seed}")


def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SWEEP_CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def run_ic_sweep(ics,
                 sweep_root="single_orbit_sweep",
                 n_seeds=1,
                 base_seed=1234,
                 use_adaptive=False,
                 l2_target=1e-3,
                 skip_completed=True,
                 train_kwargs=None,
                 verbose=True,
                 record_loss_components=False,
                 adaptive_trainer=None):
    """Train one model per (IC, seed) and persist everything.

    ics          : list of diagnostic dicts from generate_ics()
    n_seeds      : independent initializations per IC. n_seeds >= 3 is what
                   makes the local-minimum claim statistical rather than
                   anecdotal - it measures the SCATTER of the landscape.
    use_adaptive : route through train_orbit_adaptive (PN-regime diagnosis)
                   instead of calling train_near_circular directly.
    train_kwargs : forwarded verbatim to the trainer.
    record_loss_components : [repo] install the per-epoch loss-channel
                   recorder (fills l_pde_final / l_energy_final).  Default
                   False reproduces the notebook, where LOSS_REC was absent.
    adaptive_trainer : [repo] callable used when use_adaptive=True; defaults
                   to ``gravinns.training.train_orbit_adaptive``.
    """
    if record_loss_components:
        _loss_components.install_loss_recorder()
    if adaptive_trainer is None:          # [repo] the dissipative case supplies it
        from ..training.adaptive import train_orbit_adaptive as adaptive_trainer
    train_orbit_adaptive = adaptive_trainer
    train_kwargs = dict(train_kwargs or {})
    os.makedirs(sweep_root, exist_ok=True)
    csv_path = os.path.join(sweep_root, "sweep_index.csv")

    manifest = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_ics=len(ics), n_seeds=n_seeds, base_seed=base_seed,
        l2_target=l2_target, use_adaptive=use_adaptive,
        train_kwargs={k: (v if isinstance(v, (int, float, str, bool, type(None)))
                          else str(v)) for k, v in train_kwargs.items()},
        environment=dict(python=platform.python_version(),
                         torch=torch.__version__,
                         numpy=np.__version__,
                         device=("cuda" if torch.cuda.is_available() else "cpu"),
                         gpu=(torch.cuda.get_device_name(0)
                              if torch.cuda.is_available() else None)),
        constants=dict(C_SQ=float(_const.C_SQ), R0_REF_KM=float(_const.R0_REF_KM)),
    )
    with open(os.path.join(sweep_root, "sweep_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    results, t_sweep = [], time.time()
    total = len(ics) * n_seeds
    done = 0

    for d in ics:
        for si in range(n_seeds):
            seed = base_seed + 977 * si
            rid  = _run_id(d, seed)
            ckpt_dir = os.path.join(sweep_root, rid)
            rec_path = os.path.join(ckpt_dir, "run_record.json")
            done += 1

            if skip_completed and os.path.exists(rec_path):
                try:
                    with open(rec_path) as f:
                        prev = json.load(f)
                    if prev.get("status") == "complete":
                        if verbose:
                            print(f"[{done}/{total}] SKIP (done) {rid}")
                        results.append(prev)
                        continue
                except Exception:
                    pass

            os.makedirs(ckpt_dir, exist_ok=True)
            if verbose:
                print(f"\n[{done}/{total}] {rid}  "
                      f"ecc={d['ecc']:.3f} rp={d['rp_over_rg']:.1f} r_g "
                      f"({d['pn_class']})")

            # Independent initialization per seed - this is what makes the
            # seed scatter a measurement of the loss landscape.
            torch.manual_seed(seed)
            np.random.seed(seed)

            _lrec = _loss_components.LOSS_REC     # None unless installed
            if _lrec is not None:
                _lrec.reset()
                _lrec.active = True

            rec = dict(run_id=rid, seed=seed, ckpt_dir=ckpt_dir,
                       status="running", **{k: v for k, v in d.items()})
            rec["l2_target"] = l2_target
            with open(rec_path, "w") as f:
                json.dump(rec, f, indent=2)

            t0 = time.time()
            try:
                kw = dict(train_kwargs)
                kw["base_dir"] = sweep_root
                # NOTE: l2_target is passed explicitly to each trainer below,
                # never via kw, to avoid a duplicate-keyword TypeError.
                if use_adaptive:
                    # train_orbit_adaptive CHOOSES these per regime; if the
                    # user left them in train_kwargs they would collide.
                    for _k in ("warmstart_pn_order", "target_pn_order",
                               "transfer_from_2pn"):
                        if _k in kw:
                            if done == 1 and verbose:
                                print(f"    [adaptive] ignoring user '{_k}'"
                                      f" (regime-selected automatically)")
                            kw.pop(_k)
                    diag, best_l2s, _ = train_orbit_adaptive(
                        nu=d["nu"], p0_factor=d["p0_factor"],
                        r0_km=d["r0_km"], pr0_factor=d["pr0_factor"],
                        l2_target=l2_target, run_tag=rid, auto_resume=False, seed=seed, **kw)
                    rec["regime"] = diag.get("regime")
                    rec["warmstart"] = diag.get("warmstart_pn_order")
                    rec["target_pn_order"] = diag.get("target_pn_order")
                    rec["gap_12"] = diag.get("gap_12")
                    rec["gap_23"] = diag.get("gap_23")
                    if diag.get("regime") == "STOP":
                        rec["status"] = "skipped_stop"
                        rec["stop_reason"] = diag.get("reason")
                else:
                    _, best_l2s, _ = train_near_circular(
                        nu=d["nu"], p0_factor=d["p0_factor"],
                        r0_km=d["r0_km"], pr0_factor=d["pr0_factor"],
                        l2_target=l2_target,
                        run_tag=rid, auto_resume=False, seed=seed, **kw)
                    rec["regime"] = "fixed"
                    rec["warmstart"] = kw.get("warmstart_pn_order", "auto")
                    rec["target_pn_order"] = kw.get("target_pn_order", 2)
                rec["best_l2_returned"] = (float(best_l2s)
                                           if best_l2s is not None else None)
                if rec["status"] == "running":
                    rec["status"] = "complete"
            except Exception as e:
                rec["status"] = "failed"
                rec["error"] = f"{type(e).__name__}: {e}"
                if verbose:
                    print(f"    !! {rec['error']}")

            rec["wall_time_s"] = float(time.time() - t0)

            # With run_tag=rid the trainer writes directly into ckpt_dir,
            # so the artefacts are exactly where this harness expects them.
            art_dir = ckpt_dir
            rec["artifact_dir"] = art_dir

            if _lrec is not None:
                _lrec.active = False
                p = _lrec.save(art_dir)
                if p:
                    rec["loss_components"] = os.path.basename(p)

            rec.update(summarize_run(art_dir, l2_target=l2_target))
            with open(rec_path, "w") as f:
                json.dump(rec, f, indent=2)
            _append_csv(csv_path, rec)
            results.append(rec)

            if verbose and rec["status"] == "complete":
                print(f"    L2_best={rec['l2_best']:.3e}  "
                      f"conv={rec['converged']}  "
                      f"l_pde={rec['l_pde_final']:.2e}  "
                      f"{rec['wall_time_s']:.0f}s")

    if verbose:
        print(f"\nSweep finished: {len(results)} runs in "
              f"{(time.time()-t_sweep)/60:.1f} min -> {sweep_root}")
    return results
