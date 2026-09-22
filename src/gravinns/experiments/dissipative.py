"""Dissipative (2.5PN inspiral) survey: every orbit of the paper.

Origin: GitHub_Dissipative.ipynb, cells 4-5, and instructions_dissipative.txt.

24 runs, one configuration: 14 at six radial periods and 10 at ten.  P1..P14
are the published rows (their measured L2 is recorded in ``IC_SET`` as metadata
only), D1..D10 extend the analysis.  Only the initial data and ``n_orbits``
change between runs, which is what makes the survey a one-factor study.

    from gravinns.experiments.dissipative import COMMON, IC_SET, show_ic_set, run_case
    show_ic_set()                     # the table of the 24 runs
    run_case("D1")                    # train one case, by tag
    COMMON["n_epochs"] = 50_000       # COMMON is a plain dict: edit before running

Results go to ``<base_dir>/<tag>/<regime folder>/`` (default ``diss_all``).
``paper_table()`` rebuilds the paper's table from those folders.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from .. import constants as _const
from ..models.angle_pinn import AngleOrbitPINN4, reconstruct_qp4
from ..physics.hamiltonian import compute_hamiltonian_r
from ..training.adaptive import diagnose_pn_regime, train_orbit_adaptive

BASE_DIR = "diss_all"

# ═══════════════════════════════════════════════════════════════════════════
#  EVERY DISSIPATIVE ORBIT IN THE PAPER — one place, one configuration
#
#  24 runs: 14 at six radial periods, 10 at ten. Ordered as in the draft
#  table -- published rows first (P1..P14, in table order), then the new
#  runs that extend the analysis (D1..D10).
#
#    P1..P14  already run; their measured L2 is recorded below so this file
#             is the single source of truth. Re-run only to reproduce.
#    D1..D10  to run. Chosen to fix three specific weaknesses, see NOTES.
#
#  nu AND pr0 VARY BY RUN, so they live in each entry rather than in COMMON.
#  r0 is given in km (= 200 x the dimensionless r0 of the draft table).
#
#  Everything else is identical across all 24 runs, which is what makes the
#  survey a one-factor study: only the initial data and n_orbits change.
#
#  ── VERIFICATION ───────────────────────────────────────────────────────
#  Every published row was re-derived from its (nu, p0, r0, pr0) and matches
#  the draft table on e0, N_rev, r_p/r_g and gap to within rounding (14/14).
#  P7 correctly reproduces the post-Newtonian truncation at 4.12 periods.
#
#  ── NOTES ON THE NEW RUNS ──────────────────────────────────────────────
#  1. HIGH-END LEVERAGE. The fitted slope of log10(L2) vs e0*N_rev rests on
#     ONE published point above e0*N_rev = 5.3 (P14). Dropping it shifts the
#     slope by 9%, so the upper end is not established. D5, D6, D8, D9, D10
#     add five points from 5.33 to 8.33 and extend the range past P14.
#
#  2. PRODUCT vs FACTORS. That e0*N_rev governs the error -- rather than e0
#     and N_rev separately -- is currently untested. Matched-product pairs
#     at different splits make it falsifiable:
#         D2 (e0=0.51 @  6.5 rev)  vs  D7 (e0=0.36 @ 10.7 rev)
#         D5 (e0=0.75 @  7.4 rev)  vs  D8 (e0=0.58 @ 10.6 rev)
#         D6 (e0=0.70 @  7.6 rev)  vs  D9 (e0=0.70 @ 10.7 rev)
#     Similar L2 within a pair supports the product; divergence refutes it.
#
#  3. FIELD STRENGTH IS CONFOUNDED. Across P1..P14, corr(e0, r_p/r_g) =
#     -0.69, and the point supporting "accuracy is not set by field strength"
#     is P7 -- one of the two EXCLUDED runs. D3 and D4 break the degeneracy
#     at nearly equal r_p (16.6 vs 15.2) with products differing 3x
#     (1.58 vs 4.61): the product hypothesis predicts D3 accurate and D4
#     poor, a field-strength account predicts them similar.
#     Adding all ten moves the correlation from -0.69 to -0.54.
#
#  Also note D4 and D10 share initial conditions and differ ONLY in
#  n_orbits (6 vs 10): a clean one-factor test of the N_rev dependence at
#  fixed e0. The tag in base_dir keeps their output folders apart.
#
#  ── PREDICTIONS (so the test is falsifiable in advance) ────────────────
#  From log10(L2) = -5.76 + 0.606 e0*N_rev, fitted to the 12 included
#  published runs (r = +0.94):
#      D3 ~ 7e-5   D1 ~ 2e-4   D2 ~ 4e-4   D7 ~ 9e-4   D4 ~ 3e-3
#      D6 ~ 1e-2   D5 ~ 1e-2   D8 ~ 3e-2   D9 ~ 6e-2   D10 ~ 1.5e-1
#  The high-product runs are PREDICTED to be poor; that is deliberate. They
#  test the trend, and their captured fractions bound where the method stops
#  working -- more informative than ten more successes where data exist.
#
#   id  n_orb    nu    p0   r0(km)  pr0     e0   Nrev  e0Nrev  rp/rg    gap    L2
#  --- 6 radial periods ---
#   P1      6  0.25  0.83     260  0.00  0.311   6.50   2.023   25.2 1.4e-02  5.95e-06
#   P2      6  0.25  0.88     300  0.00  0.226   6.36   1.436   37.4 5.5e-03  8.20e-06
#   P3      6  0.16  0.85     140  0.00  0.278   7.38   2.047   15.0 6.6e-02  9.86e-06
#   P4      6  0.24  0.72     250  0.17  0.501   6.97   3.489   13.9 5.1e-02  3.00e-04
#   P5      6  0.22  0.75     180  0.00  0.437   8.38   3.664   11.8 1.1e-01  4.66e-04
#   P6      6  0.25  0.65     320  0.00  0.577   7.49   4.326   13.1 5.6e-02  1.55e-03
#   P7      6  0.14  0.65     220  0.00  0.577   4.12   2.380    9.0 5.6e-02  9.45e-05
#   P8      6  0.07  1.20     210  0.30  0.574   7.70   4.421   29.5 1.4e-03  1.28e-05
#   D1      6  0.25  0.80     200  0.00  0.360   6.92   2.491   16.7 4.5e-02    to run
#   D2      6  0.25  0.70     400  0.00  0.510   6.46   3.292   20.8 1.6e-02    to run
#   D3      6  0.25  0.90     120  0.00  0.190   8.29   1.575   16.6 1.1e-01    to run
#   D4      6  0.25  0.55     600  0.00  0.697   6.60   4.605   15.2 3.7e-02    to run
#   D5      6  0.25  0.50     700  0.00  0.750   7.38   5.531   13.8 5.3e-02    to run
#   D6      6  0.25  0.55     500  0.00  0.698   7.63   5.325   12.7 6.0e-02    to run
#  --- 10 radial periods ---
#   P9     10  0.25  0.92     220  0.00  0.154  10.94   1.681   33.8 1.7e-02  1.54e-05
#  P10     10  0.22  0.95     140  0.00  0.098  12.23   1.192   25.4 5.7e-02  4.47e-05
#  P11     10  0.24  0.75     320  0.00  0.437  10.93   4.783   21.1 2.9e-02  1.04e-03
#  P12     10  0.16  0.85     140  0.00  0.278  13.69   3.799   15.0 1.3e-01  1.08e-03
#  P13     10  0.25  0.75     260  0.00  0.438  12.00   5.252   17.1 6.1e-02  2.64e-03
#  P14     10  0.25  0.65     380  0.00  0.578  12.03   6.946   15.6 6.1e-02  1.46e-02
#   D7     10  0.25  0.80     320  0.00  0.360  10.74   3.867   26.8 1.7e-02    to run
#   D8     10  0.25  0.65     600  0.00  0.577  10.58   6.111   24.6 1.6e-02    to run
#   D9     10  0.25  0.55     800  0.00  0.697  10.70   7.463   20.3 3.0e-02    to run
#  D10     10  0.25  0.55     600  0.00  0.697  11.95   8.334   15.2 6.5e-02    to run
# ═══════════════════════════════════════════════════════════════════════════

# Shared by all 24 runs. nu and pr0_factor are per-run and are NOT here.
COMMON = dict(
    dissipative=True,
    no_25pn_data=True,          # PHYSICS-ONLY (audited at first backward pass)
    l2_target=1e-3, gap_factor=10.0,
    w_secular=50.0,
    w_L=1.0, w_anchor=10.0,
    n_harmonic=0,
    n_march=10, march_fraction=0.4,
    use_causal_training=False,
    transfer_from_2pn=True,
    pretrain_epochs=10_000, pretrain_lr=1e-3,
    curriculum_start=1_000, curriculum_epochs=20_000,
    n_fourier=12, fourier_sigma=3.0,
    n_fourier_hi=20, fourier_sigma_hi=8.0,
    hidden=128, depth=5, lr=1e-3,
    n_colloc=6000,
    n_epochs=250_000,
    convergence_window=20, convergence_cv=0.5,
    compute_l2_time=False,
    plot_every=5_000, log_every=2_000, checkpoint_every=10_000,
    record_every_loss=500, record_every_l2=500, record_every_energy=5_000,
)

# 'L2' is the published value where known, None where the run is pending.
# It is metadata only -- the trainer never reads it.
IC_SET = [

  # ───────────────── six radial periods ─────────────────
  # published (draft table order)
  dict(tag='P1', nu=0.25, p0_factor=0.83, r0_km=260., pr0_factor=0.0,
       n_orbits=6, L2=5.9490e-06),   # e0=0.311 Nrev=6.50 e0Nrev=2.02
  dict(tag='P2', nu=0.25, p0_factor=0.88, r0_km=300., pr0_factor=0.0,
       n_orbits=6, L2=8.2015e-06),   # e0=0.226 Nrev=6.36 e0Nrev=1.44
  dict(tag='P3', nu=0.16, p0_factor=0.85, r0_km=140., pr0_factor=0.0,
       n_orbits=6, L2=9.8612e-06),   # e0=0.278 Nrev=7.38 e0Nrev=2.05
  dict(tag='P4', nu=0.24, p0_factor=0.72, r0_km=250., pr0_factor=0.17,
       n_orbits=6, L2=3.0008e-04),   # e0=0.501 Nrev=6.97 e0Nrev=3.49  reference configuration
  dict(tag='P5', nu=0.22, p0_factor=0.75, r0_km=180., pr0_factor=0.0,
       n_orbits=6, L2=4.6641e-04),   # e0=0.437 Nrev=8.38 e0Nrev=3.66
  dict(tag='P6', nu=0.25, p0_factor=0.65, r0_km=320., pr0_factor=0.0,
       n_orbits=6, L2=1.5463e-03),   # e0=0.577 Nrev=7.49 e0Nrev=4.33
  dict(tag='P7', nu=0.14, p0_factor=0.65, r0_km=220., pr0_factor=0.0,
       n_orbits=6, L2=9.4518e-05),   # e0=0.577 Nrev=4.12 e0Nrev=2.38  TRUNCATES at 4.12 periods (PN cut-off)
  dict(tag='P8', nu=0.07, p0_factor=1.2, r0_km=210., pr0_factor=0.3,
       n_orbits=6, L2=1.2820e-05),   # e0=0.574 Nrev=7.70 e0Nrev=4.42  weak dissipation; radius increases
  dict(tag='D1', nu=0.25, p0_factor=0.8, r0_km=200., pr0_factor=0.0,
       n_orbits=6, L2=None),   # e0=0.360 Nrev=6.92 e0Nrev=2.49  fills the 2.05-3.49 gap in e0*Nrev
  dict(tag='D2', nu=0.25, p0_factor=0.7, r0_km=400., pr0_factor=0.0,
       n_orbits=6, L2=None),   # e0=0.510 Nrev=6.46 e0Nrev=3.29  matched-product partner of D7
  dict(tag='D3', nu=0.25, p0_factor=0.9, r0_km=120., pr0_factor=0.0,
       n_orbits=6, L2=None),   # e0=0.190 Nrev=8.29 e0Nrev=1.58  DECONFOUND: near-circular but deep field
  dict(tag='D4', nu=0.25, p0_factor=0.55, r0_km=600., pr0_factor=0.0,
       n_orbits=6, L2=None),   # e0=0.697 Nrev=6.60 e0Nrev=4.60  DECONFOUND: eccentric but shallow field
  dict(tag='D5', nu=0.25, p0_factor=0.5, r0_km=700., pr0_factor=0.0,
       n_orbits=6, L2=None),   # e0=0.750 Nrev=7.38 e0Nrev=5.53  high end, 6 periods; partner of D8
  dict(tag='D6', nu=0.25, p0_factor=0.55, r0_km=500., pr0_factor=0.0,
       n_orbits=6, L2=None),   # e0=0.698 Nrev=7.63 e0Nrev=5.33  high end; partner of D9

  # ───────────────── ten radial periods ─────────────────
  # published (draft table order)
  dict(tag='P9', nu=0.25, p0_factor=0.92, r0_km=220., pr0_factor=0.0,
       n_orbits=10, L2=1.5371e-05),   # e0=0.154 Nrev=10.94 e0Nrev=1.68
  dict(tag='P10', nu=0.22, p0_factor=0.95, r0_km=140., pr0_factor=0.0,
       n_orbits=10, L2=4.4686e-05),   # e0=0.098 Nrev=12.23 e0Nrev=1.19
  dict(tag='P11', nu=0.24, p0_factor=0.75, r0_km=320., pr0_factor=0.0,
       n_orbits=10, L2=1.0433e-03),   # e0=0.437 Nrev=10.93 e0Nrev=4.78
  dict(tag='P12', nu=0.16, p0_factor=0.85, r0_km=140., pr0_factor=0.0,
       n_orbits=10, L2=1.0833e-03),   # e0=0.278 Nrev=13.69 e0Nrev=3.80
  dict(tag='P13', nu=0.25, p0_factor=0.75, r0_km=260., pr0_factor=0.0,
       n_orbits=10, L2=2.6401e-03),   # e0=0.438 Nrev=12.00 e0Nrev=5.25
  dict(tag='P14', nu=0.25, p0_factor=0.65, r0_km=380., pr0_factor=0.0,
       n_orbits=10, L2=1.4586e-02),   # e0=0.578 Nrev=12.03 e0Nrev=6.95  least accurate of the survey
  dict(tag='D7', nu=0.25, p0_factor=0.8, r0_km=320., pr0_factor=0.0,
       n_orbits=10, L2=None),   # e0=0.360 Nrev=10.74 e0Nrev=3.87  matched-product partner of D2
  dict(tag='D8', nu=0.25, p0_factor=0.65, r0_km=600., pr0_factor=0.0,
       n_orbits=10, L2=None),   # e0=0.577 Nrev=10.58 e0Nrev=6.11  fills the 5.26-6.95 gap; partner of D5
  dict(tag='D9', nu=0.25, p0_factor=0.55, r0_km=800., pr0_factor=0.0,
       n_orbits=10, L2=None),   # e0=0.697 Nrev=10.70 e0Nrev=7.46  replicates the P14 product at shallower field
  dict(tag='D10', nu=0.25, p0_factor=0.55, r0_km=600., pr0_factor=0.0,
       n_orbits=10, L2=None),   # e0=0.697 Nrev=11.95 e0Nrev=8.33  EXTRAPOLATION beyond all existing data
]

IC_BY_TAG = {d["tag"]: d for d in IC_SET}
TAGS = list(IC_BY_TAG)


# ── the run list, in a friendly format ────────────────────────────────────
def ic_table():
    """The 24 runs as a DataFrame (tag, n_orbits, nu, p0, r0_km, pr0, L2)."""
    rows = []
    for d in IC_SET:
        rows.append(dict(tag=d["tag"], n_orbits=d["n_orbits"], nu=d["nu"],
                         p0_factor=d["p0_factor"], r0_km=d["r0_km"],
                         pr0_factor=d["pr0_factor"],
                         published_L2=d["L2"],
                         status="published" if d["L2"] is not None else "new"))
    return pd.DataFrame(rows)


def show_ic_set(base_dir=BASE_DIR):
    """Print the 24 runs, marking which already have a trained folder."""
    t = ic_table()
    t["trained"] = ["yes" if os.path.isdir(os.path.join(base_dir, tag)) else "-"
                    for tag in t.tag]
    print(f"  {'tag':>4} {'N_orb':>6} {'nu':>6} {'p0':>6} {'r0[km]':>8} "
          f"{'pr0':>6} {'published L2':>13} {'trained':>8}")
    print("  " + "-" * 64)
    for r in t.itertuples():
        pub = "-" if r.published_L2 is None or (isinstance(r.published_L2, float)
                                                and np.isnan(r.published_L2)) \
              else f"{r.published_L2:.3e}"
        print(f"  {r.tag:>4} {r.n_orbits:>6} {r.nu:>6.2f} {r.p0_factor:>6.2f} "
              f"{r.r0_km:>8.0f} {r.pr0_factor:>6.2f} {pub:>13} {r.trained:>8}")
    return t


# ── train one case, by tag ────────────────────────────────────────────────
def run_case(tag, common=None, base_dir=BASE_DIR, dev=None, **overrides):
    """Train ONE case of the survey, selected by its tag ('D1', 'P4', ...).

    common    : the shared settings (defaults to the module-level COMMON)
    overrides : merged last, e.g. run_case("D1", n_epochs=50_000)
    """
    if tag not in IC_BY_TAG:
        raise KeyError(f"unknown tag {tag!r}; choose one of {TAGS}")
    cfg = dict(IC_BY_TAG[tag])
    cfg.pop("tag"); published = cfg.pop("L2")
    settings = {**(COMMON if common is None else common), **overrides}
    settings.setdefault("dev", dev or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"\n{'='*70}")
    print(f"  RUNNING {tag}   {cfg}")
    if published is not None:
        print(f"  (published value for this configuration: L2 = {published:.4e})")
    print(f"{'='*70}")

    diag, best_l2_shape, _ = train_orbit_adaptive(
        **settings, **cfg, base_dir=os.path.join(base_dir, tag))

    if best_l2_shape is None:
        print(f"\n>>> {tag}: aborted by the regime diagnosis")
        return diag, None
    print(f"\n>>> {tag}: L2 = {best_l2_shape:.4e}")
    if published is not None:
        print(f"    published: {published:.4e}   ratio {best_l2_shape/published:.2f}")
    return diag, best_l2_shape


# ── the paper table, rebuilt from the trained folders ─────────────────────
def run_dir(tag, base_dir=BASE_DIR):
    """The <base_dir>/<tag>/<regime folder>/ of a finished run, or None."""
    d = os.path.join(base_dir, tag)
    if not os.path.isdir(d):
        return None
    subs = [s for s in sorted(os.listdir(d)) if not s.startswith(".")
            and os.path.isdir(os.path.join(d, s))]
    return os.path.join(d, subs[0]) if subs else None


def load_model(path, pn_order=2):
    """Rebuild the trained network of a run folder from angle_best.pth.

    The checkpoint is a plain state_dict that carries u0/pr0/L0/u_lo/u_hi/
    pr_max and the Fourier banks, so only hidden/depth are read from config.json.
    """
    cfg = json.load(open(os.path.join(path, "config.json")))
    sd = torch.load(os.path.join(path, "angle_best.pth"), map_location="cpu",
                    weights_only=False)
    phi_max = float(np.load(os.path.join(path, "analysis.npz"))["phi_grid"].max())
    model = AngleOrbitPINN4(phi_max, float(sd["u0"]), float(sd["pr0"]),
                            float(sd["L0"]), float(sd["u_lo"]), float(sd["u_hi"]),
                            dissipative=bool(cfg["regime"].get("dissipative", True)),
                            n_fourier=len(sd["B"]), n_fourier_hi=len(sd["B_hi"]),
                            hidden=cfg["training"]["hidden"],
                            depth=cfg["training"]["depth"],
                            pr_max_override=float(sd["pr_max"]))
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


def energy_diagnostics(path, pn_order=2):
    """H(phi) of the trained network and of the 2.5PN reference on the same
    grid: total change (in % of |H(0)|) and the fraction of rising steps.

    A true 2.5PN inspiral radiates, so H must decrease monotonically: rising
    steps are the unphysical fraction.
    """
    a = np.load(os.path.join(path, "analysis.npz"))
    phi = a["phi_grid"]
    model, cfg = load_model(path, pn_order)
    nu = cfg["regime"]["nu"]

    P = torch.tensor(phi, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        o = model(P)
    qx, qy, px, py = reconstruct_qp4(o[:, 0], o[:, 1], o[:, 2], P.view(-1))
    q = torch.stack([qx, qy], 1).double(); p = torch.stack([px, py], 1).double()
    H = compute_hamiltonian_r(q, p, pn_order, nu, _const.C_SQ, _const.C_QD).numpy()

    ref = a["ref_target"]
    Href = compute_hamiltonian_r(torch.tensor(ref[:, :2]), torch.tensor(ref[:, 2:]),
                                 pn_order, nu, _const.C_SQ, _const.C_QD).numpy()
    return dict(dH_pct=100*(H[-1]-H[0])/abs(H[0]),
                frac_up=float((np.diff(H) > 0).mean()),
                dH_ref_pct=100*(Href[-1]-Href[0])/abs(Href[0]),
                H=H, H_ref=Href, phi=phi)


def gap_2p5(path, n_pts=1500):
    """2PN <-> 2.5PN 4D-L2 gap: how much signal radiation reaction puts into
    the orbit.  Same computation as ``_l2_2p5_gap`` inside the trainer, applied
    to the two references stored in analysis.npz (verified: reproduces the
    trainer's printed value, e.g. D1 4.51e-2 vs 4.5e-2 in the paper)."""
    a = np.load(os.path.join(path, "analysis.npz"))
    L0 = json.load(open(os.path.join(path, "analysis_summary.json")))["regime"]["L0"]

    def _polar(y):
        phi = np.unwrap(np.arctan2(y[:, 1], y[:, 0])); phi -= phi[0]
        r = np.hypot(y[:, 0], y[:, 1])
        pr = (y[:, 0]*y[:, 2] + y[:, 1]*y[:, 3]) / r
        return phi, r, pr

    p2, r2, pr2 = _polar(a["ref_2pn_cons"])
    p25, r25, pr25 = _polar(a["ref_target"])
    pg = np.linspace(0, min(p2[-1], p25[-1]), n_pts)
    u2, u25 = np.interp(pg, p2, 1/r2), np.interp(pg, p25, 1/r25)
    pa2, pa25 = np.interp(pg, p2, pr2), np.interp(pg, p25, pr25)

    def _state(u, pa):
        return np.stack([np.cos(pg)/u, np.sin(pg)/u,
                         pa*np.cos(pg) - (L0*u)*np.sin(pg),
                         pa*np.sin(pg) + (L0*u)*np.cos(pg)])

    A, B = _state(u2, pa2), _state(u25, pa25)
    gap = float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-16))
    rp_rg = float((1/u25).min() * _const.R0_REF_KM / _const.R_G_KM)
    return gap, rp_rg


SUMMARY_JSON = "paper_table.json"
SUMMARY_CSV  = "paper_table.csv"


def paper_table(base_dir=BASE_DIR, save=True, verbose=True):
    """Rebuild the paper's dissipative table from the trained folders.

    One row per run found in ``base_dir``: N_orb, nu, p0, r0 (in R0 = 200 km,
    as in the paper), e0, N_rev, r_p/r_g, gap, L2, and the energy diagnostics
    (dH in %, rising-step fraction).  Saved to ``<base_dir>/paper_table.csv``
    and ``.json`` so the table never has to be recomputed.
    """
    rows = []
    for tag in TAGS:
        path = run_dir(tag, base_dir)
        if path is None:
            continue
        ic = IC_BY_TAG[tag]
        summ = json.load(open(os.path.join(path, "analysis_summary.json")))
        a = np.load(os.path.join(path, "analysis.npz"))
        gap, rp_rg = gap_2p5(path)
        en = energy_diagnostics(path)
        phi_max = float(a["phi_grid"].max())
        rows.append(dict(
            tag=tag, n_orbits=ic["n_orbits"], nu=ic["nu"],
            p0_factor=ic["p0_factor"], r0=ic["r0_km"]/_const.R0_REF_KM,
            r0_km=ic["r0_km"], pr0_factor=ic["pr0_factor"],
            e0=summ["regime"]["ecc"], N_rev=phi_max/(2*np.pi),
            rp_rg=rp_rg, gap=gap,
            L2=summ["training"]["best_l2_shape"],
            published_L2=ic["L2"],
            dH_pct=en["dH_pct"], rising_pct=100*en["frac_up"],
            dH_ref_pct=en["dH_ref_pct"],
            e0_Nrev=summ["regime"]["ecc"]*phi_max/(2*np.pi),
        ))
    t = pd.DataFrame(rows).sort_values(["n_orbits", "e0_Nrev"]).reset_index(drop=True)
    if save and len(t):
        t.to_csv(os.path.join(base_dir, SUMMARY_CSV), index=False)
        t.to_json(os.path.join(base_dir, SUMMARY_JSON), orient="records", indent=2)
        if verbose:
            print(f"  saved {os.path.join(base_dir, SUMMARY_CSV)} and {SUMMARY_JSON}")
    return t


def show_paper_table(base_dir=BASE_DIR, table=None):
    """The paper table, formatted for Jupyter (falls back to text)."""
    t = paper_table(base_dir, save=False, verbose=False) if table is None else table
    if not len(t):
        print(f"no trained runs under {os.path.abspath(base_dir)}"); return t
    show = t[["n_orbits", "nu", "p0_factor", "r0", "e0", "N_rev", "rp_rg",
              "gap", "L2", "dH_pct", "rising_pct"]].copy()
    show.columns = ["N_orb", "nu", "p0_hat", "r0", "e0", "N_rev", "rp/rg",
                    "gap", "L2", "dH [%]", "rising [%]"]
    fmt = {"nu": "{:.2f}", "p0_hat": "{:.2f}", "r0": "{:.2f}", "e0": "{:.3f}",
           "N_rev": "{:.2f}", "rp/rg": "{:.1f}", "gap": "{:.1e}", "L2": "{:.1e}",
           "dH [%]": "{:+.1f}", "rising [%]": "{:.1f}"}
    try:
        from IPython.display import display
        display(show.style.format(fmt).hide(axis="index")
                    .set_caption("Dissipative survey (2.5PN)"))
    except Exception:
        print(show.to_string(index=False))
    return t


def latex_paper_table(base_dir=BASE_DIR, table=None):
    """The same table as LaTeX rows, grouped by six / ten radial periods."""
    t = paper_table(base_dir, save=False, verbose=False) if table is None else table
    def sci(x):
        s = f"{x:.1e}".split("e"); return f"${float(s[0]):.1f}\\times10^{{{int(s[1])}}}$"
    out = []
    for n_orb, title in ((6, "six radial periods"), (10, "ten radial periods")):
        sub = t[t.n_orbits == n_orb]
        if not len(sub): continue
        out.append(r"\multicolumn{9}{c}{\emph{" + title + r"}}\\")
        for r in sub.itertuples():
            out.append(f"{r.n_orbits} & ${r.nu:.2f}$ & ${r.p0_factor:.2f}$ & "
                       f"${r.r0:.2f}$ & ${r.e0:.3f}$ & ${r.N_rev:.2f}$ & "
                       f"${r.rp_rg:.1f}$ & {sci(r.gap)} & {sci(r.L2)} \\\\")
    return "\n".join(out)
