"""Single-orbit statistics sweep (notebook cell 3).

B.  SINGLE-ORBIT STATISTICS SWEEP

Trains one independent PINN per initial condition over a stratified set of
ICs, and records - per run, on disk - everything needed to reconstruct any
figure or table in the paper WITHOUT retraining.

WHAT IT IS DESIGNED TO MEASURE
------------------------------
(1) DIFFICULTY vs PHYSICS. Final L2, epochs-to-target, and the terminal
    residual are stratified by the physically meaningful variables the
    draft already uses: eccentricity, periapsis r_p/r_g, and the PN
    expansion parameter x = v^2/c^2 at periapsis. This is the single-orbit
    analogue of Table II/III of the draft.

(2) THE NEAR-CIRCULAR LOCAL-MINIMUM EFFECT. The claim is that as ecc -> 0
    the global minimum becomes nearly degenerate with spurious minima, so
    the optimizer can converge on the residual while sitting on the wrong
    orbit. That claim needs a DISCRIMINATING statistic, not just a mean L2.
    Two are computed here:

      * SEED SCATTER. Every IC is trained with n_seeds independent
        initializations. A well-conditioned problem gives tightly clustered
        final L2; a landscape with competing minima gives multi-modal
        scatter. The spread across seeds - not the mean - is the evidence.

      * RESIDUAL-vs-ERROR DECOUPLING. For each run we store the terminal
        l_pde alongside the terminal L2. A run with small residual and
        large L2 is, by the argument of Sec. IV of the draft, sitting in a
        phase-degenerate/spurious minimum. The fraction of such runs per
        eccentricity bin is the quantitative statement of the effect.

    Note these two effects have OPPOSITE eccentricity trends (stiffness
    grows with ecc, degeneracy grows as ecc -> 0), so a plot of L2 vs ecc
    alone would show a U-shape whose two branches have entirely different
    causes. Separating them is the point of the diagnostics below.

ON-DISK LAYOUT (one directory per run, self-contained):
    <sweep_root>/<run_id>/
        angle_best.pth        model weights (written by the trainer)
        angle_latest.pth      resume state
        history.json          loss + L2 curves (trainer)
        loss_components.npz   per-epoch loss channels (cell A)
        analysis.npz          reference + prediction arrays (trainer)
        config.json           trainer config
        run_record.json       <-- THIS HARNESS: IC, diagnostics, timings,
                                  convergence summary, seed, git-free
                                  provenance, environment
    <sweep_root>/sweep_index.csv    one row per run, appended live
    <sweep_root>/sweep_manifest.json

The sweep is RESUMABLE: a run whose run_record.json exists and is marked
complete is skipped. Kill and restart at will.
"""
from .analysis import classify_failure, summarize_run, wilson_interval
from .ic_generation import (ecc_bin, generate_ics, generate_ics_stratified,
                            ic_diagnostics, rp_bin)
from .runner import run_ic_sweep

__all__ = ["ic_diagnostics", "ecc_bin", "rp_bin", "generate_ics",
           "generate_ics_stratified", "summarize_run", "classify_failure",
           "wilson_interval", "run_ic_sweep"]
