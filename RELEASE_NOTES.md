# GravINNs v1.0.0 — publication release

This archive is prepared for the accompanying scientific manuscript.

## Release cleanup

- PN Hamiltonian centralized in `gravinns.physics.hamiltonian`; NumPy and polar helpers delegate to the canonical implementation.
- Single-orbit trainers renamed by scientific role: `train_near_circular`, `train_long_orbit`, and `train_dissipative`.
- Historical `train_regime_unified*` imports retained as compatibility shims.
- Shared curriculum, causal-weight, marching, optimizer/resume, checkpoint, diagnostic, recording, and plotting machinery factored into reusable modules.
- Compiled Python artifacts removed from the archive.
- Publication metadata updated and a pinned validation environment added.
- Fast release test suite: 30 passed, 3 slow tests deselected.

The slow tests execute reduced-budget real training loops and are intentionally not part of the fast release audit.

## 1.0.1

Fixes found by re-running the reduced equivalence harnesses against the four
original notebooks:

* **Long-orbit trainer grid.** The notebook uses two resolutions: `_n_ref =
  80*n_orbits` for the phi grid and `max(4000, 400*n_orbits)` for the RK45
  references. They had been collapsed into a single `n_ref =
  max(4000, 2000*n_orbits)` (a line the author had left commented out), which
  changed both grids and the trained model (L2 1.018e-2 -> 1.024e-2 on the
  reduced run; 4000 -> 100000 phi points at N=50). Restored.
* **Dissipative artefacts.** `history.json` and `angle_latest.pth` are shipped
  again for the 20 published runs.
* Removed the unused duplicate `recording/plot_record_dissipative.py`.

Verified after the fixes: near-circular (both arms), long-orbit, parametric and
dissipative all reproduce their notebooks bit-for-bit.
