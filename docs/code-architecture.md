# Code architecture

## Single source of truth for PN dynamics

The 0PN--3PN Hamiltonian is defined once in
`gravinns.physics.hamiltonian.compute_hamiltonian_r`. NumPy and polar-coordinate
helpers delegate to that implementation; they do not carry independent copies
of the PN series. This prevents RK integration, adaptive regime diagnosis, and
post-hoc long-orbit analysis from silently drifting apart if the Hamiltonian is
changed.

## Training protocols

The project contains three distinct single-orbit training protocols:

- `near_circular_trainer.py`: phase-degeneracy / near-circular study;
- `long_orbit_trainer.py`: radial-harmonic and expanding-window long-orbit study;
- `dissipative_trainer.py`: 2.5PN radiation-reaction study with secular balance.

Shared mechanics live in `checkpointing.py`, `reference.py`, `diagnostics.py`,
`residuals.py`, `recorders.py`, `live_plot.py`, and `schedules.py`. The protocol
files retain only regime-specific orchestration and loss definitions. Legacy
`unified_trainer*.py` modules are nine-line compatibility shims and should not
be used by new code.
