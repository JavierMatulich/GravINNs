# Training architecture

GravINNs contains three single-orbit training protocols because the paper uses
three distinct optimization regimes. They share the same model, Hamiltonian,
reference-orbit construction, diagnostics, checkpoint I/O, recorders, and live
plotting utilities.

## Shared components

- `training/reference.py`: 0PN--3PN RK reference construction and warm-start selection.
- `training/residuals.py`: angle-domain conservative and dissipative residuals.
- `training/diagnostics.py`: phase-space L2 and time-domain diagnostics.
- `training/checkpointing.py`: resumable checkpoint/state/history I/O.
- `training/recorders.py`: full-resolution loss/L2 plot records.
- `training/live_plot.py`: training visualization only; it never enters the loss.
- `training/audit.py`: provenance check for physics-only dissipative runs.

The public trainer functions retain their original call signatures so scripts and
notebooks used for the paper remain compatible.

## Near-circular protocol

`train_near_circular` is the conservative protocol used by the near-circular
reliability study. It supports cold and transfer starts, causal weighting, and
sweep-specific seeding/resume controls.

## Long-orbit protocol

`train_long_orbit` adds a learnable radial-harmonic representation,
expanding-window marching, denser RK references, causal annealing, and an
optional frozen-batch L-BFGS polish. These changes target accumulated phase
error over long integrations.

## Dissipative protocol

`train_dissipative` adds the 2.5PN angular-momentum and secular
energy-balance constraints. With `no_25pn_data=True`, the 2.5PN RK trajectory is
used only for evaluation and plotting; it is not part of the training loss.
The provenance audit checks this at runtime.

## Why the three entry points remain separate

The protocols differ in their scientific training schedules, not merely in
cosmetic options. The common numerical machinery has been factored into shared
modules, while the protocol-specific optimization loops remain explicit. This
keeps the paper workflows readable and avoids hiding scientifically relevant
schedule changes behind a large set of flags.
