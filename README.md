# GravINNs: Gravity-Informed Neural Networks for Post-Newtonian Dynamics

GravINNs introduces a gravity-informed neural network framework designed to model post-Newtonian (PN) binary orbital dynamics. Authored by G. Barbagallo and J. Matulich, the study develops physics-informed neural networks (PINNs) as efficient surrogates for both conservative and dissipative relativistic two-body problems. The primary objective is to replace repeated numerical integrations with a single trained network capable of generating continuous orbital families conditioned on physical parameters.

## Key Methodological Innovations
* **Angle-Domain Formulation:** The equations of motion are parameterized by the orbital angle φ rather than time, allowing for a compact, bounded domain well-suited for neural network representation.
* **Architectural Constraints:** The framework builds physical structure directly into the network by enforcing initial conditions exactly and conserving angular momentum by construction. 
* **Overcoming Phase Blindness:** We identify that autonomous physics residuals are blind to rigid rotations of the orbit. To fix the orbital phase, we implement a "warm start" transfer learning strategy from lower-order 1PN solutions.
* **Radial-Harmonic Features:** To prevent phase error accumulation over long integration times, the network's input representation includes a learnable harmonic feature bank locked to the orbit's radial frequency.
* **Secular Energy-Balance Loss:** For the 2.5PN dissipative inspiral, the radiation-reaction force is too small to be resolved by a standard pointwise residual alone. We introduce an integrated energy-balance constraint to accurately capture the secular orbital decay.

## Primary Findings
* **Parametric Generalization:** A single surrogate network successfully maps a four-parameter continuous space—initial separation, radial momentum, tangential momentum, and symmetric mass ratio—to accurate 2PN trajectories.
* **Physics-Driven Accuracy Limits:** The parametric surrogate achieves a median relative L² error of 3 × 10⁻⁴ within the valid PN regime. Performance only collapses when the orbit's periapsis enters the strong-field region, demonstrating that accuracy is limited by the physical validity of the 2PN expansion rather than network capacity.
* **Long-Time Evolution:** Through self-composition and exact restart mapping, the network can chain evaluations to autonomously evolve orbits for over 116 revolutions.


The post-Newtonian physics is implemented once and shared across all workflows. Workflow-specific modules contain only the model, training, evaluation, and analysis logic needed for that study.


The repository is designed to separate physical models from training infrastructure: Hamiltonians and radiation reaction live in `physics/`, neural architectures in `models/`, reusable optimization machinery in `training/`, and each paper study in `experiments/` or `parametric/`.

## Installation

From a fresh checkout:

```bash
git clone https://github.com/JavierMatulich/GravINNs.git
cd GravINNs
python -m pip install -e .
```

For notebooks and development tools:

```bash
python -m pip install -e ".[notebook,dev]"
```

Python 3.9 or newer is required. GPU execution is optional and is handled by PyTorch when CUDA is available. For the pinned validation environment used for this release, install `requirements-paper.txt`.

## Paper workflows

### 1. Near-circular reliability and transfer learning

```bash
gravinns-nc30
```

or open `notebooks/01_near_circular_n30.ipynb`.

This study compares cold physics-only optimization with lower-order warm starts and uses `gravinns.training.train_near_circular`.

### 2. Long-orbit conservative dynamics

```bash
gravinns-longorbit
```

or open `notebooks/02_long_orbit.ipynb`.

The long-orbit workflow adds a representation adapted to radial periodicity and uses `gravinns.training.train_long_orbit`.

### 3. Four-parameter conservative surrogate

Open `notebooks/03_parametric.ipynb`. The parametric package implements a network conditioned on the orbital angle and four physical parameters and includes utilities for reference-orbit generation, training, and benchmark evaluation.

### 4. Dissipative 2.5PN dynamics

Open `notebooks/04_dissipative.ipynb`. The dissipative workflow uses `gravinns.training.train_dissipative`, including the angular-momentum and secular energy-balance constraints.

## Repository layout

```text
src/gravinns/
├── physics/        # PN Hamiltonians and radiation reaction
├── models/         # neural architectures
├── training/       # shared single-orbit training infrastructure
├── parametric/     # parameter-conditioned surrogate
├── experiments/    # paper-level experiment drivers
├── recording/      # compact run records
├── plotting/       # plotting helpers
└── sweep/          # sweep generation and analysis

notebooks/           # four paper workflows
models/              # selected pretrained checkpoints and analysis products
tests/               # unit and reduced-budget integration tests
docs/                # implementation notes
```

The three single-orbit trainer entry points remain separate because they correspond to distinct scientific regimes, while common reference integration, residual construction, schedules, optimizer setup, diagnostics, checkpointing, recording, and live plotting are factored into reusable modules. The PN Hamiltonian itself has a single canonical implementation in `physics/hamiltonian.py`. See [`docs/training.md`](docs/training.md) and [`docs/code-architecture.md`](docs/code-architecture.md).

## Pretrained results

`models/parametric/` contains the checkpoints used for the parametric comparison. `models/dissipative/` contains the best checkpoints and compact analysis products for the dissipative configurations.

Large per-epoch histories and resume-only checkpoints are not committed for the shipped dissipative runs. They are regenerated automatically by training and are not required to inspect or reproduce the reported best-checkpoint statistics. See [`models/README.md`](models/README.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Testing

Fast unit and compatibility tests:

```bash
pytest -m "not slow"
```

Reduced-budget end-to-end training tests:

```bash
pytest -m slow
```

The slow tests exercise the real training loops and may take several minutes on CPU.

## Reproducibility

Detailed workflow-to-code mapping, artifact policy, and release guidance are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Random seeds are exposed by the experiment drivers and trainer APIs; paper releases should be archived from a tagged commit.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). If you use GravINNs in published work, please cite the accompanying paper and the archived software release.

## License

GravINNs is released under License see [`LICENSE`](LICENSE).
