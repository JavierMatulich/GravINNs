# Reproducibility guide

This repository contains the code paths used for the four numerical studies in the accompanying manuscript. The notebooks are the most direct entry points; command-line drivers are provided for the two sweep-style experiments.

## Environment

```bash
python -m pip install -e ".[notebook]"
```

GPU training is supported through PyTorch, but the library and fast test suite also run on CPU.

## Paper workflows

| Study | Notebook / entry point | Main trainer |
|---|---|---|
| Near-circular reliability | `notebooks/01_near_circular_n30.ipynb`, `gravinns-nc30` | `gravinns.training.train_near_circular` |
| Long-orbit conservative dynamics | `notebooks/02_long_orbit.ipynb`, `gravinns-longorbit` | `gravinns.training.train_long_orbit` |
| Four-parameter surrogate | `notebooks/03_parametric.ipynb` | `parametric/` |
| Dissipative 2.5PN sector | `notebooks/04_dissipative.ipynb` | `gravinns.training.train_dissipative` |

The three trainer entry points implement distinct scientific protocols. Their reference integration, diagnostics, residual construction, checkpoint I/O, recording, and live plotting are shared through small modules in `src/gravinns/training/`; see `docs/training.md`.

## Shipped model artifacts

The repository includes the complete artefacts of the published runs: the best checkpoint, the compact analysis products, the per-epoch `history.json` (loss and L2 curves — the only such record for these runs, which predate `plot_record.npz`) and the `angle_latest.pth` resume checkpoint. Together they are about 150 MB; use Git LFS if clone size matters (see `.gitignore`).

Parametric runs retain their final/best/last checkpoints because those files represent distinct states used by the parametric workflow.

## Numerical tests

Fast validation:

```bash
pytest -m "not slow"
```

Training smoke tests:

```bash
pytest -m slow
```

The latter executes real, reduced-budget optimization loops and can take several minutes on CPU.

## Archiving a paper release

For a manuscript release, tag the exact commit (for example `v1.0.0-paper`) and archive that tag with Zenodo or an equivalent long-term repository. Record the resulting DOI in the manuscript and in `CITATION.cff`.

## Pinned validation environment

The general package requirements intentionally use version ranges.  For a
fully pinned environment matching the software-validation run for this release,
install `requirements-paper.txt` instead.  The scientific training artifacts in
`models/` are stored separately from the environment lock; the lock is intended
to make code execution and the automated tests reproducible.

## Training protocol names

The three single-orbit trainers are scientific protocols, not successive
software versions.  New code should use:

- `gravinns.training.train_near_circular`
- `gravinns.training.train_long_orbit`
- `gravinns.training.train_dissipative`

The historical `train_regime_unified`, `train_regime_unified_v3`, and
`train_regime_unified_v4` names remain as compatibility aliases for archived
notebooks and scripts.
