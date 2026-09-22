# Contributing

Contributions that improve correctness, reproducibility, documentation, or test coverage are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

## Tests

The default test suite excludes training-heavy integration tests:

```bash
pytest -m "not slow"
```

Run the complete suite, including short end-to-end training jobs, with:

```bash
pytest
```

## Code changes

- Keep physical formulae in `src/gravinns/physics/` and shared training machinery in `src/gravinns/training/`.
- Do not copy helper functions into experiment drivers when a shared implementation already exists.
- Preserve public trainer call signatures unless a breaking release is intended.
- Add or update tests when changing numerical behaviour.
- Do not commit generated run directories, per-epoch histories, or resume checkpoints.
- Keep paper-facing scripts deterministic by exposing random seeds explicitly.

## Reproducibility

When a change affects a paper workflow, record the command, configuration, seed, and expected numerical tolerance. See `REPRODUCIBILITY.md` for the reference workflows.
