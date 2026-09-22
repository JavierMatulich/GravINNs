# Pretrained model artifacts

This directory contains selected checkpoints and compact analysis products associated with the paper workflows.

## Parametric surrogate

`parametric/` contains the six runs used in the physics-informed versus supervised comparison at anchor counts `K = 500`, `5000`, and `20000`.

Each run retains its distinct final, best, and last-refresh checkpoints because the parametric workflow can explicitly select among them. Configuration files and compact summaries are stored alongside the weights.

Example:

```python
from gravinns.parametric import load_parametric_model

model = load_parametric_model(
    "models/parametric/parametric_pinn_1_5mill_PINN_20000",
    which="param_pinn_best.pt",
    device="cpu",
)
```

## Dissipative surrogate

`dissipative/` contains the best checkpoint and compact analysis products for each representative 2.5PN run:

- `angle_best.pth`: best network checkpoint;
- `analysis.npz`: compact numerical analysis arrays;
- `analysis_summary.json`: scalar summary quantities;
- `config.json`: run configuration;
- `state.json`: compact training state metadata.

Large per-epoch `history.json` files and `angle_latest.pth` resume checkpoints are intentionally omitted from the published repository. They are produced automatically when a run is trained and are not required to inspect the best-checkpoint results reported in the manuscript.
