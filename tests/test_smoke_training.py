"""A few-epoch run of both arms through the real sweep driver (CPU, ~1 min)."""
import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from gravinns.experiments import near_circular_n30 as nc30

pytestmark = pytest.mark.slow

TINY = dict(n_epochs=20, n_orbits=1, log_every=10, plot_every=10,
            n_colloc=64, checkpoint_every=10, pretrain_epochs=10,
            hidden=16, depth=2)


@pytest.mark.parametrize("arm", ["cold", "transfer"])
def test_run_and_report(tmp_path, arm):
    ics = nc30.get_ics_30(results_dir=str(tmp_path), verbose=False)[:1]
    nc30.run_to30(arm_order=(arm,), results_dir=str(tmp_path), ics=ics,
                  train_overrides=TINY)
    root = tmp_path / nc30.ROOT_30.format(pn=nc30.PN_30, arm=arm)
    idx = pd.read_csv(root / "sweep_index.csv")
    assert (idx.status == "complete").all() and len(idx) == 2
    run_dir = root / idx.run_id.iloc[0]
    for f in ("angle_best.pth", "history.json", "analysis.npz", "plot_record.npz"):
        assert os.path.exists(run_dir / f)
