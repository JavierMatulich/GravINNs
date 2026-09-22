"""Model backward compatibility + a few-epoch long-orbit run (CPU, ~1 min)."""
import json
import os

import matplotlib
matplotlib.use("Agg")

import torch
import pytest

import gravinns.experiments.long_orbit as lo
from gravinns.models import AngleOrbitPINN4


def test_model_without_harmonics_is_the_sweep_model():
    torch.manual_seed(0)
    m = AngleOrbitPINN4(10.0, 1.0, 0.0, 1.0, 0.5, 2.0, n_fourier_hi=16)
    assert "log_omega" not in dict(m.named_parameters())
    assert "harm_n" not in dict(m.named_buffers())
    assert m.net[0].in_features == 1 + 2*8 + 2*16


def test_model_with_harmonics():
    m = AngleOrbitPINN4(10.0, 1.0, 0.0, 1.0, 0.5, 2.0, n_harmonic=4,
                        omega_radial_init=6.5)
    out = m(torch.linspace(0, 10, 7).view(-1, 1))
    assert out.shape == (7, 3) and torch.isfinite(out).all()


@pytest.mark.slow
def test_run_ic_tiny(tmp_path, monkeypatch):
    monkeypatch.setitem(lo.BASE_CFG, "pretrain_epochs", 20)
    for k, v in dict(n_colloc=64, hidden=16, depth=2, log_every=10,
                     plot_every=10, checkpoint_every=10).items():
        monkeypatch.setitem(lo.BASE_CFG, k, v)
    monkeypatch.setitem(lo.STAGE_A, "n_epochs", 20)
    monkeypatch.setitem(lo.STAGE_B_ADAM, "n_epochs", 30)
    root = str(tmp_path / "longorbit")
    rec = lo.run_ic("e036", refine="adam", sweep_root=root, n_orbits=2)
    assert "error" not in rec
    summ = json.load(open(os.path.join(root, "all_results.json")))
    assert [r["ic_name"] for r in summ] == ["e036"]
    assert os.path.exists(os.path.join(root, "N2", rec["leaf"], "analysis.npz"))


def test_stage_budgets():
    assert lo.STAGE_A["n_epochs"] == 100_000
    assert lo.STAGE_B_ADAM["n_epochs"] == 150_000
    assert lo.BASE_CFG["curriculum_epochs"] == 15_000


def test_long_orbit_uses_the_notebook_grid_resolutions():
    """The long-orbit protocol has TWO resolutions, and they must stay apart.

    phi grid: 80*n_orbits; RK45 references: max(4000, 400*n_orbits).  Passing a
    single n_ref to build_reference_orbits (or using the commented-out
    max(4000, 2000*n_orbits) variant) changes the trained model and blows the
    grid up to 100k points at N=50.
    """
    import inspect
    from gravinns.training import long_orbit_trainer as lot
    src = inspect.getsource(lot.train_long_orbit)
    assert "_n_ref = int(80 * n_orbits)" in src
    assert "_n_ref_t = max(4000, int(400 * n_orbits))" in src
    assert "n_ref=_n_ref_t" in src
    assert "n_ref=_n_ref," not in src
