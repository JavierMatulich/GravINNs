"""Parametric case: shipped models load and predict; cache build / evaluation."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import gravinns.constants as const
from gravinns.parametric import (C_QD, C_SQ, anchors_and_probes_from_cache,
                                 build_ref_data_from_cache, build_reference_cache,
                                 eval_model_on_cache, load_parametric_model,
                                 load_reference_cache, load_reference_cache_lazy,
                                 make_anchor_thetas, predict_orbit, score_on_cache)
from gravinns.parametric.orbits import _pham_batch, _pham_batch_local
from gravinns.physics import compute_hamiltonian_r

MODELS = Path(__file__).resolve().parents[1] / "models" / "parametric"
PR = dict(p0_factor=(0.6, 1.2), r0_km=(100.0, 400.0),
          pr0_factor=(-0.35, 0.35), nu=(0.001, 0.25))


def test_shared_pieces():
    assert C_SQ == const.C_SQ and C_QD == const.C_QD
    assert _pham_batch is compute_hamiltonian_r is _pham_batch_local


@pytest.mark.parametrize("name", sorted(os.listdir(MODELS)))
def test_shipped_models_load_and_predict(name):
    m = load_parametric_model(str(MODELS / name), which="param_pinn_best.pt", device="cpu")
    orb = predict_orbit(m, 0.8, 200.0, 0.1, 0.2, n_pts=200)
    assert np.isfinite(orb["u"]).all() and (orb["u"] > 0).all()


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("c") / "refs_tiny.npz")
    anchors = make_anchor_thetas(PR, n_anchors=4, seed=0)
    build_reference_cache(PR, path=p, n_samples=12, anchor_thetas=anchors,
                          pde_pn_order=2, n_orbits=2, n_pts=200, n_steps=1500,
                          seed=5678, batch_size=12, device="cpu")
    return p


def test_cache_eval_lazy_and_chunked(tiny_cache):
    eager = load_reference_cache(tiny_cache)
    lazy = load_reference_cache_lazy(tiny_cache)
    m = load_parametric_model(str(MODELS / "parametric_pinn_1_5mill_PINN_500"),
                              which="param_pinn_best.pt", device="cpu")
    ref = eval_model_on_cache(m, eager, device="cpu", batch_size=4, verbose=False)["l2"]
    got = score_on_cache(m, lazy, device="cpu", batch_size=4, verbose=False)
    # float32 storage in the lazy loader / scorer -> small relative differences
    np.testing.assert_allclose(got, ref, rtol=1e-3)


def test_ref_data_from_cache(tiny_cache):
    c = load_reference_cache(tiny_cache)
    idx, anchors, probes, seen = anchors_and_probes_from_cache(c, 6)
    rd = build_ref_data_from_cache(c, idx, float(min(c["span"][idx])), n_pts=50)
    assert rd["U"].shape == (6, 50) and len(probes) == 6 and seen == [True]*3 + [False]*3
    assert isinstance(rd["THETA"], torch.Tensor)


def test_joint_tables(capsys):
    from gravinns.parametric import compare_models_joint, joint_table_2d, periapsis_rg
    rng = np.random.default_rng(0)
    ecc = rng.uniform(0, 0.8, 5000); rp = rng.uniform(5, 60, 5000)
    a = rng.lognormal(-7, 1, 5000); b = a * 0.5
    joint_table_2d(a, ecc, rp, label="A")
    compare_models_joint(a, b, ecc, rp, name_a="A", name_b="B")
    out = capsys.readouterr().out
    assert "JOINT STRATIFICATION -- A" in out and "B minus A" in out
    th = np.array([[0.8, 200.0, 0.0, 0.2]])
    assert abs(periapsis_rg(th, np.array([1 - 0.8**2]))[0] - 200*0.64/(2-0.64)/4.1358) < 0.05


def test_score_db_heatmaps_and_progress(tiny_cache, capsys):
    import matplotlib
    matplotlib.use("Agg")
    from gravinns.parametric import db_pair_heatmaps, score_db
    lazy = load_reference_cache_lazy(tiny_cache, verbose=False)
    m = load_parametric_model(str(MODELS / "parametric_pinn_1_5mill_PINN_500"),
                              which="param_pinn_best.pt", device="cpu")
    capsys.readouterr()
    l2 = score_on_cache(m, lazy, batch_size=2, device="cpu", print_every=5)
    lines = [l for l in capsys.readouterr().out.splitlines() if "score:" in l]
    assert [l.split()[1] for l in lines] == ["6/12", "10/12", "12/12"]
    db = score_db(lazy, l2)
    assert "PHI" not in db and len(db["l2"]) == 12
    db_pair_heatmaps(db, nbins=3, stat="frac", tol=1e-3)
    db_pair_heatmaps(db, nbins=3, stat="median")
    import matplotlib.pyplot as plt
    assert plt.get_fignums() == []   # closed after display -> shown once in Jupyter
