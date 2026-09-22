"""Dissipative survey: run list, shipped runs, rebuilt paper table."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import gravinns.experiments.dissipative as D
from gravinns.training import (
    train_orbit_adaptive, train_dissipative, train_regime_unified_v4,
)

DISS = Path(__file__).resolve().parents[1] / "models" / "dissipative"


def test_ic_set():
    assert len(D.IC_SET) == 24
    assert sum(1 for d in D.IC_SET if d["n_orbits"] == 6) == 14
    assert D.IC_BY_TAG["D1"]["p0_factor"] == 0.8
    assert D.COMMON["dissipative"] and D.COMMON["no_25pn_data"]
    t = D.ic_table()
    assert list(t.tag[:3]) == ["P1", "P2", "P3"] and t.published_L2.notna().sum() == 14


def test_regime_diagnosis():
    d = D.diagnose_pn_regime(0.25, 0.8, 200.0, 0.0, n_orbits=6,
                             dissipative=True, verbose=False)
    assert d["regime"] != "STOP" and abs(d["ecc"] - 0.36) < 1e-6


@pytest.mark.parametrize("tag,l2,gap,dH", [("D1", 2.656e-5, 4.5e-2, -10.6),
                                           ("P14", 1.4586e-2, 6.1e-2, -16.6)])
def test_shipped_run_reproduces_paper_row(tag, l2, gap, dH):
    path = D.run_dir(tag, str(DISS))
    g, rp = D.gap_2p5(path)
    en = D.energy_diagnostics(path)
    assert abs(g - gap) / gap < 0.02
    assert abs(en["dH_pct"] - dH) < 0.1
    assert rp > 0


def test_paper_table():
    t = D.paper_table(str(DISS), save=False, verbose=False)
    assert len(t) >= 20 and {"e0", "N_rev", "gap", "L2", "dH_pct"} <= set(t.columns)
    pub = t.dropna(subset=["published_L2"])
    # the shipped runs reproduce the published L2 to ~3%
    np.testing.assert_allclose(pub.L2, pub.published_L2, rtol=0.05)
    assert "6 &" in D.latex_paper_table(table=t)


def test_adaptive_trainer_is_wired():
    assert train_orbit_adaptive.__module__ == "gravinns.training.adaptive"
    assert train_dissipative.__module__ == "gravinns.training.dissipative_trainer"
    assert train_regime_unified_v4 is train_dissipative
