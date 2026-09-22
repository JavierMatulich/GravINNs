import numpy as np

from gravinns.sweep import (ecc_bin, generate_ics_stratified, ic_diagnostics,
                            wilson_interval)
from gravinns.sweep.ic_generation import _p0_for_ecc


def test_apsis_eccentricity_closed_form():
    d = ic_diagnostics(0.2, 0.9, 200.0, 0.0)
    assert d["bound"]
    assert abs(d["ecc"] - (1 - 0.9**2)) < 1e-12


def test_p0_solver_off_apsis():
    p0 = _p0_for_ecc(0.2, 250.0, 0.3, 0.40)
    assert p0 is not None
    assert abs(ic_diagnostics(0.2, p0, 250.0, 0.3)["ecc"] - 0.40) < 1e-6


def test_stratified_generation_is_deterministic_and_binned():
    a, _ = generate_ics_stratified(n_per_bin=5, seed=987, ecc_targets=(0.027,),
                                   pr0_range=(-0.003, 0.003), min_rp_rg=20.0,
                                   verbose=False)
    b, _ = generate_ics_stratified(n_per_bin=5, seed=987, ecc_targets=(0.027,),
                                   pr0_range=(-0.003, 0.003), min_rp_rg=20.0,
                                   verbose=False)
    assert [d["p0_factor"] for d in a] == [d["p0_factor"] for d in b]
    assert all(ecc_bin(d["ecc"]) == d["ecc_bin"] for d in a)


def test_wilson():
    lo, hi = wilson_interval(8, 18)
    assert 0.23 < lo < 0.25 and 0.65 < hi < 0.67
    assert np.isnan(wilson_interval(0, 0)[0])
