import math

import numpy as np
import torch

import gravinns.constants as const
from gravinns.physics import (
    compute_hamiltonian_r, compute_hamiltonian_numpy,
    compute_hamiltonian_polar_numpy, F_RR_np, F_RR_torch,
)


def test_newtonian_limit():
    q = torch.tensor([[1.3, -0.4]], dtype=torch.float64)
    p = torch.tensor([[0.2, 0.7]], dtype=torch.float64)
    H0 = compute_hamiltonian_r(q, p, 0, 0.25, const.C_SQ, const.C_QD)
    r = math.hypot(1.3, -0.4)
    expected = 0.5 * (0.2**2 + 0.7**2) - 1.0 / math.sqrt(r**2 + 1e-8)
    assert abs(H0.item() - expected) < 1e-12


def test_pn_corrections_shrink_with_order():
    q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    p = torch.tensor([[0.0, 0.8]], dtype=torch.float64)
    H = [compute_hamiltonian_r(q, p, n, 0.25, const.C_SQ, const.C_QD).item()
         for n in range(4)]
    d = [abs(H[i + 1] - H[i]) for i in range(3)]
    assert d[0] > d[1] > d[2] > 0


def test_radiation_reaction_numpy_matches_torch():
    args = (1.1, 0.3, -0.05, 0.9)
    fx, fy = F_RR_np(*args, 0.25, math.sqrt(const.C_SQ))
    tx, ty = F_RR_torch(*[torch.tensor(a, dtype=torch.float64) for a in args],
                        0.25, math.sqrt(const.C_SQ))
    assert abs(fx - tx.item()) < 1e-12 and abs(fy - ty.item()) < 1e-12


def test_constants_match_notebook():
    assert abs(const.C_SQ - 4.835850e+01) < 1e-4
    assert abs(const.R0_REF_KM - 200.0) < 1e-12
    assert abs(const.NU - 0.25) < 1e-12


def test_numpy_hamiltonian_wrapper_matches_canonical():
    q = np.array([[1.3, -0.4], [0.9, 0.2]], dtype=float)
    p = np.array([[0.2, 0.7], [-0.1, 0.85]], dtype=float)
    expected = compute_hamiltonian_r(
        torch.tensor(q, dtype=torch.float64),
        torch.tensor(p, dtype=torch.float64),
        3, 0.21, const.C_SQ, const.C_QD,
    ).detach().numpy()
    actual = compute_hamiltonian_numpy(q, p, 3, 0.21, const.C_SQ, const.C_QD)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


def test_polar_hamiltonian_wrapper_matches_canonical():
    r = np.array([0.8, 1.2, 1.7])
    pr = np.array([-0.1, 0.0, 0.12])
    L = 0.9
    q = np.column_stack([r, np.zeros_like(r)])
    p = np.column_stack([pr, L / r])
    expected = compute_hamiltonian_numpy(q, p, 2, 0.25, const.C_SQ, const.C_QD)
    actual = compute_hamiltonian_polar_numpy(
        r, pr, L, 0.25, 2, const.C_SQ, const.C_QD
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
