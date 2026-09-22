"""Canonical ADM Hamiltonian for the relative two-body problem, 0PN--3PN.

This module is the single source of truth for the PN Hamiltonian used by the
trainers, RK reference integrations, adaptive regime diagnosis, and long-orbit
post-processing. Code units use ``G M = 1`` with momenta per reduced mass.
"""
import math as _math

import torch


def compute_hamiltonian_r(q, p, pn_order, nu, c_sq, c_qd):
    """H(q, p) truncated at ``pn_order`` (0, 1, 2 or 3).

    q, p : tensors of shape (N, 2).  Returns a tensor of shape (N,).
    """
    r2 = q[:, 0]**2 + q[:, 1]**2
    r = torch.sqrt(r2 + 1e-8)
    p2 = p[:, 0]**2 + p[:, 1]**2
    pr = (q[:, 0]*p[:, 0] + q[:, 1]*p[:, 1]) / r
    H = 0.5*p2 - 1.0/r
    if pn_order == 0:
        return H
    H1 = ((1/8)*(3*nu-1)*p2**2
          - (1/(2*r))*((3+nu)*p2 + nu*pr**2) + 0.5/r2)
    H1pn = H + H1/c_sq
    if pn_order == 1:
        return H1pn
    H2 = ((1/16)*(1-5*nu+5*nu**2)*p2**3
          + (1/(8*r))*((5-20*nu-3*nu**2)*p2**2
                       - 2*nu**2*p2*pr**2 - 3*nu**2*pr**4)
          + (1/(2*r**2))*((5+8*nu)*p2 + 3*nu*pr**2)
          - (1/(4*r**3))*(1+3*nu))
    H2pn = H1pn + H2/c_qd
    if pn_order == 2:
        return H2pn
    _pi2 = _math.pi**2
    c6 = c_sq**3
    H3 = ((1/128)*(-5+35*nu-70*nu**2+35*nu**3)*p2**4
          + (1/(16*r))*((-7+42*nu-53*nu**2-5*nu**3)*p2**3
                        + nu**2*(2-3*nu)*pr**2*p2**2
                        + 3*nu**2*(1-nu)*pr**4*p2
                        - 5*nu**3*pr**6)
          + (1/(8*r2))*((-27+136*nu+109*nu**2/4)*p2**2
                        + (17+30*nu)*nu*pr**2*p2/2
                        + (5+43*nu)*nu*pr**4/4)
          + (1/r**3)*((-25/8+(1/64)*_pi2-335/48)*nu-(23/8)*nu**2)*p2
          + (1/r**3)*((-85/16-(3/64)*_pi2-(7/4)*nu)*nu*pr**2)
          + (1/r**4)*(1/8+(109/12-(21/32)*_pi2)*nu))
    return H2pn + H3/c6

def compute_hamiltonian_numpy(q, p, pn_order, nu, c_sq, c_qd):
    """NumPy-facing wrapper around :func:`compute_hamiltonian_r`.

    This keeps the PN Hamiltonian formula defined in one place while allowing
    SciPy/RK utilities to work with NumPy arrays. The returned value is a
    NumPy array for batched input and a scalar-shaped array for one sample.
    """
    import numpy as _np

    q_arr = _np.asarray(q, dtype=_np.float64)
    p_arr = _np.asarray(p, dtype=_np.float64)
    q2 = _np.atleast_2d(q_arr)
    p2 = _np.atleast_2d(p_arr)
    qt = torch.as_tensor(q2, dtype=torch.float64)
    pt = torch.as_tensor(p2, dtype=torch.float64)
    with torch.no_grad():
        out = compute_hamiltonian_r(qt, pt, pn_order, nu, c_sq, c_qd).cpu().numpy()
    if q_arr.ndim == 1:
        return out[0]
    return out


def compute_hamiltonian_polar_numpy(r, pr, L, nu, pn_order, c_sq, c_qd):
    """Evaluate the shared PN Hamiltonian in polar variables.

    The conversion uses ``q=(r,0)`` and ``p=(pr,L/r)`` and then delegates to
    :func:`compute_hamiltonian_r`. Scalar and NumPy-array inputs are supported.
    """
    import numpy as _np

    r_arr, pr_arr = _np.broadcast_arrays(
        _np.asarray(r, dtype=_np.float64),
        _np.asarray(pr, dtype=_np.float64),
    )
    flat_r = r_arr.reshape(-1)
    flat_pr = pr_arr.reshape(-1)
    q = _np.column_stack([flat_r, _np.zeros_like(flat_r)])
    p = _np.column_stack([flat_pr, _np.asarray(L, dtype=_np.float64) / flat_r])
    vals = compute_hamiltonian_numpy(q, p, pn_order, nu, c_sq, c_qd)
    vals = _np.asarray(vals).reshape(r_arr.shape)
    return float(vals) if vals.ndim == 0 else vals

