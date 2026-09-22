"""Physics residuals for angle-domain post-Newtonian orbit PINNs."""
from __future__ import annotations

import torch

from ..physics.hamiltonian import compute_hamiltonian_r
from ..physics.radiation_reaction import F_RR_torch


def angle_residual(
    model,
    phi: torch.Tensor,
    *,
    dissipative: bool,
    target_pn_order: int,
    nu: float,
    csq: float,
    cqd: float,
    c_scalar: float,
    include_energy_balance: bool = False,
):
    """Evaluate the angle-domain equations of motion.

    Returns ``(R_u, R_pr, R_L, dphi_dt, H)`` for the conservative/v2-v3
    trainers.  When ``include_energy_balance`` is true, the 2.5PN balance-law
    residual ``R_E`` is appended as a sixth value.
    """
    out = model(phi)
    u, pr, L = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    du_dphi = torch.autograd.grad(u.sum(), phi, create_graph=True)[0]
    dpr_dphi = torch.autograd.grad(pr.sum(), phi, create_graph=True)[0]
    dL_dphi = (
        torch.autograd.grad(L.sum(), phi, create_graph=True)[0]
        if dissipative
        else None
    )

    r = 1.0 / u
    c, s = torch.cos(phi), torch.sin(phi)
    qx, qy = r * c, r * s
    px = pr * c - (L * u) * s
    py = pr * s + (L * u) * c
    q = torch.cat([qx, qy], dim=1).requires_grad_(True)
    p = torch.cat([px, py], dim=1).requires_grad_(True)

    H = compute_hamiltonian_r(q, p, target_pn_order, nu, csq, cqd)
    dH_dq, dH_dp = torch.autograd.grad(H.sum(), [q, p], create_graph=True)
    dqx_dt, dqy_dt = dH_dp[:, 0:1], dH_dp[:, 1:2]
    dpx_dt, dpy_dt = -dH_dq[:, 0:1], -dH_dq[:, 1:2]

    Fx = Fy = None
    if dissipative:
        Fx, Fy = F_RR_torch(qx, qy, px, py, nu, c_scalar)
        dpx_dt = dpx_dt + Fx
        dpy_dt = dpy_dt + Fy

    dphi_dt = (qx * dqy_dt - qy * dqx_dt) / r**2
    dr_dt = (qx * dqx_dt + qy * dqy_dt) / r
    du_dt = -(u**2) * dr_dt
    qp_dot = dqx_dt * px + qx * dpx_dt + dqy_dt * py + qy * dpy_dt
    dpr_dt = qp_dot / r - pr * dr_dt / r
    dL_dt = dqx_dt * py + qx * dpy_dt - dqy_dt * px - qy * dpx_dt

    R_u = du_dphi - du_dt / dphi_dt
    R_pr = dpr_dphi - dpr_dt / dphi_dt
    R_L = dL_dphi - dL_dt / dphi_dt if dissipative else dL_dt

    if not include_energy_balance:
        return R_u, R_pr, R_L, dphi_dt, H

    if dissipative:
        dH_dphi = torch.autograd.grad(H.sum(), phi, create_graph=True)[0]
        power_rr = Fx * dqx_dt + Fy * dqy_dt
        R_E = dH_dphi - power_rr / dphi_dt
    else:
        R_E = torch.zeros_like(R_u)
    return R_u, R_pr, R_L, dphi_dt, H, R_E
