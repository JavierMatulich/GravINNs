"""2.5PN radiation-reaction force (Burke-Thorne / Iyer-Will gauge form).

Origin: ``_F_RR_torch`` (module level) and ``_FRR_np`` (inside the trainer)
in notebook cell 1.  Both share the same expression; one works on torch
tensors (training residual), the other on floats (RK45 references).
"""
import numpy as np
import torch


def F_RR_torch(qx, qy, px, py, nu, c):
    r = torch.sqrt(qx**2 + qy**2 + 1e-14)
    nx, ny = qx/r, qy/r
    vx, vy = px, py
    rdot = vx*nx + vy*ny
    v2 = vx**2 + vy**2
    A = 3*v2 + (17.0/3.0)/r
    B = v2 + 3.0/r
    pref = (8.0/5.0)*(nu/c**5)/r**3
    return pref*(A*rdot*nx - B*vx), pref*(A*rdot*ny - B*vy)


def F_RR_np(qx, qy, px, py, nu, c):
    r = np.hypot(qx, qy)
    nx, ny = qx/r, qy/r
    rdot = px*nx + py*ny
    v2 = px**2 + py**2
    A = 3*v2 + (17/3)/r
    B = v2 + 3/r
    pref = (8/5)*(nu/c**5)/r**3
    return pref*(A*rdot*nx - B*px), pref*(A*rdot*ny - B*py)
