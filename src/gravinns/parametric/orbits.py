"""PN reference-orbit integration for the parametric surrogate.

All Hamiltonian evaluations delegate to
:func:`gravinns.physics.compute_hamiltonian_r`; this module contains no private
copy of the PN series.
"""
import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

from .. import constants as _const
from ..physics.hamiltonian import compute_hamiltonian_r

# Shared Hamiltonian used throughout the repository.
_pham_batch = compute_hamiltonian_r



# ══════════════════════════════════════════════════════════════════════════ #
#  Reference-data generation (small set of 2PN RK orbits) + parametric PDE.    #
# ══════════════════════════════════════════════════════════════════════════ #


# Shared physical normalization.
C_SQ = _const.C_SQ
C_QD = _const.C_QD



def _pham_np(q, p, pn_order, nu):
    """Evaluate the shared PN Hamiltonian for one RK sample."""
    q = torch.as_tensor(q, dtype=torch.float64).view(1, 2)
    p = torch.as_tensor(p, dtype=torch.float64).view(1, 2)
    return compute_hamiltonian_r(q, p, pn_order, nu, C_SQ, C_QD)


def _rhs(pn_order, nu):
    def f(t, Y):
        q = torch.tensor([[Y[0], Y[1]]], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([[Y[2], Y[3]]], dtype=torch.float64, requires_grad=True)
        H = _pham_np(q[0], p[0], pn_order, nu)
        dq = torch.autograd.grad(H.sum(), q, retain_graph=True)[0][0].numpy()
        dp = torch.autograd.grad(H.sum(), p)[0][0].numpy()
        return [dp[0], dp[1], -dq[0], -dq[1]]
    return f


def _grad0(qx,qy,px,py,nu,cs,cq):
    return (qx/(qx**2 + qy**2)**(3/2), qy/(qx**2 + qy**2)**(3/2), px, py)

def _grad1(qx,qy,px,py,nu,cs,cq):
    return (qx/(qx**2 + qy**2)**(3/2) + (-qx/(qx**2 + qy**2)**2 + (1/2)*qx*(nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (nu + 3)*(px**2 + py**2))/(qx**2 + qy**2)**(3/2) - 1/2*(2*nu*px*(px*qx + py*qy)/(qx**2 + qy**2) - 2*nu*qx*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/math.sqrt(qx**2 + qy**2))/cs, qy/(qx**2 + qy**2)**(3/2) + (-qy/(qx**2 + qy**2)**2 + (1/2)*qy*(nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (nu + 3)*(px**2 + py**2))/(qx**2 + qy**2)**(3/2) - 1/2*(2*nu*py*(px*qx + py*qy)/(qx**2 + qy**2) - 2*nu*qy*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/math.sqrt(qx**2 + qy**2))/cs, px + (4*px*((3/8)*nu - 1/8)*(px**2 + py**2) - 1/2*(2*nu*qx*(px*qx + py*qy)/(qx**2 + qy**2) + 2*px*(nu + 3))/math.sqrt(qx**2 + qy**2))/cs, py + (4*py*((3/8)*nu - 1/8)*(px**2 + py**2) - 1/2*(2*nu*qy*(px*qx + py*qy)/(qx**2 + qy**2) + 2*py*(nu + 3))/math.sqrt(qx**2 + qy**2))/cs)

def _grad2(qx,qy,px,py,nu,cs,cq):
    return (qx/(qx**2 + qy**2)**(3/2) + (-qx/(qx**2 + qy**2)**2 + (1/2)*qx*(nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (nu + 3)*(px**2 + py**2))/(qx**2 + qy**2)**(3/2) - 1/2*(2*nu*px*(px*qx + py*qy)/(qx**2 + qy**2) - 2*nu*qx*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/math.sqrt(qx**2 + qy**2))/cs + ((3/4)*qx*(3*nu + 1)/(qx**2 + qy**2)**(5/2) - 4*qx*(3*nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (8*nu + 5)*(px**2 + py**2))/(2*qx**2 + 2*qy**2)**2 - 1/8*qx*(-2*nu**2*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2) - 3*nu**2*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + (px**2 + py**2)**2*(-3*nu**2 - 20*nu + 5))/(qx**2 + qy**2)**(3/2) + (6*nu*px*(px*qx + py*qy)/(qx**2 + qy**2) - 6*nu*qx*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/(2*qx**2 + 2*qy**2) + (1/8)*(-4*nu**2*px*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*px*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*nu**2*qx*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2)**2 + 12*nu**2*qx*(px*qx + py*qy)**4/(qx**2 + qy**2)**3)/math.sqrt(qx**2 + qy**2))/cq, qy/(qx**2 + qy**2)**(3/2) + (-qy/(qx**2 + qy**2)**2 + (1/2)*qy*(nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (nu + 3)*(px**2 + py**2))/(qx**2 + qy**2)**(3/2) - 1/2*(2*nu*py*(px*qx + py*qy)/(qx**2 + qy**2) - 2*nu*qy*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/math.sqrt(qx**2 + qy**2))/cs + ((3/4)*qy*(3*nu + 1)/(qx**2 + qy**2)**(5/2) - 4*qy*(3*nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (8*nu + 5)*(px**2 + py**2))/(2*qx**2 + 2*qy**2)**2 - 1/8*qy*(-2*nu**2*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2) - 3*nu**2*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + (px**2 + py**2)**2*(-3*nu**2 - 20*nu + 5))/(qx**2 + qy**2)**(3/2) + (6*nu*py*(px*qx + py*qy)/(qx**2 + qy**2) - 6*nu*qy*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/(2*qx**2 + 2*qy**2) + (1/8)*(-4*nu**2*py*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*py*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*nu**2*qy*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2)**2 + 12*nu**2*qy*(px*qx + py*qy)**4/(qx**2 + qy**2)**3)/math.sqrt(qx**2 + qy**2))/cq, px + (4*px*((3/8)*nu - 1/8)*(px**2 + py**2) - 1/2*(2*nu*qx*(px*qx + py*qy)/(qx**2 + qy**2) + 2*px*(nu + 3))/math.sqrt(qx**2 + qy**2))/cs + (6*px*(px**2 + py**2)**2*((5/16)*nu**2 - 5/16*nu + 1/16) + (6*nu*qx*(px*qx + py*qy)/(qx**2 + qy**2) + 2*px*(8*nu + 5))/(2*qx**2 + 2*qy**2) + (1/8)*(-4*nu**2*px*(px*qx + py*qy)**2/(qx**2 + qy**2) - 4*nu**2*qx*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*qx*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*px*(px**2 + py**2)*(-3*nu**2 - 20*nu + 5))/math.sqrt(qx**2 + qy**2))/cq, py + (4*py*((3/8)*nu - 1/8)*(px**2 + py**2) - 1/2*(2*nu*qy*(px*qx + py*qy)/(qx**2 + qy**2) + 2*py*(nu + 3))/math.sqrt(qx**2 + qy**2))/cs + (6*py*(px**2 + py**2)**2*((5/16)*nu**2 - 5/16*nu + 1/16) + (6*nu*qy*(px*qx + py*qy)/(qx**2 + qy**2) + 2*py*(8*nu + 5))/(2*qx**2 + 2*qy**2) + (1/8)*(-4*nu**2*py*(px*qx + py*qy)**2/(qx**2 + qy**2) - 4*nu**2*qy*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*qy*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*py*(px**2 + py**2)*(-3*nu**2 - 20*nu + 5))/math.sqrt(qx**2 + qy**2))/cq)

def _grad3(qx,qy,px,py,nu,cs,cq):
    return (qx/(qx**2 + qy**2)**(3/2) + (-qx/(qx**2 + qy**2)**2 + (1/2)*qx*(nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (nu + 3)*(px**2 + py**2))/(qx**2 + qy**2)**(3/2) - 1/2*(2*nu*px*(px*qx + py*qy)/(qx**2 + qy**2) - 2*nu*qx*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/math.sqrt(qx**2 + qy**2))/cs + (2*nu*px*(px*qx + py*qy)*(-7/4*nu - 85/16 - 3/64*math.pi**2)/(qx**2 + qy**2)**(5/2) - 5*nu*qx*(px*qx + py*qy)**2*(-7/4*nu - 85/16 - 3/64*math.pi**2)/(qx**2 + qy**2)**(7/2) - 3*qx*(-23/8*nu**2 + nu*(-485/48 + (1/64)*math.pi**2))*(px**2 + py**2)/(qx**2 + qy**2)**(5/2) - 16*qx*((1/2)*nu*(30*nu + 17)*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2) + (1/4)*nu*(43*nu + 5)*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + (px**2 + py**2)**2*((109/4)*nu**2 + 136*nu - 27))/(8*qx**2 + 8*qy**2)**2 - 4*qx*(nu*(109/12 - 21/32*math.pi**2) + 1/8)/(qx**2 + qy**2)**3 - 1/16*qx*(-5*nu**3*(px*qx + py*qy)**6/(qx**2 + qy**2)**3 + 3*nu**2*(1 - nu)*(px**2 + py**2)*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + nu**2*(2 - 3*nu)*(px**2 + py**2)**2*(px*qx + py*qy)**2/(qx**2 + qy**2) + (px**2 + py**2)**3*(-5*nu**3 - 53*nu**2 + 42*nu - 7))/(qx**2 + qy**2)**(3/2) + (nu*px*(30*nu + 17)*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) + nu*px*(43*nu + 5)*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 - nu*qx*(30*nu + 17)*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2)**2 - nu*qx*(43*nu + 5)*(px*qx + py*qy)**4/(qx**2 + qy**2)**3)/(8*qx**2 + 8*qy**2) + (1/16)*(-30*nu**3*px*(px*qx + py*qy)**5/(qx**2 + qy**2)**3 + 30*nu**3*qx*(px*qx + py*qy)**6/(qx**2 + qy**2)**4 + 12*nu**2*px*(1 - nu)*(px**2 + py**2)*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 2*nu**2*px*(2 - 3*nu)*(px**2 + py**2)**2*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*qx*(1 - nu)*(px**2 + py**2)*(px*qx + py*qy)**4/(qx**2 + qy**2)**3 - 2*nu**2*qx*(2 - 3*nu)*(px**2 + py**2)**2*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/math.sqrt(qx**2 + qy**2))/cs**3 + ((3/4)*qx*(3*nu + 1)/(qx**2 + qy**2)**(5/2) - 4*qx*(3*nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (8*nu + 5)*(px**2 + py**2))/(2*qx**2 + 2*qy**2)**2 - 1/8*qx*(-2*nu**2*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2) - 3*nu**2*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + (px**2 + py**2)**2*(-3*nu**2 - 20*nu + 5))/(qx**2 + qy**2)**(3/2) + (6*nu*px*(px*qx + py*qy)/(qx**2 + qy**2) - 6*nu*qx*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/(2*qx**2 + 2*qy**2) + (1/8)*(-4*nu**2*px*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*px*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*nu**2*qx*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2)**2 + 12*nu**2*qx*(px*qx + py*qy)**4/(qx**2 + qy**2)**3)/math.sqrt(qx**2 + qy**2))/cq, qy/(qx**2 + qy**2)**(3/2) + (-qy/(qx**2 + qy**2)**2 + (1/2)*qy*(nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (nu + 3)*(px**2 + py**2))/(qx**2 + qy**2)**(3/2) - 1/2*(2*nu*py*(px*qx + py*qy)/(qx**2 + qy**2) - 2*nu*qy*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/math.sqrt(qx**2 + qy**2))/cs + (2*nu*py*(px*qx + py*qy)*(-7/4*nu - 85/16 - 3/64*math.pi**2)/(qx**2 + qy**2)**(5/2) - 5*nu*qy*(px*qx + py*qy)**2*(-7/4*nu - 85/16 - 3/64*math.pi**2)/(qx**2 + qy**2)**(7/2) - 3*qy*(-23/8*nu**2 + nu*(-485/48 + (1/64)*math.pi**2))*(px**2 + py**2)/(qx**2 + qy**2)**(5/2) - 16*qy*((1/2)*nu*(30*nu + 17)*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2) + (1/4)*nu*(43*nu + 5)*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + (px**2 + py**2)**2*((109/4)*nu**2 + 136*nu - 27))/(8*qx**2 + 8*qy**2)**2 - 4*qy*(nu*(109/12 - 21/32*math.pi**2) + 1/8)/(qx**2 + qy**2)**3 - 1/16*qy*(-5*nu**3*(px*qx + py*qy)**6/(qx**2 + qy**2)**3 + 3*nu**2*(1 - nu)*(px**2 + py**2)*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + nu**2*(2 - 3*nu)*(px**2 + py**2)**2*(px*qx + py*qy)**2/(qx**2 + qy**2) + (px**2 + py**2)**3*(-5*nu**3 - 53*nu**2 + 42*nu - 7))/(qx**2 + qy**2)**(3/2) + (nu*py*(30*nu + 17)*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) + nu*py*(43*nu + 5)*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 - nu*qy*(30*nu + 17)*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2)**2 - nu*qy*(43*nu + 5)*(px*qx + py*qy)**4/(qx**2 + qy**2)**3)/(8*qx**2 + 8*qy**2) + (1/16)*(-30*nu**3*py*(px*qx + py*qy)**5/(qx**2 + qy**2)**3 + 30*nu**3*qy*(px*qx + py*qy)**6/(qx**2 + qy**2)**4 + 12*nu**2*py*(1 - nu)*(px**2 + py**2)*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 2*nu**2*py*(2 - 3*nu)*(px**2 + py**2)**2*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*qy*(1 - nu)*(px**2 + py**2)*(px*qx + py*qy)**4/(qx**2 + qy**2)**3 - 2*nu**2*qy*(2 - 3*nu)*(px**2 + py**2)**2*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/math.sqrt(qx**2 + qy**2))/cs**3 + ((3/4)*qy*(3*nu + 1)/(qx**2 + qy**2)**(5/2) - 4*qy*(3*nu*(px*qx + py*qy)**2/(qx**2 + qy**2) + (8*nu + 5)*(px**2 + py**2))/(2*qx**2 + 2*qy**2)**2 - 1/8*qy*(-2*nu**2*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2) - 3*nu**2*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + (px**2 + py**2)**2*(-3*nu**2 - 20*nu + 5))/(qx**2 + qy**2)**(3/2) + (6*nu*py*(px*qx + py*qy)/(qx**2 + qy**2) - 6*nu*qy*(px*qx + py*qy)**2/(qx**2 + qy**2)**2)/(2*qx**2 + 2*qy**2) + (1/8)*(-4*nu**2*py*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*py*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*nu**2*qy*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2)**2 + 12*nu**2*qy*(px*qx + py*qy)**4/(qx**2 + qy**2)**3)/math.sqrt(qx**2 + qy**2))/cq, px + (4*px*((3/8)*nu - 1/8)*(px**2 + py**2) - 1/2*(2*nu*qx*(px*qx + py*qy)/(qx**2 + qy**2) + 2*px*(nu + 3))/math.sqrt(qx**2 + qy**2))/cs + (2*nu*qx*(px*qx + py*qy)*(-7/4*nu - 85/16 - 3/64*math.pi**2)/(qx**2 + qy**2)**(5/2) + 2*px*(-23/8*nu**2 + nu*(-485/48 + (1/64)*math.pi**2))/(qx**2 + qy**2)**(3/2) + 8*px*(px**2 + py**2)**3*((35/128)*nu**3 - 35/64*nu**2 + (35/128)*nu - 5/128) + (nu*px*(30*nu + 17)*(px*qx + py*qy)**2/(qx**2 + qy**2) + nu*qx*(30*nu + 17)*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) + nu*qx*(43*nu + 5)*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*px*(px**2 + py**2)*((109/4)*nu**2 + 136*nu - 27))/(8*qx**2 + 8*qy**2) + (1/16)*(-30*nu**3*qx*(px*qx + py*qy)**5/(qx**2 + qy**2)**3 + 6*nu**2*px*(1 - nu)*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + 4*nu**2*px*(2 - 3*nu)*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2) + 12*nu**2*qx*(1 - nu)*(px**2 + py**2)*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 2*nu**2*qx*(2 - 3*nu)*(px**2 + py**2)**2*(px*qx + py*qy)/(qx**2 + qy**2) + 6*px*(px**2 + py**2)**2*(-5*nu**3 - 53*nu**2 + 42*nu - 7))/math.sqrt(qx**2 + qy**2))/cs**3 + (6*px*(px**2 + py**2)**2*((5/16)*nu**2 - 5/16*nu + 1/16) + (6*nu*qx*(px*qx + py*qy)/(qx**2 + qy**2) + 2*px*(8*nu + 5))/(2*qx**2 + 2*qy**2) + (1/8)*(-4*nu**2*px*(px*qx + py*qy)**2/(qx**2 + qy**2) - 4*nu**2*qx*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*qx*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*px*(px**2 + py**2)*(-3*nu**2 - 20*nu + 5))/math.sqrt(qx**2 + qy**2))/cq, py + (4*py*((3/8)*nu - 1/8)*(px**2 + py**2) - 1/2*(2*nu*qy*(px*qx + py*qy)/(qx**2 + qy**2) + 2*py*(nu + 3))/math.sqrt(qx**2 + qy**2))/cs + (2*nu*qy*(px*qx + py*qy)*(-7/4*nu - 85/16 - 3/64*math.pi**2)/(qx**2 + qy**2)**(5/2) + 2*py*(-23/8*nu**2 + nu*(-485/48 + (1/64)*math.pi**2))/(qx**2 + qy**2)**(3/2) + 8*py*(px**2 + py**2)**3*((35/128)*nu**3 - 35/64*nu**2 + (35/128)*nu - 5/128) + (nu*py*(30*nu + 17)*(px*qx + py*qy)**2/(qx**2 + qy**2) + nu*qy*(30*nu + 17)*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) + nu*qy*(43*nu + 5)*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*py*(px**2 + py**2)*((109/4)*nu**2 + 136*nu - 27))/(8*qx**2 + 8*qy**2) + (1/16)*(-30*nu**3*qy*(px*qx + py*qy)**5/(qx**2 + qy**2)**3 + 6*nu**2*py*(1 - nu)*(px*qx + py*qy)**4/(qx**2 + qy**2)**2 + 4*nu**2*py*(2 - 3*nu)*(px**2 + py**2)*(px*qx + py*qy)**2/(qx**2 + qy**2) + 12*nu**2*qy*(1 - nu)*(px**2 + py**2)*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 2*nu**2*qy*(2 - 3*nu)*(px**2 + py**2)**2*(px*qx + py*qy)/(qx**2 + qy**2) + 6*py*(px**2 + py**2)**2*(-5*nu**3 - 53*nu**2 + 42*nu - 7))/math.sqrt(qx**2 + qy**2))/cs**3 + (6*py*(px**2 + py**2)**2*((5/16)*nu**2 - 5/16*nu + 1/16) + (6*nu*qy*(px*qx + py*qy)/(qx**2 + qy**2) + 2*py*(8*nu + 5))/(2*qx**2 + 2*qy**2) + (1/8)*(-4*nu**2*py*(px*qx + py*qy)**2/(qx**2 + qy**2) - 4*nu**2*qy*(px**2 + py**2)*(px*qx + py*qy)/(qx**2 + qy**2) - 12*nu**2*qy*(px*qx + py*qy)**3/(qx**2 + qy**2)**2 + 4*py*(px**2 + py**2)*(-3*nu**2 - 20*nu + 5))/math.sqrt(qx**2 + qy**2))/cq)

_GRADS = {0: _grad0, 1: _grad1, 2: _grad2, 3: _grad3}


def _rhs_fast(pn_order, nu):
    """Exact PN Hamilton RHS via symbolic gradients (no autograd, no torch)."""
    g = _GRADS[int(pn_order)]
    cs, cq = C_SQ, C_QD
    def f(t, Y):
        gqx, gqy, gpx, gpy = g(Y[0], Y[1], Y[2], Y[3], nu, cs, cq)
        return [gpx, gpy, -gqx, -gqy]
    return f


_ORBIT_CACHE = {}

def integrate_orbit_cached(p0f, r0km, pr0f, nu, pn_order, n_orbits, n_pts):
    """integrate_orbit with memoisation. Reference orbits never change during
    training, so re-integrating them every heatmap refresh is pure waste."""
    key = (round(float(p0f), 10), round(float(r0km), 10), round(float(pr0f), 10),
           round(float(nu), 10), int(pn_order), int(n_orbits), int(n_pts))
    hit = _ORBIT_CACHE.get(key)
    if hit is None:
        hit = integrate_orbit(p0f, r0km, pr0f, nu, pn_order, n_orbits, n_pts)
        _ORBIT_CACHE[key] = hit
    return hit


def integrate_orbit(p0f, r0km, pr0f, nu, pn_order, n_orbits, n_pts, rtol=1e-9, atol=1e-11):
    """Integrate a reference orbit and return angle-domain diagnostics."""
    q0 = r0km / 200.0
    p0t = p0f / np.sqrt(q0)
    pr0 = pr0f
    y0 = np.ascontiguousarray([q0, 0.0, pr0, p0t], dtype=np.float64)
    L0 = q0 * p0t

    # Energy and orbital period (Newtonian approximation)
    E = 0.5 * (pr0**2 + p0t**2) - 1.0 / q0
    a = -0.5 / E
    T = 2 * np.pi * a**1.5
    t_max = n_orbits * T

    t_eval = np.ascontiguousarray(np.linspace(0, t_max, n_pts), dtype=np.float64)
    t_span = np.ascontiguousarray([0.0, t_max], dtype=np.float64)

    # Get the RHS function (it should return a callable)
    rhs_func = _rhs_fast(pn_order, nu)

    # DOP853 is robust for the multi-orbit reference integrations used here.
    sol = solve_ivp(rhs_func, t_span, y0, t_eval=t_eval, method='DOP853',
                    rtol=rtol, atol=atol)
    # Extract the solution
    x, y = sol.y[0], sol.y[1]
    r = np.hypot(x, y)
    phi = np.unwrap(np.arctan2(y, x))
    phi -= phi[0]
    pr = (x * sol.y[2] + y * sol.y[3]) / r

    return dict(
        phi=phi,
        u=1.0 / r,
        pr=pr,
        L0=L0,
        q0=q0,
        p0t=p0t,
        pr0=pr0,
        phi_max=phi[-1],
        u_min=float((1.0 / r).min()),
        u_max=float((1.0 / r).max()),
        pr_absmax=float(np.abs(pr).max()),
    )



def build_reference_set(anchor_thetas, data_pn_order, n_orbits, n_pts,
                        phi_max_global):
    """Integrate a 2PN (or data_pn_order) orbit at each anchor theta and
    resample u,pr onto a COMMON phi grid [0, phi_max_global].
    Returns tensors: THETA (K,4), U (K,n_pts), PR (K,n_pts), PHI (n_pts,)."""
    phi_grid = np.linspace(0, phi_max_global, n_pts)
    U, PR, TH = [], [], []
    meta = []
    for (p0f, r0km, pr0f, nu) in anchor_thetas:
        o = integrate_orbit(p0f, r0km, pr0f, nu, data_pn_order, n_orbits, n_pts)
        # resample onto common grid (clip beyond this orbit's own phi_max)
        u = np.interp(phi_grid, o["phi"], o["u"])
        pr = np.interp(phi_grid, o["phi"], o["pr"])
        U.append(u); PR.append(pr); TH.append([p0f, r0km, pr0f, nu])
        meta.append(o)
    return (torch.tensor(np.array(TH), dtype=torch.float32),
            torch.tensor(np.array(U), dtype=torch.float32),
            torch.tensor(np.array(PR), dtype=torch.float32),
            torch.tensor(phi_grid, dtype=torch.float32),
            meta)


# ── Parametric PDE residual (the physics loss) ─────────────────────────────
def phi_residual_param(model, phi, theta, pn_order, c_sq, c_qd):
    """Angle-domain Hamilton residual for a batch of (phi, theta).

    Mirrors EXACTLY the verified single-orbit reduction: build the full
    Hamilton equations dq/dt=dH/dp, dp/dt=-dH/dq, then form dphi/dt, du/dt,
    dpr/dt and divide by dphi/dt. Conservative only (L frozen).

    phi: (N,1) requires_grad; theta: (N,4). Returns R_u,R_pr (N,), H (N,)."""
    out = model(phi, theta)              # (N,3): u, pr, L
    u = out[:, 0:1]; pr = out[:, 1:2]; L = out[:, 2:3]
    du_dphi  = torch.autograd.grad(u.sum(),  phi, create_graph=True)[0]
    dpr_dphi = torch.autograd.grad(pr.sum(), phi, create_graph=True)[0]
    nu = theta[:, 3:4]
    r = 1.0/u; cs = torch.cos(phi); sn = torch.sin(phi)
    qx = r*cs; qy = r*sn
    px = pr*cs - (L*u)*sn; py = pr*sn + (L*u)*cs
    q = torch.cat([qx, qy], dim=1).requires_grad_(True)
    p = torch.cat([px, py], dim=1).requires_grad_(True)
    H = _pham_batch_local(q, p, pn_order, nu[:, 0], c_sq, c_qd)
    dHdq, dHdp = torch.autograd.grad(H.sum(), [q, p], create_graph=True)
    dqx_dt = dHdp[:, 0:1]; dqy_dt = dHdp[:, 1:2]
    dpx_dt = -dHdq[:, 0:1]; dpy_dt = -dHdq[:, 1:2]
    dphi_dt = (qx*dqy_dt - qy*dqx_dt)/r**2
    dr_dt   = (qx*dqx_dt + qy*dqy_dt)/r
    du_dt   = -(u**2)*dr_dt
    qp_dot  = dqx_dt*px + qx*dpx_dt + dqy_dt*py + qy*dpy_dt
    dpr_dt  = qp_dot/r - pr*dr_dt/r
    R_u  = du_dphi  - du_dt /dphi_dt
    R_pr = dpr_dphi - dpr_dt/dphi_dt
    return R_u[:, 0], R_pr[:, 0], H



# identical (AST-checked) to the shared Hamiltonian -> reuse it
_pham_batch_local = compute_hamiltonian_r
