"""Reference-orbit construction shared by the single-orbit trainers.

This module contains deterministic numerical helpers only.  It deliberately
keeps training policy out of the reference construction so the v2, v3 and v4
trainers can share the same Hamiltonian/RK machinery while retaining their
version-specific optimization schedules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ..physics.hamiltonian import compute_hamiltonian_r
from ..physics.radiation_reaction import F_RR_np


@dataclass
class OrbitSlice:
    """Minimal ``solve_ivp``-like view used for a trimmed warm-start orbit."""

    y: np.ndarray
    t: np.ndarray


@dataclass
class ReferenceOrbits:
    t_eval: np.ndarray
    sol_0pn: Optional[object]
    sol_1pn: Optional[object]
    sol_2pn: object
    sol_25pn: object
    sol_3pn: Optional[object]
    l2_23: float
    r_pn_min: float
    early_exit_l2: Optional[float] = None


def _hamiltonian_gradient(state, pn_order: int, nu: float, csq: float, cqd: float):
    qx, qy, px, py = state
    q = torch.tensor([[qx, qy]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[px, py]], dtype=torch.float64, requires_grad=True)
    H = compute_hamiltonian_r(q, p, pn_order, nu, csq, cqd)
    dH_dq = torch.autograd.grad(H.sum(), q, retain_graph=True)[0][0].detach().numpy()
    dH_dp = torch.autograd.grad(H.sum(), p)[0][0].detach().numpy()
    return dH_dq, dH_dp


def newtonian_rhs(_t, state):
    qx, qy, px, py = state
    r = np.hypot(qx, qy) + 1e-12
    return [px, py, -qx / r**3, -qy / r**3]


def make_pn_rhs(
    *,
    pn_order: int,
    nu: float,
    csq: float,
    cqd: float,
    dissipative: bool = False,
    c_scalar: float | None = None,
):
    """Return a SciPy RHS for a conservative PN orbit or a 2.5PN inspiral."""

    def rhs(_t, state):
        dH_dq, dH_dp = _hamiltonian_gradient(state, pn_order, nu, csq, cqd)
        ax, ay = -dH_dq[0], -dH_dq[1]
        if dissipative:
            if c_scalar is None:
                raise ValueError("c_scalar is required for dissipative reference integration")
            Fx, Fy = F_RR_np(*state, nu, c_scalar)
            ax += Fx
            ay += Fy
        return [dH_dp[0], dH_dp[1], ax, ay]

    return rhs


def orbit_phase(solution) -> np.ndarray:
    phi = np.unwrap(np.arctan2(solution.y[1], solution.y[0]))
    return phi - phi[0]


def orbit_l2_4d(sol_a, sol_b, angular_momentum: float, n_grid: int = 2000) -> float:
    """Relative 4D phase-space L2 distance on a shared angle grid."""
    pha, phb = orbit_phase(sol_a), orbit_phase(sol_b)
    ra = np.hypot(sol_a.y[0], sol_a.y[1])
    rb = np.hypot(sol_b.y[0], sol_b.y[1])
    pra = (sol_a.y[0] * sol_a.y[2] + sol_a.y[1] * sol_a.y[3]) / ra
    prb = (sol_b.y[0] * sol_b.y[2] + sol_b.y[1] * sol_b.y[3]) / rb
    phi = np.linspace(0, min(pha[-1], phb[-1]), n_grid)
    ua = np.interp(phi, pha, 1 / ra)
    ub = np.interp(phi, phb, 1 / rb)
    pai = np.interp(phi, pha, pra)
    pbi = np.interp(phi, phb, prb)

    qxa, qya = np.cos(phi) / ua, np.sin(phi) / ua
    qxb, qyb = np.cos(phi) / ub, np.sin(phi) / ub
    pxa = pai * np.cos(phi) - (angular_momentum * ua) * np.sin(phi)
    pya = pai * np.sin(phi) + (angular_momentum * ua) * np.cos(phi)
    pxb = pbi * np.cos(phi) - (angular_momentum * ub) * np.sin(phi)
    pyb = pbi * np.sin(phi) + (angular_momentum * ub) * np.cos(phi)

    ref_norm = np.linalg.norm(np.stack([qxb, qyb, pxb, pyb])) + 1e-16
    diff = np.stack([qxa - qxb, qya - qyb, pxa - pxb, pya - pyb])
    return float(np.linalg.norm(diff) / ref_norm)


def periapsis_precession(solution) -> float:
    r = np.hypot(solution.y[0], solution.y[1])
    phi = np.unwrap(np.arctan2(solution.y[1], solution.y[0]))
    peaks = [i for i in range(1, len(r) - 1) if r[i] < r[i - 1] and r[i] < r[i + 1]]
    if len(peaks) < 2:
        return float("nan")
    increments = [phi[peaks[k]] - phi[peaks[k - 1]] - 2 * np.pi for k in range(1, len(peaks))]
    return float(np.mean(increments))


def _bounded_for_plot(solution, a: float, ecc: float) -> bool:
    return np.hypot(solution.y[0], solution.y[1]).max() < 5 * a * (1 + ecc)


def _print_pn_hierarchy(sol_0pn, sol_1pn, sol_2pn, sol_3pn, *, L0, a, ecc, csq, l2_23):
    try:
        have_1pn = sol_1pn is not None
        g01 = orbit_l2_4d(sol_0pn, sol_1pn, L0) if (sol_0pn is not None and have_1pn) else None
        g12 = orbit_l2_4d(sol_1pn, sol_2pn, L0) if have_1pn else None
        rmin = a * (1 - ecc)
        xrel = (2.0 / rmin - 1.0 / a) / csq
        print("  ── PN convergence hierarchy (4D-L2 successive gaps) ──")
        print(f"     relativistic parameter  x = v^2/c^2|_peri = {xrel:.4e}")
        if not have_1pn:
            print("     L2(0PN->1PN) = N/A   (1PN orbit is UNBOUNDED at these initial")
            print("                            conditions — the comparison doesn't apply)")
            print("     L2(1PN->2PN) = N/A   (same reason)")
        else:
            print(f"     L2(0PN->1PN) = {g01:.4e}")
            print(f"     L2(1PN->2PN) = {g12:.4e}   ratio g12/g01 = {g12/g01:.3e}")
        print(
            f"     L2(2PN->3PN) = {l2_23:.4e}"
            + (f"   ratio g23/g12 = {l2_23/g12:.3e}" if have_1pn else "")
        )
        pr1 = periapsis_precession(sol_1pn) if have_1pn else None
        pr2 = periapsis_precession(sol_2pn)
        pr3 = periapsis_precession(sol_3pn)
        pr1_str = f"{pr1:+.3e}" if have_1pn else "N/A"
        print(
            f"     perihelion precession/orbit:  1PN={pr1_str}  "
            f"2PN={pr2:+.3e}  3PN={pr3:+.3e} rad"
        )
        print(f"     precession increment 2PN->3PN = {pr3-pr2:+.3e} rad/orbit")
        if have_1pn and (l2_23 / g12 < 0.05):
            print(f"     >> CRITERION: g23/g12 = {l2_23/g12:.2e} < 0.05  =>  PN series has")
            print("        effectively CONVERGED by 2PN. The 3PN orbit is genuinely almost")
            print("        identical to 2PN here (coefficient near-cancellation at low x).")
            print("        This is correct physics, not a bug. To see a larger 3PN effect,")
            print("        use a more relativistic orbit (smaller r0 or higher eccentricity).")
    except Exception as exc:
        print(f"  (PN-hierarchy diagnostic skipped: {exc})")


def build_reference_orbits(
    *,
    y0,
    nu: float,
    csq: float,
    cqd: float,
    c_scalar: float,
    t_max: float,
    n_ref: int,
    dissipative: bool,
    target_pn_order: int,
    pn_valid_thresh: float,
    a: float,
    ecc: float,
    L0: float,
    l2_target: float,
    two_pn_rtol: float = 1e-10,
    two_pn_atol: float = 1e-12,
) -> ReferenceOrbits:
    """Integrate the reference PN hierarchy used by all three trainers."""
    t_eval = np.linspace(0, t_max, int(n_ref))
    print("  Building references ...", end=" ", flush=True)

    r_pn_min = 1.0 / (pn_valid_thresh * csq)

    def plunge_event(_t, state):
        return np.hypot(state[0], state[1]) - r_pn_min

    plunge_event.terminal = True
    plunge_event.direction = -1

    rhs_2pn = make_pn_rhs(pn_order=2, nu=nu, csq=csq, cqd=cqd)
    rhs_25pn = make_pn_rhs(
        pn_order=2,
        nu=nu,
        csq=csq,
        cqd=cqd,
        dissipative=dissipative,
        c_scalar=c_scalar,
    )
    sol_2pn = solve_ivp(rhs_2pn, [0, t_max], y0, t_eval=t_eval, rtol=two_pn_rtol, atol=two_pn_atol)
    sol_25pn = solve_ivp(
        rhs_25pn,
        [0, t_max],
        y0,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        events=plunge_event if dissipative else None,
    )

    sol_1pn = None
    try:
        trial = solve_ivp(
            make_pn_rhs(pn_order=1, nu=nu, csq=csq, cqd=cqd),
            [0, t_max],
            y0,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-11,
        )
        if _bounded_for_plot(trial, a, ecc):
            sol_1pn = trial
    except Exception as exc:
        print(f"  (1PN reference orbit skipped: {exc})")

    sol_0pn = None
    try:
        trial = solve_ivp(newtonian_rhs, [0, t_max], y0, t_eval=t_eval, rtol=1e-9, atol=1e-11)
        if _bounded_for_plot(trial, a, ecc):
            sol_0pn = trial
    except Exception as exc:
        print(f"  (0PN reference skipped: {exc})")

    sol_3pn = None
    l2_23 = float("inf")
    early_exit_l2 = None
    if target_pn_order >= 3 and not dissipative:
        sol_3pn = solve_ivp(
            make_pn_rhs(pn_order=3, nu=nu, csq=csq, cqd=cqd),
            [0, t_max],
            y0,
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
        )
        l2_23 = orbit_l2_4d(sol_2pn, sol_3pn, L0, n_grid=1000)
        p2, p3 = orbit_phase(sol_2pn), orbit_phase(sol_3pn)
        print(f"  3PN RK orbit integrated: {sol_3pn.y.shape[1]} pts")
        print(
            f"  2PN-vs-3PN orbit difference (4D L2): {l2_23:.3e} "
            f"(phi_max: 2PN={p2[-1]:.2f} 3PN={p3[-1]:.2f})"
        )
        _print_pn_hierarchy(sol_0pn, sol_1pn, sol_2pn, sol_3pn, L0=L0, a=a, ecc=ecc, csq=csq, l2_23=l2_23)
        if l2_23 < l2_target:
            print(f"  [EARLY EXIT] 2PN-3PN gap ({l2_23:.2e}) < l2_target ({l2_target:.0e}).")
            print("  The 3PN correction is negligible at these orbital parameters.")
            print("  Returning the 2PN orbit as the solution (no training needed).")
            early_exit_l2 = l2_23

    return ReferenceOrbits(
        t_eval=t_eval,
        sol_0pn=sol_0pn,
        sol_1pn=sol_1pn,
        sol_2pn=sol_2pn,
        sol_25pn=sol_25pn,
        sol_3pn=sol_3pn,
        l2_23=l2_23,
        r_pn_min=r_pn_min,
        early_exit_l2=early_exit_l2,
    )


def select_warmstart_orbit(
    *,
    warmstart_pn_order,
    target_pn_order: int,
    sol_2pn,
    y0,
    t_max: float,
    nu: float,
    csq: float,
    cqd: float,
    a: float,
    ecc: float,
):
    """Select and trim the lower-order orbit used for Stage-1 transfer."""
    phi_2pn = orbit_phase(sol_2pn)
    phi_max_2pn = float(phi_2pn[-1])
    r_apo_safe = a * (1 + ecc) * 2.5

    choice = str(warmstart_pn_order).lower().strip()
    if choice == "auto":
        choice = {2: "1pn", 3: "2pn"}.get(int(target_pn_order), "1pn")
        print(f"  Warm-start auto -> {choice.upper()}")

    def integrate(pn: int):
        rhs = newtonian_rhs if pn == 0 else make_pn_rhs(pn_order=pn, nu=nu, csq=csq, cqd=cqd)
        return solve_ivp(
            rhs,
            [0, 4 * t_max],
            y0,
            max_step=t_max / 2000,
            rtol=1e-8,
            atol=1e-10,
        )

    def is_bound(solution) -> bool:
        radius = np.hypot(solution.y[0], solution.y[1])
        phi = orbit_phase(solution)
        return (not (radius > r_apo_safe).any()) and phi[-1] >= phi_max_2pn * 0.8

    if choice == "2pn":
        selected = sol_2pn
        label = "2PN (Stage1 fit)"
        print("  Warm-start: 2PN (reuses sol_2pn - always bound)")
    elif choice == "1pn":
        trial = integrate(1)
        if is_bound(trial):
            selected, label = trial, "1PN (Stage1 fit)"
            print("  Warm-start: 1PN")
        else:
            print("  1PN escapes -> 0PN")
            selected, label = integrate(0), "0PN (Stage1 fit)"
    elif choice == "0pn":
        selected, label = integrate(0), "0PN (Stage1 fit)"
        print("  Warm-start: 0PN")
    else:
        raise ValueError(
            "warmstart_pn_order must be 'auto','0pn','1pn','2pn'; "
            f"got '{choice}'"
        )

    phi_selected = orbit_phase(selected)
    idx_stop = min(np.searchsorted(phi_selected, phi_max_2pn) + 1, selected.y.shape[1] - 1)
    trimmed = OrbitSlice(y=selected.y[:, :idx_stop], t=selected.t[:idx_stop])
    return trimmed, label, phi_max_2pn
