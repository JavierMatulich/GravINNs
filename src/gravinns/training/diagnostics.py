"""Reusable diagnostics for single-orbit PINN training."""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid

from ..physics.hamiltonian import compute_hamiltonian_r


class OrbitDiagnostics:
    """Evaluate a trained angle-domain orbit without affecting optimization."""

    def __init__(
        self,
        *,
        model,
        device: str,
        phi_max: float,
        phi_grid: np.ndarray,
        qx_ref: np.ndarray,
        qy_ref: np.ndarray,
        px_ref: np.ndarray,
        py_ref: np.ndarray,
        ref_norm: float,
        ref_time_state: np.ndarray,
        t_eval: np.ndarray,
        target_pn_order: int,
        nu: float,
        csq: float,
        cqd: float,
        compute_l2_time: bool,
        dissipative: bool,
        t_ref_ang: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.phi_max = float(phi_max)
        self.phi_grid = np.asarray(phi_grid)
        self.qx_ref = np.asarray(qx_ref)
        self.qy_ref = np.asarray(qy_ref)
        self.px_ref = np.asarray(px_ref)
        self.py_ref = np.asarray(py_ref)
        self.ref_norm = float(ref_norm)
        self.ref_time_state = np.asarray(ref_time_state)
        self.t_eval = np.asarray(t_eval)
        self.target_pn_order = int(target_pn_order)
        self.nu = float(nu)
        self.csq = float(csq)
        self.cqd = float(cqd)
        self.compute_l2_time = bool(compute_l2_time)
        self.dissipative = bool(dissipative)
        self.t_ref_ang = None if t_ref_ang is None else np.asarray(t_ref_ang)

    def net_out(self) -> np.ndarray:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            phi = torch.linspace(
                0, self.phi_max, len(self.phi_grid), device=self.device
            ).view(-1, 1)
            out = self.model(phi).cpu().numpy()
        if was_training:
            self.model.train()
        return out

    def phase_space_from_output(self, out: np.ndarray) -> np.ndarray:
        u = out[:, 0]
        pr = out[:, 1]
        L = out[:, 2]
        r = 1.0 / u
        c = np.cos(self.phi_grid)
        s = np.sin(self.phi_grid)
        qx = r * c
        qy = r * s
        px = pr * c - (L * u) * s
        py = pr * s + (L * u) * c
        return np.stack([qx, qy, px, py], axis=1)

    def shape_error(self) -> float:
        pred = self.phase_space_from_output(self.net_out())
        diff = pred.T - np.stack(
            [self.qx_ref, self.qy_ref, self.px_ref, self.py_ref]
        )
        return float(np.linalg.norm(diff) / self.ref_norm)

    def shape_prediction(self, *, include_upl: bool = False):
        out = self.net_out()
        pred = self.phase_space_from_output(out)
        if include_upl:
            return pred, out[:, :3].copy()
        return pred

    def dphi_dt(self, u: np.ndarray, pr: np.ndarray, L: np.ndarray) -> np.ndarray:
        r = 1.0 / u
        c = np.cos(self.phi_grid)
        s = np.sin(self.phi_grid)
        qx, qy = r * c, r * s
        px = pr * c - (L * u) * s
        py = pr * s + (L * u) * c
        result = np.zeros_like(u)
        for i in range(len(u)):
            q = torch.tensor([[qx[i], qy[i]]], dtype=torch.float64, requires_grad=True)
            p = torch.tensor([[px[i], py[i]]], dtype=torch.float64, requires_grad=True)
            H = compute_hamiltonian_r(
                q, p, self.target_pn_order, self.nu, self.csq, self.cqd
            )
            dH_dp = torch.autograd.grad(H.sum(), p)[0][0].detach().numpy()
            result[i] = (qx[i] * dH_dp[1] - qy[i] * dH_dp[0]) / (r[i] ** 2)
        return result

    def time_error_and_prediction(self):
        if not self.compute_l2_time:
            return float("inf"), np.zeros_like(self.ref_time_state)

        out = self.net_out()
        pred_phi = self.phase_space_from_output(out)
        u, pr, L = out[:, 0], out[:, 1], out[:, 2]

        if self.dissipative:
            if self.t_ref_ang is None:
                raise ValueError("t_ref_ang is required for dissipative time diagnostics")
            t_phi = self.t_ref_ang
        else:
            omega = self.dphi_dt(u, pr, L)
            dt_dphi = 1.0 / (omega + 1e-12)
            t_phi = np.concatenate([[0], cumulative_trapezoid(dt_dphi, self.phi_grid)])

        pred = np.empty_like(self.ref_time_state)
        cap = min(t_phi[-1], self.t_eval[-1])
        mask = self.t_eval <= cap
        for k in range(4):
            pred[mask, k] = np.interp(self.t_eval[mask], t_phi, pred_phi[:, k])
        pred[~mask] = pred[mask][-1]
        err = np.linalg.norm(pred - self.ref_time_state) / (
            np.linalg.norm(self.ref_time_state) + 1e-16
        )
        return float(err), pred

    def energy(self, out: np.ndarray | None = None) -> np.ndarray:
        out = self.net_out() if out is None else out
        pred = self.phase_space_from_output(out)
        q = torch.tensor(pred[:, :2], dtype=torch.float64)
        p = torch.tensor(pred[:, 2:], dtype=torch.float64)
        with torch.no_grad():
            H = compute_hamiltonian_r(
                q, p, self.target_pn_order, self.nu, self.csq, self.cqd
            )
        return H.cpu().numpy().reshape(-1)
