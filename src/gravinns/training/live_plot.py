"""Live training figure shared by the single-orbit trainers.

Plotting is intentionally isolated from optimization: this module only reads
network outputs and histories and therefore cannot alter the training graph.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .geometry import trim_xy
from ..plotting.panels import save_panels_separately


@dataclass
class TrainingPlotter:
    fig: object
    axes: Sequence
    display: object
    scheduler: object
    sol_0pn: object | None
    sol_1pn: object | None
    sol_2pn: object
    sol_25pn: object
    sol_3pn: object | None
    phi_max: float
    phi_grid: np.ndarray
    warm_phi: np.ndarray
    warm_r: np.ndarray
    warmstart_label: str
    target_pn_order: int
    transfer_from_2pn: bool
    dissipative: bool
    q_min: float
    q_max: float
    stage_label: dict
    ep_arr: list
    loss_arr: list
    l2_ep: list
    l2s_arr: list
    l2t_arr: list
    compute_l2_time: bool
    l2_target: float
    label: str
    tag: str
    net_out: Callable[[], np.ndarray]
    ckpt_dir: str
    record_every_orbit: int = 0
    save_every_redraw: bool = False
    show_dissipative_target: bool = True
    energy_fn: Callable[[], np.ndarray] | None = None
    energy_ref: np.ndarray | None = None
    energy_target: float | None = None

    def redraw(self, ep, l_val, l2s, l2t, pred, stage=0, *, best_l2s, best_l2t):
        cur_lr = self.scheduler.get_last_lr()[0]
        ax = self.axes[0]
        ax.cla()

        if self.sol_0pn is not None:
            xy = trim_xy(self.sol_0pn, self.phi_max)
            ax.plot(xy[0], xy[1], color="grey", ls=":", lw=1.2, alpha=.7, label="0PN RK45")
        if self.sol_1pn is not None:
            xy = trim_xy(self.sol_1pn, self.phi_max)
            ax.plot(xy[0], xy[1], color="purple", ls="-.", lw=1.2, alpha=.6, label="1PN RK45")

        xy2 = trim_xy(self.sol_2pn, self.phi_max)
        ax.plot(xy2[0], xy2[1], color="steelblue", ls="--", lw=1.6, alpha=.85,
                label="2PN RK45 [pretrain src]")

        if self.dissipative and self.show_dissipative_target:
            ax.plot(self.sol_25pn.y[0], self.sol_25pn.y[1], color="seagreen", ls="-",
                    lw=1.9, alpha=.95, label="2.5PN RK45 [inspiral target]")
        elif self.target_pn_order >= 3 and self.sol_3pn is not None:
            ax.plot(self.sol_3pn.y[0], self.sol_3pn.y[1], color="seagreen", ls="-",
                    lw=1.8, alpha=.9, label=f"{self.target_pn_order}PN RK45 [target]")
        else:
            ax.plot(self.sol_2pn.y[0], self.sol_2pn.y[1], color="seagreen", ls="-",
                    lw=1.8, alpha=.9, label="2PN RK45 [target]")

        ws_mask = self.warm_phi <= self.phi_max + 1e-6
        ax.plot(
            self.warm_r[ws_mask] * np.cos(self.warm_phi[ws_mask]),
            self.warm_r[ws_mask] * np.sin(self.warm_phi[ws_mask]),
            color="darkorange", ls=":", lw=1.3, alpha=.6,
            label=f"warm-start: {self.warmstart_label}",
        )

        out = self.net_out()
        r_pred = 1.0 / out[:, 0]
        ax.plot(r_pred * np.cos(self.phi_grid), r_pred * np.sin(self.phi_grid),
                color="crimson", lw=2, alpha=.95, label="PINN")
        ax.scatter([0], [0], color="k", marker="*", s=120, zorder=5)
        ax.set_xlim(self.q_min, self.q_max)
        ax.set_ylim(self.q_min, self.q_max)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        stage_suffix = f" [{self.stage_label[stage]}]" if stage > 0 else ""
        ws_short = self.warmstart_label.split(" ")[0]
        scheme = f" | {ws_short}→{self.target_pn_order}PN transfer" if self.transfer_from_2pn else ""
        ax.set_title(
            f"Orbit — ep {ep:,}{stage_suffix}{scheme}"
            + ("  (inspiral)" if self.dissipative else "")
        )

        ax = self.axes[1]
        ax.cla()
        if len(self.ep_arr) > 1:
            ax.semilogy(self.ep_arr, self.loss_arr, color="steelblue", lw=.7, alpha=.5)
            w = min(50, len(self.loss_arr))
            ax.semilogy(
                self.ep_arr[w - 1:],
                np.convolve(self.loss_arr, np.ones(w) / w, "valid"),
                color="navy", lw=2, label=f"mean(w={w})",
            )
            ax.legend(fontsize=8)
        ax.set_title("Loss")
        ax.grid(which="both", alpha=.3)
        ax.set_xlabel("epoch")

        ax = self.axes[2]
        ax.cla()
        if len(self.l2_ep) > 1:
            ax.semilogy(self.l2_ep, self.l2s_arr, color="crimson", lw=2, label="L² shape")
            ax.fill_between(self.l2_ep, self.l2s_arr, alpha=.12, color="crimson")
            if self.compute_l2_time and any(v < float("inf") for v in self.l2t_arr):
                ax.semilogy(self.l2_ep, self.l2t_arr, color="royalblue", lw=1.5, ls="--",
                            label="L² time")
        ax.axhline(self.l2_target, color="crimson", ls=":", lw=1.2,
                   label=f"target {self.l2_target:.0e}")
        if self.compute_l2_time and best_l2t < float("inf") and best_l2t < 1.0:
            ax.axhline(best_l2t, color="seagreen", ls="--", lw=1.0, alpha=.7,
                       label=f"best L²t={best_l2t:.1e}")
        both = best_l2s < self.l2_target and best_l2t < self.l2_target
        ax.set_title(
            "L² convergence" + (" ★ BOTH" if both else " ★ SHAPE" if best_l2s < self.l2_target else ""),
            color="seagreen" if both else "black",
        )
        ax.legend(fontsize=7)
        ax.grid(which="both", alpha=.3)
        ax.set_xlabel("epoch")

        if self.dissipative and len(self.axes) > 3 and self.energy_fn is not None:
            ax = self.axes[3]
            ax.cla()
            try:
                H_net = self.energy_fn()
                ax.plot(self.phi_grid, H_net, color="crimson", lw=1.8, label="H(φ) — PINN")
                if self.energy_ref is not None:
                    ax.plot(self.phi_grid, self.energy_ref, color="seagreen", lw=1.5, ls="--",
                            alpha=.9, label="H(φ) — 2.5PN ref")
                if self.energy_target is not None:
                    ax.axhline(self.energy_target, color="steelblue", ls=":", lw=1.2,
                               label=f"E* (conservative) = {self.energy_target:.4f}")
                dH = np.diff(H_net)
                frac_up = float((dH > 0).mean())
                dE_tot = float(H_net[-1] - H_net[0])
                ok = frac_up < 0.05 and dE_tot < 0
                ax.set_title(
                    "Energy H(φ)  ✓ monotonic decay" if ok else "Energy H(φ)  ⚠ not monotonic",
                    color="seagreen" if ok else "darkorange",
                )
                ax.set_xlabel("φ [rad]")
                ax.set_ylabel("H")
                ax.grid(alpha=.3)
                ax.legend(fontsize=7, loc="best")
                ax.text(
                    0.02, 0.02,
                    f"ΔH = {dE_tot:+.3e}\nrising steps = {100 * frac_up:.1f}%",
                    transform=ax.transAxes, fontsize=7, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=.7),
                )
            except Exception as exc:
                ax.text(0.5, 0.5, f"energy panel error:\n{exc}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)

        l2t_str = f" L²t {l2t:.2e}" if (self.compute_l2_time and l2t < float("inf")) else ""
        self.fig.suptitle(
            f"{self.label} [{self.tag}] | ep {ep:,} | L²s {l2s:.2e}{l2t_str} lr {cur_lr:.1e}",
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self.display.update(self.fig)

        names = ["orbit", "loss", "l2"] + (["energy"] if self.dissipative else [])
        save_now = self.save_every_redraw or bool(
            self.record_every_orbit and ep % self.record_every_orbit == 0
        )
        if save_now:
            save_panels_separately(
                self.fig,
                list(self.axes),
                names,
                out_dir=os.path.join(self.ckpt_dir, "figures"),
                prefix=f"ep{ep:06d}",
                dpi=300,
            )
