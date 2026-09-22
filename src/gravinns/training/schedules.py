"""Shared curriculum, marching, and causal-weight schedules.

These helpers contain schedule mechanics common to the conservative,
long-orbit, and dissipative training protocols.  Protocol-specific choices
remain explicit in the trainer that calls them.
"""
from __future__ import annotations

import torch


def curriculum_weights(
    epoch: int,
    *,
    dissipative: bool,
    target_pn_order: int,
    curriculum_start: int,
    curriculum_epochs: int,
    anchor_floor: float = 0.0,
):
    """Return ``(alpha, beta)`` for anchor/PDE curriculum weights.

    ``anchor_floor`` is used only in the dissipative protocol.  Conservative
    1PN->2PN transfer decays the anchor to zero; 2PN->3PN uses no anchor.
    """
    curric_end = curriculum_start + curriculum_epochs

    if dissipative:
        if epoch < curriculum_start:
            return 1.0, float(epoch) / max(1, curriculum_start)
        if epoch < curric_end:
            frac = float(epoch - curriculum_start) / max(1, curriculum_epochs)
            alpha = (1.0 - frac) * (1.0 - anchor_floor) + anchor_floor
            return alpha, 1.0
        return anchor_floor, 1.0

    if target_pn_order <= 2:
        if curriculum_start == 0:
            return 0.0, 1.0
        if epoch < curriculum_start:
            return 1.0, float(epoch) / max(1, curriculum_start)
        if epoch < curric_end:
            alpha = 1.0 - float(epoch - curriculum_start) / max(1, curriculum_epochs)
            return alpha, 1.0
        return 0.0, 1.0

    # 2PN -> 3PN conservative transfer: no anchor, gradual PDE turn-on.
    beta = 1.0 if curriculum_start == 0 else min(
        1.0, float(epoch) / max(1, curriculum_start)
    )
    return 0.0, beta


def marching_window(
    epoch: int,
    *,
    n_epochs: int,
    n_march: int,
    march_fraction: float,
    phi_max: float,
):
    """Return ``(stage, phi_window)`` for expanding-window training."""
    if n_march <= 1:
        return 1, phi_max

    march_end = int(n_epochs * march_fraction)
    if epoch >= march_end:
        stage = n_march
    else:
        frac = epoch / max(1, march_end)
        stage = min(int(frac * n_march) + 1, n_march)
    return stage, phi_max * stage / n_march


def causal_weights(residual_per_point, *, eps: float, epoch: int | None = None,
                   end_epoch: int | None = None, anneal: bool = False):
    """Return detached causal weights for residuals sorted by angle."""
    cum = torch.cumsum(residual_per_point.detach(), dim=0)
    cum = torch.cat([torch.zeros(1, device=residual_per_point.device), cum[:-1]])
    cum_norm = cum / (cum[-1] + 1e-12)
    eps_now = eps
    if anneal and epoch is not None and end_epoch is not None:
        eps_now = eps * max(0.0, 1.0 - epoch / max(1, end_epoch))
    return torch.exp(-eps_now * cum_norm).detach()
