"""Small runtime helpers shared by all single-orbit training protocols."""
from __future__ import annotations

import torch


def initialize_adam_cosine(model, *, lr, n_epochs, load_checkpoint,
                           auto_resume=True, announce=True):
    """Create Adam + cosine scheduler and optionally restore a checkpoint.

    Returns ``(optimizer, scheduler, start_epoch, best_shape, best_time,
    target_reached, history)``.  When resuming, the scheduler is rebuilt over
    the remaining budget exactly as in the original notebook protocols.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 1e-3
    )
    if auto_resume:
        start_ep, best_shape, best_time, target, history = load_checkpoint(
            model, optimizer, scheduler
        )
    else:
        start_ep, best_shape, best_time, target, history = (
            0, float("inf"), float("inf"), False, {}
        )

    remaining = max(n_epochs - start_ep, 1)
    if start_ep > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=lr * 1e-3
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        if announce:
            print(f"  Scheduler rebuilt: {remaining} remaining epochs, lr={lr:.1e}")

    return optimizer, scheduler, start_ep, best_shape, best_time, target, history
