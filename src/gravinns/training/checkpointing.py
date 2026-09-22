"""Checkpoint I/O shared by the single-orbit trainers.

The public trainers all write the same three runtime artefacts: the latest
checkpoint, a compact JSON state file, and the training history.  Keeping the
I/O here avoids three near-identical private implementations and makes resume
semantics explicit and testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointNames:
    latest: str = "angle_latest.pth"
    state: str = "state.json"
    history: str = "history.json"


class CheckpointStore:
    """Read and write a trainer checkpoint directory."""

    def __init__(
        self,
        directory: str | Path,
        *,
        device: str,
        dissipative: bool,
        names: CheckpointNames | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.dissipative = bool(dissipative)
        self.names = names or CheckpointNames()

    @property
    def latest_path(self) -> Path:
        return self.directory / self.names.latest

    @property
    def state_path(self) -> Path:
        return self.directory / self.names.state

    @property
    def history_path(self) -> Path:
        return self.directory / self.names.history

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        best_l2_shape: float,
        best_l2_time: float,
        target_reached: bool,
        history: dict[str, Any],
    ) -> None:
        """Persist the latest train state.

        ``optimizer`` and ``scheduler`` are optional because some trainer paths
        save before Stage 2 has created them.
        """
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            },
            self.latest_path,
        )
        self.state_path.write_text(
            json.dumps(
                {
                    "epoch": int(epoch),
                    "best_l2_shape": float(best_l2_shape),
                    "best_l2_time": float(best_l2_time),
                    "target_reached": bool(target_reached),
                    "dissipative": self.dissipative,
                },
                indent=2,
            )
        )
        self.history_path.write_text(json.dumps(history, indent=2))

    def load(self, model: torch.nn.Module, optimizer: Any, scheduler: Any):
        """Restore the latest checkpoint, returning the legacy trainer tuple."""
        if not (self.state_path.exists() and self.latest_path.exists()):
            return 0, float("inf"), float("inf"), False, {}

        state = json.loads(self.state_path.read_text())
        checkpoint = torch.load(self.latest_path, map_location=self.device, weights_only=False)
        missing, _ = model.load_state_dict(checkpoint["model_state"], strict=False)
        if missing:
            print(f"  [resume] new layers init randomly: {missing}")

        try:
            opt_state = checkpoint.get("optimizer_state")
            sched_state = checkpoint.get("scheduler_state")
            if optimizer is not None and opt_state is not None:
                optimizer.load_state_dict(opt_state)
            if scheduler is not None and sched_state is not None:
                scheduler.load_state_dict(sched_state)
        except Exception as exc:  # keep historical best-effort resume semantics
            print(f"  [resume] opt/sched reset ({exc})")

        history: dict[str, Any] = {}
        if self.history_path.exists():
            history = json.loads(self.history_path.read_text())

        print(
            f"  Resumed from ep={state['epoch']}  "
            f"best_L2s={state['best_l2_shape']:.3e}"
        )
        return (
            state["epoch"],
            state["best_l2_shape"],
            state["best_l2_time"],
            state["target_reached"],
            history,
        )
