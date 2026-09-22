"""Backward-compatible import for the near-circular training protocol.

New code should import :func:`gravinns.training.train_near_circular`.
"""
from .near_circular_trainer import train_near_circular

train_regime_unified = train_near_circular

__all__ = ["train_regime_unified", "train_near_circular"]
