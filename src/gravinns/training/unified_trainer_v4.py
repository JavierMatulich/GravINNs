"""Backward-compatible import for the dissipative training protocol.

New code should import :func:`gravinns.training.train_dissipative`.
"""
from .dissipative_trainer import train_dissipative

train_regime_unified_v4 = train_dissipative

__all__ = ["train_regime_unified_v4", "train_dissipative"]
