"""Backward-compatible import for the long-orbit training protocol.

New code should import :func:`gravinns.training.train_long_orbit`.
"""
from .long_orbit_trainer import train_long_orbit

train_regime_unified_v3 = train_long_orbit

__all__ = ["train_regime_unified_v3", "train_long_orbit"]
