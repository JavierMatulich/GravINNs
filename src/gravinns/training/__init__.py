"""Public training protocols.

The three trainers are separate scientific protocols, not software versions:

``train_near_circular``
    Conservative single-orbit protocol used in the phase-degeneracy study.
``train_long_orbit``
    Conservative long-orbit protocol with radial harmonics and marching.
``train_dissipative``
    Dissipative 2.5PN protocol with angular-momentum and secular-balance terms.

Legacy ``train_regime_unified*`` names remain as aliases for reproducibility of
older notebooks and scripts.
"""
from .near_circular_trainer import train_near_circular
from .long_orbit_trainer import train_long_orbit
from .dissipative_trainer import train_dissipative
from .adaptive import diagnose_pn_regime, train_orbit_adaptive

# Backward-compatible aliases.
train_regime_unified = train_near_circular
train_regime_unified_v3 = train_long_orbit
train_regime_unified_v4 = train_dissipative

__all__ = [
    "train_near_circular", "train_long_orbit", "train_dissipative",
    "train_regime_unified", "train_regime_unified_v3", "train_regime_unified_v4",
    "train_orbit_adaptive", "diagnose_pn_regime",
]
