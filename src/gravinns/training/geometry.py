"""Small geometry helpers shared by trainer diagnostics and recorders."""
from __future__ import annotations

import numpy as np


def trim_xy(solution, phi_max: float):
    """Return x/y coordinates up to ``phi_max`` for a solve_ivp-like orbit."""
    if solution is None:
        return np.zeros((2, 0))
    phi = np.unwrap(np.arctan2(solution.y[1], solution.y[0]))
    phi -= phi[0]
    keep = phi <= phi_max
    return np.stack([solution.y[0, keep], solution.y[1, keep]])
