"""Optional per-epoch logging of the individual loss channels.

The notebook's trainer and sweep call a global ``LOSS_REC`` object ("cell A")
that is *not defined* in Single_orbit_plot_final_clean.ipynb, so there the
logging is silently disabled (``NameError`` / ``"LOSS_REC" in globals()``).
This module reproduces that contract:

* by default no recorder is installed  ->  behaviour identical to the notebook;
* ``install_loss_recorder()`` installs one  ->  ``run_ic_sweep`` activates it
  for every run and writes ``<run_dir>/loss_components.npz`` with the keys
  ``epoch, loss, l_pde, l_L, l_energy, l_anchor, alpha, beta, lr``.  That file
  is what ``summarize_run`` reads to fill ``l_pde_final`` / ``l_energy_final``.

Recording converts every loss term to a Python float each epoch, which forces
a GPU synchronisation, so it costs some speed.
"""
import os
from typing import Optional

import numpy as np

LOSS_COMPONENTS_FILENAME = "loss_components.npz"


class LossComponentRecorder:
    def __init__(self):
        self.active = False
        self._buf = {}

    def reset(self):
        self._buf = {}

    def push(self, **vals):
        if not self.active:
            return
        for k, v in vals.items():
            self._buf.setdefault(k, []).append(v)

    def __len__(self):
        return len(self._buf.get("epoch", []))

    def save(self, ckpt_dir) -> Optional[str]:
        if not self._buf:
            return None
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, LOSS_COMPONENTS_FILENAME)
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in self._buf.items()})
        return path


# The installed recorder, or None (== the notebook, where LOSS_REC is undefined).
LOSS_REC: Optional[LossComponentRecorder] = None


def install_loss_recorder() -> LossComponentRecorder:
    global LOSS_REC
    if LOSS_REC is None:
        LOSS_REC = LossComponentRecorder()
    return LOSS_REC


def uninstall_loss_recorder() -> None:
    global LOSS_REC
    LOSS_REC = None


def get_loss_recorder() -> Optional[LossComponentRecorder]:
    return LOSS_REC
