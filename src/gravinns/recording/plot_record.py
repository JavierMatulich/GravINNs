"""On-disk record of everything the live training figure draws.

Origin: ``_PlotRecorder`` in notebook cell 1.  One ``.npz`` (no pickle) per
run, so the 3-panel figure can be redrawn later with numpy only.
"""
import os

import numpy as np

PLOTREC_FILENAME = "plot_record.npz"   # <ckpt_dir>/plot_record.npz


class PlotRecorder:
    """Stores everything needed to redraw the 3-panel training figure
    (orbit, loss, L2 convergence) in ONE .npz file (no pickle).

    Key groups
      static_*   written once: RK reference orbits (0PN/1PN/2PN/target,
                 trimmed to phi_max), the warm-start orbit, phi_grid, axis
                 limits, labels, and the run's regime/config values.
      full_*     loss_epoch/loss_value and l2_epoch/l2_shape/l2_time -- EXACT
                 copies of ep_arr/loss_arr/l2_ep/l2s_arr/l2t_arr, the very
                 same arrays the live figure plots. Kept in sync as training
                 runs (and correctly resumed, since those source arrays are
                 already reloaded from history.json on resume), so the
                 re-plotted curves are pixel-identical to what training showed.
      orbit_snapshot_*  one row every `record_every_orbit` epochs: epoch,
                 the network's (u,pr,L) on phi_grid, and the L2 shape error --
                 enough to redraw the ORBIT panel at any saved stage, not just
                 the end. Resumable: rows at or after the resume epoch are
                 dropped and re-recorded, so nothing is duplicated.
    """
    _GROUPS = ("orbit_snapshot",)

    def __init__(self, path, start_ep=0):
        self.path   = path
        self.static = {}
        self.full   = {}
        self.final  = {}
        self.rows   = {g: {} for g in self._GROUPS}
        if start_ep > 0 and os.path.exists(path):
            self._resume(start_ep)

    def _resume(self, start_ep):
        try:
            d = np.load(self.path, allow_pickle=False)
        except Exception as e:
            print(f"  [plot-record] could not reload {self.path} ({e}); starting fresh")
            return
        for g in self._GROUPS:
            ek = f"{g}_epoch"
            if ek not in d.files:
                continue
            keep = np.asarray(d[ek]) < start_ep
            for k in d.files:
                if k.startswith(g + "_"):
                    self.rows[g][k[len(g)+1:]] = list(np.asarray(d[k])[keep])
        n = {g: len(self.rows[g].get("epoch", [])) for g in self._GROUPS}
        print(f"  [plot-record] resumed {n} rows (epoch < {start_ep}) from {self.path}")

    def add(self, group, **vals):
        r = self.rows[group]
        for k, v in vals.items():
            r.setdefault(k, []).append(v)

    def last_epoch(self, group):
        e = self.rows[group].get("epoch", [])
        return e[-1] if e else None

    def save(self):
        out = {}
        for k, v in self.static.items():
            out["static_" + k] = np.asarray(v)
        for k, v in self.full.items():
            out["full_" + k] = np.asarray(v)
        for k, v in self.final.items():
            out["final_" + k] = np.asarray(v)
        for g in self._GROUPS:
            for k, v in self.rows[g].items():
                out[f"{g}_{k}"] = np.asarray(v)
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **out)
        os.replace(tmp, self.path)


_PlotRecorder = PlotRecorder   # notebook name
