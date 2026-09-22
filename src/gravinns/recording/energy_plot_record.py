"""Plot record of the dissipative trainer (v4), with energy snapshots.

Origin: ``_PlotRecorder`` in GitHub_Dissipative.ipynb, cell 2.  Same idea as
``gravinns.recording.PlotRecorder`` (cases 1-2) but a different file layout:
it also stores periodic ENERGY snapshots (network u/pr/L, H(phi), the total
energy change and the fraction of rising steps), which is what
``gravinns.plotting.replot`` redraws.
"""
import os

import numpy as np

PLOTREC_FILENAME = "plot_record.npz"


class EnergyPlotRecorder:
    """Stores everything needed to redraw the live training figure without
    retraining, in ONE .npz file (no pickle; load with allow_pickle=False).

    Key groups inside the file
      static_*        written once: reference orbits, phi_grid, H_ref(phi), E*,
                      l2_target, axis limits, labels ...
      full_loss_*     exact ep_arr/loss_arr -- the SAME per-epoch arrays the
                      live figure plots, flushed to disk every
                      `record_every_loss` epochs (never thinned/sampled)
      full_l2_*       exact l2_ep/l2s_arr/l2t_arr, flushed every
                      `record_every_l2` epochs (same values as live plot)
      energy_*        one snapshot every `record_every_energy` epochs, incl.
                      the network output u, pr, L on phi_grid (so the ORBIT panel
                      can be redrawn at that stage too) and H(phi) in float64
      final_best_*    network after loading the best checkpoint
      final_polish_*  network after the L-BFGS polish
    """
    _GROUPS = ("energy",)   # sparse, resumable, row-per-snapshot groups.
                            # loss/l2 are handled separately as 'full' arrays
                            # (see .full): they are exact copies of the same
                            # ep_arr/loss_arr/l2_ep/l2s_arr/l2t_arr the LIVE
                            # figure plots, so the re-plot is pixel-identical
                            # to what training showed -- no independent
                            # sampling cadence to go out of sync with.

    def __init__(self, path, start_ep=0):
        self.path   = path
        self.static = {}
        self.full   = {}      # exact loss/L2 history, overwritten each flush
        self.final  = {}
        self.rows   = {g: {} for g in self._GROUPS}
        if start_ep > 0 and os.path.exists(path):
            self._resume(start_ep)

    # -- resume: keep rows strictly BEFORE the checkpoint epoch (the loop
    #    re-runs epoch start_ep, so that row is recorded again) --------------
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
        with open(tmp, "wb") as f:                 # atomic: never a half-written file
            np.savez_compressed(f, **out)
        os.replace(tmp, self.path)


_PlotRecorder = EnergyPlotRecorder   # notebook name
