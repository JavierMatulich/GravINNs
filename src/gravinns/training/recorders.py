"""Plot-record setup and history flushing for single-orbit trainers."""
from __future__ import annotations

import numpy as np

from .geometry import trim_xy


class StandardRecordController:
    def __init__(
        self,
        recorder,
        *,
        net_out,
        ep_arr,
        loss_arr,
        l2_ep,
        l2s_arr,
        l2t_arr,
        record_every_loss: int,
        record_every_l2: int,
        record_every_orbit: int,
    ):
        self.recorder = recorder
        self.net_out = net_out
        self.ep_arr = ep_arr
        self.loss_arr = loss_arr
        self.l2_ep = l2_ep
        self.l2s_arr = l2s_arr
        self.l2t_arr = l2t_arr
        self.record_every_loss = int(record_every_loss or 0)
        self.record_every_l2 = int(record_every_l2 or 0)
        self.record_every_orbit = int(record_every_orbit or 0)

    def _snapshot(self, epoch: int, l2_shape=float("nan")):
        out = self.net_out()
        self.recorder.add(
            "orbit_snapshot",
            epoch=int(epoch),
            l2_shape=float(l2_shape),
            u=out[:, 0].astype(np.float32),
            pr=out[:, 1].astype(np.float32),
            L=out[:, 2].astype(np.float32),
        )
        self.recorder.save()

    def record_step(self, epoch: int, force: bool = False):
        did_save = False
        if force or (self.record_every_loss and epoch % self.record_every_loss == 0):
            self.recorder.full["loss_epoch"] = np.asarray(self.ep_arr, dtype=np.int64)
            self.recorder.full["loss_value"] = np.asarray(self.loss_arr, dtype=np.float32)
            did_save = True
        if force or (self.record_every_l2 and epoch % self.record_every_l2 == 0):
            self.recorder.full["l2_epoch"] = np.asarray(self.l2_ep, dtype=np.int64)
            self.recorder.full["l2_shape"] = np.asarray(self.l2s_arr, dtype=np.float64)
            self.recorder.full["l2_time"] = np.asarray(self.l2t_arr, dtype=np.float64)
            did_save = True
        if force or (
            self.record_every_orbit
            and epoch % self.record_every_orbit == 0
            and self.recorder.last_epoch("orbit_snapshot") != epoch
        ):
            self._snapshot(epoch, self.l2s_arr[-1] if self.l2s_arr else float("nan"))
            did_save = False
        if did_save:
            self.recorder.save()


class EnergyRecordController:
    def __init__(
        self,
        recorder,
        *,
        net_out,
        net_energy,
        scheduler,
        ep_arr,
        loss_arr,
        l2_ep,
        l2s_arr,
        l2t_arr,
        record_every_loss: int,
        record_every_l2: int,
        record_every_energy: int,
    ):
        self.recorder = recorder
        self.net_out = net_out
        self.net_energy = net_energy
        self.scheduler = scheduler
        self.ep_arr = ep_arr
        self.loss_arr = loss_arr
        self.l2_ep = l2_ep
        self.l2s_arr = l2s_arr
        self.l2t_arr = l2t_arr
        self.record_every_loss = int(record_every_loss or 0)
        self.record_every_l2 = int(record_every_l2 or 0)
        self.record_every_energy = int(record_every_energy or 0)

    def _energy_snapshot(self, epoch: int, stage: int, l2_shape=float("nan")):
        out = self.net_out()
        H = np.asarray(self.net_energy(out), dtype=np.float64)
        dH = np.diff(H)
        self.recorder.add(
            "energy",
            epoch=int(epoch),
            stage=int(stage),
            lr=float(self.scheduler.get_last_lr()[0]),
            l2_shape=float(l2_shape),
            u=out[:, 0].astype(np.float32),
            pr=out[:, 1].astype(np.float32),
            L=out[:, 2].astype(np.float32),
            H=H,
            dE_tot=float(H[-1] - H[0]),
            frac_up=float((dH > 0).mean()),
        )
        self.recorder.save()

    def record_step(self, epoch: int, stage: int, force: bool = False):
        did_save = False
        if force or (self.record_every_loss and epoch % self.record_every_loss == 0):
            self.recorder.full["loss_epoch"] = np.asarray(self.ep_arr, dtype=np.int64)
            self.recorder.full["loss_value"] = np.asarray(self.loss_arr, dtype=np.float32)
            did_save = True
        if force or (self.record_every_l2 and epoch % self.record_every_l2 == 0):
            self.recorder.full["l2_epoch"] = np.asarray(self.l2_ep, dtype=np.int64)
            self.recorder.full["l2_shape"] = np.asarray(self.l2s_arr, dtype=np.float64)
            self.recorder.full["l2_time"] = np.asarray(self.l2t_arr, dtype=np.float64)
            did_save = True
        if force or (self.record_every_energy and epoch % self.record_every_energy == 0):
            self._energy_snapshot(
                epoch,
                stage,
                self.l2s_arr[-1] if self.l2s_arr else float("nan"),
            )
            did_save = False
        if did_save:
            self.recorder.save()


def configure_standard_static(
    recorder,
    *,
    label,
    tag,
    warmstart_label,
    target_pn_order,
    transfer_from_2pn,
    compute_l2_time,
    dissipative,
    nu,
    p0_factor,
    r0_km,
    pr0,
    ecc,
    L0,
    n_orbits,
    l2_target,
    phi_max,
    phi_grid,
    q_min,
    q_max,
    sol_0pn,
    sol_1pn,
    sol_2pn,
    sol_3pn,
    warm_phi,
    warm_r,
    record_every_loss,
    record_every_l2,
    record_every_orbit,
):
    if target_pn_order >= 3 and sol_3pn is not None:
        target_xy = np.stack([sol_3pn.y[0], sol_3pn.y[1]])
        target_label = f"{target_pn_order}PN RK45 [target]"
    else:
        target_xy = np.stack([sol_2pn.y[0], sol_2pn.y[1]])
        target_label = "2PN RK45 [target]"
    warm_mask = warm_phi <= phi_max + 1e-6
    recorder.static.update(
        label=label,
        tag=tag,
        warmstart_label=warmstart_label,
        target_label=target_label,
        dissipative=bool(dissipative),
        target_pn_order=int(target_pn_order),
        transfer_from_2pn=bool(transfer_from_2pn),
        compute_l2_time=bool(compute_l2_time),
        nu=float(nu),
        p0_factor=float(p0_factor),
        r0_km=float(r0_km),
        pr0=float(pr0),
        ecc=float(ecc),
        L0=float(L0),
        n_orbits=int(n_orbits),
        l2_target=float(l2_target),
        phi_max=float(phi_max),
        phi_grid=np.asarray(phi_grid, dtype=np.float64),
        q_lim=np.array([q_min, q_max], dtype=np.float64),
        orbit_0pn=trim_xy(sol_0pn, phi_max),
        orbit_1pn=trim_xy(sol_1pn, phi_max),
        orbit_2pn=trim_xy(sol_2pn, phi_max),
        orbit_target=target_xy,
        orbit_warmstart=np.stack(
            [
                warm_r[warm_mask] * np.cos(warm_phi[warm_mask]),
                warm_r[warm_mask] * np.sin(warm_phi[warm_mask]),
            ]
        ),
        record_every=np.array(
            [int(record_every_loss or 0), int(record_every_l2 or 0), int(record_every_orbit or 0)]
        ),
    )
    recorder.save()


def configure_energy_static(
    recorder,
    *,
    label,
    tag,
    warmstart_label,
    target_pn_order,
    transfer_from_2pn,
    compute_l2_time,
    dissipative,
    nu,
    p0_factor,
    r0_km,
    pr0,
    ecc,
    L0,
    n_orbits,
    l2_target,
    E_target,
    phi_max,
    phi_grid,
    q_min,
    q_max,
    H_ref,
    sol_0pn,
    sol_1pn,
    sol_2pn,
    sol_25pn,
    sol_3pn,
    warm_phi,
    warm_r,
    record_every_loss,
    record_every_l2,
    record_every_energy,
):
    if dissipative:
        target_sol, target_label = sol_25pn, "2.5PN RK45 [inspiral target]"
    elif target_pn_order >= 3 and sol_3pn is not None:
        target_sol, target_label = sol_3pn, f"{target_pn_order}PN RK45 [target]"
    else:
        target_sol, target_label = sol_2pn, "2PN RK45 [target]"
    warm_mask = warm_phi <= phi_max + 1e-6
    recorder.static.update(
        label=label,
        tag=tag,
        warmstart_label=warmstart_label,
        target_label=target_label,
        dissipative=bool(dissipative),
        target_pn_order=int(target_pn_order),
        transfer_from_2pn=bool(transfer_from_2pn),
        compute_l2_time=bool(compute_l2_time),
        nu=float(nu),
        p0_factor=float(p0_factor),
        r0_km=float(r0_km),
        pr0=float(pr0),
        ecc=float(ecc),
        L0=float(L0),
        n_orbits=int(n_orbits),
        l2_target=float(l2_target),
        E_target=float(E_target),
        phi_max=float(phi_max),
        phi_grid=np.asarray(phi_grid, dtype=np.float64),
        q_lim=np.array([q_min, q_max], dtype=np.float64),
        H_ref=np.asarray(H_ref, dtype=np.float64) if H_ref is not None else np.zeros(0),
        orbit_0pn=trim_xy(sol_0pn, phi_max),
        orbit_1pn=trim_xy(sol_1pn, phi_max),
        orbit_2pn=trim_xy(sol_2pn, phi_max),
        orbit_target=np.stack([target_sol.y[0], target_sol.y[1]]),
        orbit_warmstart=np.stack(
            [
                warm_r[warm_mask] * np.cos(warm_phi[warm_mask]),
                warm_r[warm_mask] * np.sin(warm_phi[warm_mask]),
            ]
        ),
        record_every=np.array(
            [int(record_every_loss or 0), int(record_every_l2 or 0), int(record_every_energy or 0)]
        ),
    )
    recorder.save()
