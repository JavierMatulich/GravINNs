"""Long-orbit single-orbit trainer.

This long-orbit protocol adds radial harmonics, expanding-window marching, causal
annealing, and optional L-BFGS refinement. Shared machinery is factored into
``gravinns.training`` helper modules.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .. import constants as _const
from ..models.angle_pinn import AngleOrbitPINN4
from ..physics.hamiltonian import compute_hamiltonian_r
from ..plotting.panels import live_display
from ..recording.plot_record import PLOTREC_FILENAME as _PLOTREC, PlotRecorder as _PlotRecorder
from .checkpointing import CheckpointStore
from .reference import build_reference_orbits, select_warmstart_orbit
from .diagnostics import OrbitDiagnostics
from .residuals import angle_residual
from .schedules import curriculum_weights, marching_window, causal_weights
from .runtime import initialize_adam_cosine
from .live_plot import TrainingPlotter
from .recorders import StandardRecordController, configure_standard_static
from .artifacts import (ANALYSIS_JSON as _ANJSON, ANALYSIS_NPZ as _ANPZ,
                        BEST as _BEST, BEST_BOTH as _BEST_BOTH, CONFIG as _CONFIG)

def train_long_orbit(nu, p0_factor, r0_km,
                         pr0_factor         = 0.0,    # initial radial momentum pr(0)
                         dissipative        = False,
                         # ── Transfer learning (Stage 1) ──────────────────
                         transfer_from_2pn  = True,    # Stage 1: MSE warm-start
                         warmstart_pn_order = 'auto',  # 'auto'/(0pn/1pn/2pn)
                         target_pn_order    = 2,       # PDE order: 2 or 3
                         pretrain_epochs    = 20_000,  # Stage 1 epochs (0=skip)
                         pretrain_lr        = 1e-3,    # Stage 1 learning rate
                         # ── Curriculum (Stage 2): 2PN→2.5PN blending ─────
                         curriculum_start   = 5_000,   # epochs at full 2PN anchor weight
                         curriculum_epochs  = 30_000,  # linear decay to zero
                         # ── Causal PINN (overlapping with curriculum) ─────
                         causal_eps         = 1.0,     # causality strength
                         causal_fraction    = 0.5,     # fraction of n_epochs
                         use_causal_training= None,    # None=auto True/False=force
                         # ── Expanding-window / adaptive-time-marching ────
                         n_march            = 1,       # # expanding windows (1=off)
                         march_fraction     = 0.6,     # frac of epochs spent marching
                         compute_l2_time    = True,    # False = skip L2-time entirely
                         pn_valid_thresh    = 0.10,    # stop orbit when 1/(r c^2)>this
                         # ── General ──────────────────────────────────────
                         base_dir           = None,
                         n_orbits           = 3,
                         n_epochs           = 200_000,
                         n_colloc           = 4000,
                         lbfgs_rounds       = 5,      # [PATCH] frozen-batch L-BFGS rounds (0=skip)
                         lbfgs_cycles       = 10,     # [PATCH] L-BFGS cycles per round
                         lr                 = 3e-3,
                         n_fourier          = 8,
                         fourier_sigma      = 2.0,
                         n_fourier_hi       = 16,   # high-freq Fourier bank for periapsis (0=off)
                         fourier_sigma_hi   = 8.0,  # bandwidth; periapsis residual is 100× apoapsis
                         n_harmonic         = 0,    # radial-harmonic bank (0=off); fixes phase drift
                         hidden             = 64,
                         depth              = 3,
                         checkpoint_every   = 5_000,
                         plot_every         = 2_000,
                         log_every          = 2_000,
                         # ── plot record (re-plot without retraining) ─────
                         record_every_loss  = 1_000,   # flush cadence, 0=off;
                         record_every_l2    = 1_000,   # flush cadence, 0=off
                         record_every_orbit = 20_000,  # orbit-shape snapshot cadence
                         record_file        = None,    # None -> <ckpt_dir>/plot_record.npz
                         w_L                = 1.0,     # L residual weight
                         w_energy           = 5.0,     # target-PN energy conservation weight
                         w_anchor           = 10.0,    # warm-start orbit anchor weight
                         l2_target          = 1e-3,
                         patience           = 5,       # kept for compat (ignored if window active)
                         convergence_window = 20,      # W: L2 measurements in sliding window
                         convergence_cv     = 0.5,     # max allowed std/mean (coeff of variation)
                         dev                = None,
                         # ── repo addition (does not change training) ────
                         live_plot          = None):   # None=auto (Jupyter only)

    dev=dev or ("cuda" if torch.cuda.is_available() else "cpu")
    CSQ=_const.C_SQ; CQD=_const.C_QD

    _do_causal = dissipative if use_causal_training is None else bool(use_causal_training)
    _warmstart_label = 'N/A'   # overwritten by warmstart selection block below
    C_SCALAR=float(np.sqrt(CSQ))
    if base_dir is None:
        base_dir="regime_checkpoints_angle25" if dissipative else "regime_checkpoints_angle"

    # ── regime scalars ─────────────────────────────────────────────────── #
    q0  = r0_km / _const.R0_REF_KM
    p0t = p0_factor / np.sqrt(q0)   # tangential momentum
    pr0 = float(pr0_factor)          # radial momentum  (0 = start at apsis)
    u0  = 1.0 / q0
    y0  = [q0, 0.0, pr0, p0t]
    L0  = q0 * p0t                   # angular momentum (conserved for pr0≠0 too)
    E   = 0.5*(pr0**2 + p0t**2) - 1.0/q0
    a   = -0.5/E if E < 0 else float("inf")
    T   = 2*np.pi*a**1.5 if np.isfinite(a) else 2*np.pi*q0**1.5
    ecc = float(np.sqrt(max(0.0, 1 + 2*E*L0**2)))
    if np.isfinite(a) and 0.0 < ecc < 1.0:
        u_lo = 0.70 / (a*(1+ecc))
        u_hi = (3.0 if dissipative else 1.60) / (a*(1-ecc))
    else:
        u_lo = 0.3*u0
        u_hi = (8.0 if dissipative else 6.0)*u0
    u_lo = min(u_lo, 0.95*u0)
    u_hi = max(u_hi, 1.05*u0)

    tag=f"2.5PN→{target_pn_order}PN" if dissipative else f"{target_pn_order}PN"
    _pr0_str = f"_pr{pr0:+.2f}" if abs(pr0) > 1e-6 else ""
    label    = f"nu{nu:.2f}_p{p0_factor:.2f}_r{r0_km:.0f}km{_pr0_str}"
    ckpt_dir=os.path.join(base_dir,label); os.makedirs(ckpt_dir,exist_ok=True)

    print("="*70)
    print(f"  UNIFIED ANGLE TRAINING [{label}]   mode={target_pn_order}PN   tag={tag}"
          +("  (radiation reaction ON)" if dissipative else ""))
    if dissipative:
        s1=f"Stage1=1PN→2PN-transfer({'ON' if transfer_from_2pn else 'OFF'})"
        _causal_str=f"causal(eps={causal_eps},frac={causal_fraction})" if _do_causal else "causal=OFF"
        s2=f"Stage2=curriculum({curriculum_epochs}ep,start={curriculum_start})+{_causal_str}"
        s3="Stage3=standard"
        print(f"  {s1}  {s2}  {s3}")
    print(f"  q0={q0:.4f}  p0t={p0t:.4f}  pr0={pr0:+.4f}  "
          f"L0={L0:.4f}  T={T:.3f}  ecc={ecc:.3f}")
    print(f"  u-band [{u_lo:.3f},{u_hi:.3f}]   Saving to: {ckpt_dir}")
    print("="*70)

    # ── checkpoint I/O ─────────────────────────────────────────────── #
    _checkpoint_store = CheckpointStore(ckpt_dir, device=dev, dissipative=dissipative)
    _save_ckpt = _checkpoint_store.save
    _load_ckpt = _checkpoint_store.load

    with open(os.path.join(ckpt_dir,_CONFIG),'w') as f:
        json.dump(dict(regime=dict(nu=nu,p0_factor=p0_factor,r0_km=r0_km,
                       pr0=pr0,q0=q0,ecc=ecc,T_orb=T,L0=L0,dissipative=dissipative),
                       training=dict(n_epochs=n_epochs,lr=lr,n_fourier=n_fourier,
                       hidden=hidden,depth=depth,w_L=w_L,
                       causal_eps=causal_eps,causal_fraction=causal_fraction,
                       use_causal_training=_do_causal,
                       transfer_from_2pn=transfer_from_2pn,
                       pretrain_epochs=pretrain_epochs)),f,indent=2)

    # ── RK45 references and Stage-1 warm start ─────────────────────── #
    # [fix] the notebook's live value; the max(4000, 2000*n_orbits) variant is
    # the line the author left COMMENTED OUT there. Using it changes the grid
    # (240 -> 6000 points at n_orbits=3, 4000 -> 100000 at n_orbits=50) and so
    # changes the trained model: the published long-orbit runs used this one.
    _n_ref = int(80 * n_orbits)
    #_n_ref = max(4000, int(2000*n_orbits))
    # [fix] the notebook uses TWO resolutions here: _n_ref for the phi grid and
    # a denser one for the RK45 references (40000 at N=100). Collapsing them
    # into a single n_ref changes both grids and hence the trained model.
    _n_ref_t = max(4000, int(400 * n_orbits))
    t_max = n_orbits*T
    _refs = build_reference_orbits(
        y0=y0, nu=nu, csq=CSQ, cqd=CQD, c_scalar=C_SCALAR,
        t_max=t_max, n_ref=_n_ref_t, dissipative=dissipative,
        target_pn_order=target_pn_order, pn_valid_thresh=pn_valid_thresh,
        a=a, ecc=ecc, L0=L0, l2_target=l2_target,
        two_pn_rtol=1e-13, two_pn_atol=1e-15,
    )
    t_eval = _refs.t_eval
    sol_0pn_ref = _refs.sol_0pn
    sol_1pn_ref = _refs.sol_1pn
    sol_2pn = _refs.sol_2pn
    sol_25pn = _refs.sol_25pn
    sol_3pn = _refs.sol_3pn
    _l2_23 = _refs.l2_23
    r_pn_min = _refs.r_pn_min
    if _refs.early_exit_l2 is not None:
        return None, _refs.early_exit_l2, None

    sol_1pn, _warmstart_label, phi_max_2pn = select_warmstart_orbit(
        warmstart_pn_order=warmstart_pn_order,
        target_pn_order=target_pn_order,
        sol_2pn=sol_2pn, y0=y0, t_max=t_max, nu=nu, csq=CSQ, cqd=CQD,
        a=a, ecc=ecc,
    )

    if dissipative:
        ref_sol = sol_25pn
    elif target_pn_order >= 3 and sol_3pn is not None:
        ref_sol = sol_3pn   # L2 measured against 3PN orbit
    else:
        ref_sol = sol_2pn
    ref2=ref_sol.y.T; t_eval=ref_sol.t
    if len(t_eval)<_n_ref:
        print(f"  NOTE: orbit stopped at {len(t_eval)} pts "
              f"(r reached PN-validity limit r_min={r_pn_min:.3f})")
    _r_ref=np.hypot(ref2[:,0],ref2[:,1])
    _u_max_ref=float((1.0/_r_ref).max())
    _u_min_ref=float((1.0/_r_ref).min())
    # ── [PATCH] Sigmoid band from the REFERENCE EXCURSION ────────────────
    _U_PAD = 0.20
    _span  = max(_u_max_ref - _u_min_ref, 1e-12)
    _pad   = max(_U_PAD*_span, 0.02*0.5*(_u_min_ref+_u_max_ref))
    u_lo_new = max(_u_min_ref - _pad, 1e-4)
    u_hi_new = _u_max_ref + _pad
    print(f"  Band from reference: [{u_lo:.3f},{u_hi:.3f}] -> "
          f"[{u_lo_new:.3f},{u_hi_new:.3f}]  (occupancy "
          f"{100*_span/(u_hi_new-u_lo_new):.0f}%, was "
          f"{100*_span/(u_hi-u_lo):.0f}%)")
    u_lo, u_hi = u_lo_new, u_hi_new
    u_lo = min(u_lo, 0.95*u0)
    u_hi = max(u_hi, 1.05*u0)
    _pr_ref=(ref2[:,0]*ref2[:,2]+ref2[:,1]*ref2[:,3])/_r_ref
    _pr_max_ref=float(np.abs(_pr_ref).max())
    _pr_max_heuristic=max(abs(float(pr0))*2.0+1.5, 2.5)
    # ── [PATCH] pr range from the reference excursion, mirroring the u-band ──
    pr_max_eff = max(1.25*_pr_max_ref, 1.25*abs(float(pr0)), 1e-3)
    print(f"  pr range from reference: {_pr_max_heuristic:.3f} -> "
          f"{pr_max_eff:.3f}  (occupancy "
          f"{100*_pr_max_ref/pr_max_eff:.0f}%, was "
          f"{100*_pr_max_ref/_pr_max_heuristic:.0f}%)")
    # ── Radial period Phi_r, at sub-grid precision ────────────────────── #
    _phi_ref_full=np.unwrap(np.arctan2(ref2[:,1],ref2[:,0])); _phi_ref_full-=_phi_ref_full[0]
    _r_ref_full=np.hypot(ref2[:,0],ref2[:,1])
    _peri_idx=[_i for _i in range(1,len(_r_ref_full)-1)
               if _r_ref_full[_i]<_r_ref_full[_i-1] and _r_ref_full[_i]<_r_ref_full[_i+1]]

    def _parabolic_subsample(_i):
        y0, y1, y2 = _r_ref_full[_i-1], _r_ref_full[_i], _r_ref_full[_i+1]
        denom = (y0 - 2.0*y1 + y2)
        delta = 0.5*(y0 - y2)/denom if abs(denom) > 1e-14 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))   # guard against a bad fit
        if _i+1 < len(_phi_ref_full):
            dphi = 0.5*(_phi_ref_full[_i+1] - _phi_ref_full[_i-1])
        else:
            dphi = _phi_ref_full[_i] - _phi_ref_full[_i-1]
        return _phi_ref_full[_i] + delta*dphi

    if len(_peri_idx) >= 3:
        _phi_peri = np.array([_parabolic_subsample(_i) for _i in _peri_idx])
        _cycle    = np.arange(len(_phi_peri), dtype=float)
        _Phi_r = float(np.polyfit(_cycle, _phi_peri, 1)[0])
        _resid = _phi_peri - (_Phi_r*_cycle + np.polyfit(_cycle, _phi_peri, 1)[1])
        _Phi_r_err = float(np.std(_resid)) if len(_resid) > 2 else float('nan')
    elif len(_peri_idx) >= 2:
        _Phi_r = float(np.mean(np.diff(_phi_ref_full[_peri_idx])))
        _Phi_r_err = float('nan')
    else:
        _Phi_r = 2.0*np.pi   # fallback: assume ~Newtonian period
        _Phi_r_err = float('nan')

    omega_radial_eff=_Phi_r   # passed as omega_radial_init (period, converted inside)
    if n_harmonic > 0:
        print(f"  Radial-harmonic bank: {n_harmonic} harmonics at radial period "
              f"Phi_r={_Phi_r:.6f} rad (freq {2*np.pi/_Phi_r:.6f})"
              + (f"  [fit residual {_Phi_r_err:.2e} rad, "
                 f"{len(_peri_idx)} periapses]" if np.isfinite(_Phi_r_err) else ""))
    print("done.")

    phi_ref=np.unwrap(np.arctan2(ref2[:,1],ref2[:,0]))
    phi_max=float(phi_ref[-1]-phi_ref[0])
    phi_grid=np.linspace(0,phi_max,_n_ref)
    print(f"  phi spans {phi_max:.3f} rad = {phi_max/(2*np.pi):.2f} revolutions"
          +(f"  | inspiral: r {np.hypot(ref2[0,0],ref2[0,1]):.3f}→"
            f"{np.hypot(ref2[-1,0],ref2[-1,1]):.3f}" if dissipative else ""))

    _phi0=phi_ref-phi_ref[0]; _r2=np.hypot(ref2[:,0],ref2[:,1])
    _pr =(ref2[:,0]*ref2[:,2]+ref2[:,1]*ref2[:,3])/_r2
    _L  =ref2[:,0]*ref2[:,3]-ref2[:,1]*ref2[:,2]
    u_ref =np.interp(phi_grid,_phi0,1.0/_r2)
    pr_ref=np.interp(phi_grid,_phi0,_pr)
    L_ref =np.interp(phi_grid,_phi0,_L)
    qx_r=np.cos(phi_grid)/u_ref; qy_r=np.sin(phi_grid)/u_ref
    px_r=pr_ref*np.cos(phi_grid)-(L_ref*u_ref)*np.sin(phi_grid)
    py_r=pr_ref*np.sin(phi_grid)+(L_ref*u_ref)*np.cos(phi_grid)
    ref_norm=np.linalg.norm(np.stack([qx_r,qy_r,px_r,py_r]))+1e-16

    if dissipative:
        _phi0_clipped=np.clip(_phi0,phi_grid[0],phi_grid[-1])
        t_ref_ang=np.interp(phi_grid,_phi0,
                            ref_sol.t)

    _phi0_2=np.unwrap(np.arctan2(sol_2pn.y[1],sol_2pn.y[0]))-\
             np.unwrap(np.arctan2(sol_2pn.y[1],sol_2pn.y[0]))[0]
    _r2pn=np.hypot(sol_2pn.y[0],sol_2pn.y[1])
    u_2pn_ang =torch.tensor(np.interp(phi_grid,_phi0_2,1.0/_r2pn),dtype=torch.float32,device=dev)
    pr_2pn_ang=torch.tensor(np.interp(phi_grid,_phi0_2,
                (sol_2pn.y[0]*sol_2pn.y[2]+sol_2pn.y[1]*sol_2pn.y[3])/_r2pn),
                dtype=torch.float32,device=dev)
    L_2pn_ang =torch.tensor(np.full(len(phi_grid),float(L0)),dtype=torch.float32,device=dev)
    phi_grid_t=torch.tensor(phi_grid,dtype=torch.float32,device=dev)

    # ── Warm-start tensors on shared phi grid — built from SELECTED orbit ──
    _phi_ws_ang  = np.unwrap(np.arctan2(sol_1pn.y[1], sol_1pn.y[0]))
    _phi_ws_ang -= _phi_ws_ang[0]
    _r_ws_ang    = np.hypot(sol_1pn.y[0], sol_1pn.y[1])
    _pr_ws_ang   = (sol_1pn.y[0]*sol_1pn.y[2]+sol_1pn.y[1]*sol_1pn.y[3])/_r_ws_ang
    _phi0_1 = _phi_ws_ang; _r1pn = _r_ws_ang; _pr1pn = _pr_ws_ang  # plot compat
    phi_grid_1pn_t = phi_grid_t
    u_1pn_ang  = torch.tensor(np.interp(phi_grid, _phi_ws_ang, 1.0/_r_ws_ang),
                              dtype=torch.float32, device=dev)
    pr_1pn_ang = torch.tensor(np.interp(phi_grid, _phi_ws_ang, _pr_ws_ang),
                              dtype=torch.float32, device=dev)
    L_1pn_ang  = torch.tensor(np.full(len(phi_grid), float(L0)),
                              dtype=torch.float32, device=dev)

    all_q=np.vstack([sol_2pn.y.T, sol_1pn.y.T, ref2])
    q_pad=(all_q[:,0:2].max()-all_q[:,0:2].min())*0.15
    q_min=all_q[:,0:2].min()-q_pad; q_max=all_q[:,0:2].max()+q_pad

    # ── target-PN conserved energy ─────────────────────────────────────── #
    _q_ic = torch.tensor([[float(q0),0.0]],dtype=torch.float64)
    _p_ic = torch.tensor([[float(pr0),float(p0t)]],dtype=torch.float64)
    E_target   = float(compute_hamiltonian_r(_q_ic,_p_ic,target_pn_order,nu,CSQ,CQD).item())
    E_target_t = torch.tensor(E_target, dtype=torch.float32, device=dev)
    print(f"  target-PN({target_pn_order}) energy E*={E_target:.6f}")

    # ── model ─────────────────────────────────────────────────────────── #
    torch.manual_seed(0)
    train_long_orbit._best_joint=float('inf')  # reset for this run
    model=AngleOrbitPINN4(phi_max,u0,pr0,L0,u_lo,u_hi,dissipative=dissipative,
                          n_fourier=n_fourier,fourier_sigma=fourier_sigma,
                          hidden=hidden,depth=depth,
                          n_fourier_hi=n_fourier_hi,
                          fourier_sigma_hi=fourier_sigma_hi,
                          pr_max_override=pr_max_eff,
                          n_harmonic=n_harmonic,
                          omega_radial_init=omega_radial_eff).to(dev)
    opt, sched, start_ep, best_l2s, best_l2t, _tgt, hist = initialize_adam_cosine(
        model, lr=lr, n_epochs=n_epochs, load_checkpoint=_load_ckpt,
    )

    l2s_arr=list(hist.get("l2_shape",[])); l2t_arr=list(hist.get("l2_time",[]))
    l2_ep=list(hist.get("l2_epochs",[])) or [i*log_every for i in range(len(l2s_arr))]
    ep_arr=list(hist.get("epochs",[])); loss_arr=list(hist.get("loss",[]))

    # ── physics residual ─────────────────────────────────────────────── #
    def phi_residual(phi):
        return angle_residual(
            model, phi, dissipative=dissipative, target_pn_order=target_pn_order,
            nu=nu, csq=CSQ, cqd=CQD, c_scalar=C_SCALAR,
            include_energy_balance=False,
        )

    # ── timing loss ───────────────────────────────────────────────────── #
    # ── metrics ───────────────────────────────────────────────────────── #
    _diag = OrbitDiagnostics(
        model=model, device=dev, phi_max=phi_max, phi_grid=phi_grid,
        qx_ref=qx_r, qy_ref=qy_r, px_ref=px_r, py_ref=py_r, ref_norm=ref_norm,
        ref_time_state=ref2, t_eval=t_eval, target_pn_order=target_pn_order,
        nu=nu, csq=CSQ, cqd=CQD, compute_l2_time=compute_l2_time,
        dissipative=dissipative, t_ref_ang=(t_ref_ang if dissipative else None),
    )
    _net_out = _diag.net_out
    l2_shape = _diag.shape_error
    def _pred_shape():
        return _diag.shape_prediction(include_upl=True)
    l2_time_and_pred = _diag.time_error_and_prediction

    # ── PLOT RECORD: static part ─────────────────────────────────────────── #
    _rec_path = record_file or os.path.join(ckpt_dir, _PLOTREC)
    _rec = _PlotRecorder(_rec_path, start_ep)
    configure_standard_static(
        _rec, label=label, tag=tag, warmstart_label=_warmstart_label,
        target_pn_order=target_pn_order, transfer_from_2pn=transfer_from_2pn,
        compute_l2_time=compute_l2_time, dissipative=dissipative,
        nu=nu, p0_factor=p0_factor, r0_km=r0_km, pr0=pr0, ecc=ecc, L0=L0,
        n_orbits=n_orbits, l2_target=l2_target, phi_max=phi_max, phi_grid=phi_grid,
        q_min=q_min, q_max=q_max, sol_0pn=sol_0pn_ref, sol_1pn=sol_1pn_ref,
        sol_2pn=sol_2pn, sol_3pn=sol_3pn, warm_phi=_phi0_1, warm_r=_r1pn,
        record_every_loss=record_every_loss, record_every_l2=record_every_l2,
        record_every_orbit=record_every_orbit,
    )
    print(f"  Plot record → {_rec_path}  (loss/{record_every_loss}, "
          f"L2/{record_every_l2}, orbit/{record_every_orbit} ep)")
    _record_controller = StandardRecordController(
        _rec, net_out=_net_out, ep_arr=ep_arr, loss_arr=loss_arr, l2_ep=l2_ep,
        l2s_arr=l2s_arr, l2t_arr=l2t_arr, record_every_loss=record_every_loss,
        record_every_l2=record_every_l2, record_every_orbit=record_every_orbit,
    )
    _record_step = _record_controller.record_step

    fig,axes=plt.subplots(1,3,figsize=(18,5))
    disp=live_display(fig, live_plot)
    stage_label=["","Stage1:Transfer","Stage2:Causal","Stage3:Standard"]
    _plotter = TrainingPlotter(
        fig=fig, axes=axes, display=disp, scheduler=sched,
        sol_0pn=sol_0pn_ref, sol_1pn=sol_1pn_ref, sol_2pn=sol_2pn,
        sol_25pn=sol_25pn, sol_3pn=sol_3pn, phi_max=phi_max, phi_grid=phi_grid,
        warm_phi=_phi0_1, warm_r=_r1pn, warmstart_label=_warmstart_label,
        target_pn_order=target_pn_order, transfer_from_2pn=transfer_from_2pn,
        dissipative=dissipative, q_min=q_min, q_max=q_max, stage_label=stage_label,
        ep_arr=ep_arr, loss_arr=loss_arr, l2_ep=l2_ep, l2s_arr=l2s_arr, l2t_arr=l2t_arr,
        compute_l2_time=compute_l2_time, l2_target=l2_target, label=label, tag=tag,
        net_out=_net_out, ckpt_dir=ckpt_dir, record_every_orbit=record_every_orbit,
        save_every_redraw=False, show_dissipative_target=False,
    )

    def _redraw(ep, l_val, l2s, l2t, pred, stage=0):
        _plotter.redraw(
            ep, l_val, l2s, l2t, pred, stage,
            best_l2s=best_l2s, best_l2t=best_l2t,
        )

    # ── STAGE 1: TRANSFER LEARNING (conservative OR dissipative) ────────── #
    if transfer_from_2pn and pretrain_epochs > 0 and start_ep == 0:
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  STAGE 1: Transfer learning warm-start              │")
        print(f"  │  Fits to {_warmstart_label} (same phi range)             │")
        print(f"  │  Stage2 PDE: {target_pn_order}PN Hamiltonian enforces physics.  │")
        print("  └─────────────────────────────────────────────────────┘")
        opt_pre = torch.optim.Adam([
            {'params': [p for n,p in model.named_parameters() if n != 'log_omega']},
            {'params': [model.log_omega], 'lr': lr*1e-2},
        ], lr=lr)
            
        sched_pre=torch.optim.lr_scheduler.StepLR(
            opt_pre, step_size=max(1,pretrain_epochs//3), gamma=0.3)
        best_pre=float("inf"); best_sd=None
        pbar_pre=tqdm(range(pretrain_epochs),desc=f"  Stage1 [{label}]",leave=True)
        _mse_target=1e-5
        for ep in pbar_pre:
            opt_pre.zero_grad()
            out=model(phi_grid_t.view(-1,1).requires_grad_(False))
            u_p=out[:,0]; pr_p=out[:,1]; L_p=out[:,2]
            loss_pre=(nn.functional.mse_loss(u_p, u_1pn_ang)+
                      nn.functional.mse_loss(pr_p,pr_1pn_ang)+
                      nn.functional.mse_loss(L_p, L_1pn_ang))
            loss_pre.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt_pre.step(); sched_pre.step()
            lv=float(loss_pre)
            if lv<best_pre:
                best_pre=lv; best_sd={k:v.clone() for k,v in model.state_dict().items()}
            if ep%2000==0:
                pbar_pre.set_postfix(mse=f"{lv:.2e}",best=f"{best_pre:.2e}")
            if best_pre<_mse_target: break  # good enough — move to Stage 2
        if best_sd is not None:
            model.load_state_dict(best_sd)
            print(f"  Stage1: best weights restored (MSE={best_pre:.2e})")
        else:
            print("  Stage1: WARNING — no checkpoint saved, keeping last weights")
        l2s_after_pretrain=l2_shape()
        print(f"  Stage1 done. Best MSE={best_pre:.2e}  L2_shape={l2s_after_pretrain:.3e}")
        _gap_ref = _l2_23 if (target_pn_order>=3 and sol_3pn is not None) else float('inf')
        if l2s_after_pretrain > _gap_ref:
            print(f"  [Stage1 WARNING] L2_shape ({l2s_after_pretrain:.2e}) > 2PN-3PN gap "
                  f"({_gap_ref:.2e}). Consider increasing pretrain_epochs.")
            print(f"  Stage2 starts from a position FURTHER from 3PN than the 2PN orbit itself.")
        # ── Post-pretrain snapshot: visualise the 1PN-fitted network ──── #
        print("  Plotting post-Stage1 snapshot ...")
        _dummy_pred=np.zeros_like(ref2)
        l2s_arr.append(l2s_after_pretrain); l2_ep.append(0); l2t_arr.append(float('inf'))
        ep_arr.append(0); loss_arr.append(float('nan'))
        _redraw(0, float('nan'), l2s_after_pretrain, float('inf'), _dummy_pred, stage=1)
        _record_step(0, force=True)   # flushes loss/l2 (incl. ep 0) + an orbit snapshot
        print(f"  Post-Stage1 L2 shape={l2s_after_pretrain:.3e} "
              f"(network fitted to {_warmstart_label}, measured vs {target_pn_order}PN target)")
        print(f"    → this L2 is the {_warmstart_label.split(' ')[0]}-to-{target_pn_order}PN gap; "
              f"Stage 2 PDE closes it.")
        print()
        # ── Guard: if warm-start already meets target, skip Stage 2 ─── #
        if l2s_after_pretrain <= l2_target:
            print(f"  [Stage 2 SKIPPED] Post-Stage1 L2 ({l2s_after_pretrain:.2e}) already"
                  f" ≤ l2_target ({l2_target:.0e}). Returning warm-start solution.")
            _save_ckpt(0, model, None, None, l2s_after_pretrain, float('inf'), True, hist)
            _rec.save()
            return None, l2s_after_pretrain, None
    elif not transfer_from_2pn:
        print("  [Stage 1 skipped: transfer_from_2pn=False]")
    elif start_ep > 0:
        print("  [Stage 1 skipped: resuming from checkpoint]")

    # ── STAGES 2+3: PINN training ──────────────────────────────────────── #
    causal_end_ep = int(n_epochs * causal_fraction) if _do_causal else 0
    curric_end     = curriculum_start + curriculum_epochs  # used in both branches
    l2s_below=0; l2t_below=0
    _prev_stage_march = 0   # for marching-stage transition prints
    pbar=tqdm(range(start_ep,n_epochs+1),desc=f"  {tag} [{label}]",
              initial=start_ep,total=n_epochs+1,leave=True)
    for ep in pbar:
        opt.zero_grad()

        # ── Curriculum weights ────────────────────────────────────────────
        alpha_cur, beta_cur = curriculum_weights(
            ep, dissipative=dissipative, target_pn_order=target_pn_order,
            curriculum_start=curriculum_start, curriculum_epochs=curriculum_epochs,
            anchor_floor=0.0,
        )

        use_causal=(_do_causal and causal_eps>0 and ep<causal_end_ep)

        # ── Expanding-window (adaptive time marching) ────────────────────
        _stage_march, phi_window = marching_window(
            ep, n_epochs=n_epochs, n_march=n_march,
            march_fraction=march_fraction, phi_max=phi_max,
        )

        if use_causal:
            phi_raw=torch.rand(n_colloc,device=dev)*phi_window
            phi_sorted,_=torch.sort(phi_raw)
            phi=phi_sorted.view(-1,1).requires_grad_(True)
            R_u,R_pr,R_L,_,H_col=phi_residual(phi)
            r_per_pt=(R_u**2+R_pr**2).squeeze()
            w = causal_weights(
                r_per_pt, eps=causal_eps, epoch=ep,
                end_epoch=causal_end_ep, anneal=True,
            )
            l_pde=(w*r_per_pt).mean()
            l_L  =(w*(R_L**2).squeeze()).mean()
        else:
            phi=(torch.rand(n_colloc,1,device=dev)*phi_window).requires_grad_(True)
            R_u,R_pr,R_L,_,H_col=phi_residual(phi)
            l_pde=(R_u**2).mean()+(R_pr**2).mean()
            l_L  =(R_L**2).mean() if dissipative else torch.zeros((),device=dev)

        # ── Anchor loss ────────────────────────────────────────────────
        if dissipative and alpha_cur > 0.0:
            out_c = model(phi_grid_t.view(-1,1))
            l_anchor = (nn.functional.mse_loss(out_c[:,0], u_2pn_ang)
                      + nn.functional.mse_loss(out_c[:,1], pr_2pn_ang)
                      + nn.functional.mse_loss(out_c[:,2], L_2pn_ang))
        elif (not dissipative) and transfer_from_2pn and alpha_cur > 0.0:
            out_a = model(phi_grid_t.view(-1,1))
            l_anchor = (nn.functional.mse_loss(out_a[:,0], u_1pn_ang)
                      + nn.functional.mse_loss(out_a[:,1], pr_1pn_ang))
        else:
            l_anchor = torch.zeros((), device=dev)

        # ── Energy conservation loss (target-PN H = E*) ────────────────
        l_energy = ((H_col.view(-1) - E_target_t)**2).mean()

        # ── Total loss ──────────────────────────────────────────────────
        loss = (w_anchor * alpha_cur * l_anchor
                + beta_cur * (l_pde + w_L*l_L + w_energy*l_energy)
                + (0.0 if dissipative else 0.5*l_energy))  # always-on energy anchor

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
        sched.step()
        l_val=float(loss) if torch.isfinite(loss) else float("nan")
        ep_arr.append(ep); loss_arr.append(l_val)

        # ── Stage transition announcements ────────────────────────────────
        if dissipative and ep==curriculum_start and ep>start_ep:
            print(f"\n  ── Curriculum decay START (α: 1→0 over {curriculum_epochs} ep) ──")
        if not dissipative and ep==curriculum_start and ep>start_ep:
            print(f"\n  ── Conservative: β=1 reached (full {target_pn_order}PN PDE, ep {ep}) ──")
        if dissipative and ep==curric_end and ep>start_ep:
            print(f"\n  ── Curriculum OFF: pure residuals from ep {ep} ──")
        if _do_causal and ep==causal_end_ep and ep>start_ep:
            print(f"\n  ── Causal stage END: switching to random collocation (ep {ep}) ──")
        if n_march > 1 and ep>start_ep and _stage_march != _prev_stage_march:
            print(f"\n  ── March window {_stage_march}/{n_march}: "
                  f"phi ∈ [0, {phi_window:.2f}] (ep {ep}) ──")
        if n_march > 1:
            _prev_stage_march = _stage_march

        if ep%log_every==0:
            l2s=l2_shape(); l2t,pred=l2_time_and_pred()
            l2s_arr.append(l2s); l2t_arr.append(l2t); l2_ep.append(ep)
            hist={"epochs":ep_arr,"loss":loss_arr,"l2_shape":l2s_arr,"l2_time":l2t_arr,"l2_epochs":l2_ep}
            if l2s<best_l2s:
                best_l2s=l2s; torch.save(model.state_dict(),os.path.join(ckpt_dir,_BEST))
            if l2t<best_l2t: best_l2t=l2t
            if l2s < l2_target:
                _joint=l2s*l2t
                if not hasattr(train_long_orbit,'_best_joint'):
                    train_long_orbit._best_joint=float('inf')
                if _joint<train_long_orbit._best_joint:
                    train_long_orbit._best_joint=_joint
                    torch.save(model.state_dict(),os.path.join(ckpt_dir,_BEST_BOTH))
            # ── Dual convergence criterion ────────────────────────────────────
            l2s_below=(l2s_below+1) if l2s<l2_target else 0
            l2t_below=(l2t_below+1) if (compute_l2_time and best_l2t<l2_target) else 0
            stage=2 if use_causal else 3
            if alpha_cur > 0:
                cur_str = f"α={alpha_cur:.2f}"
            elif not dissipative and beta_cur < 1.0:
                cur_str = f"β={beta_cur:.2f}"  # show PDE ramp progress
            else:
                cur_str = f"S{stage}"

            W = max(2, int(convergence_window))
            _converged = False
            _conv_reason = ""
            if len(l2s_arr) >= W:
                _win   = l2s_arr[-W:]
                _wmean = float(np.mean(_win))
                _wstd  = float(np.std(_win))
                _wcv   = _wstd / (_wmean + 1e-16)
                _wtrend= float(_win[-1] - _win[0]) / W   # positive = getting worse
                _crit_a = _wmean < l2_target
                _crit_b = _wcv < convergence_cv
                _crit_c = _wtrend > -0.5 * _wmean
                _converged = _crit_a and _crit_b and _crit_c
                _conv_reason = (f"win_mean={_wmean:.2e} cv={_wcv:.2f} trend={_wtrend:+.2e}"
                                f" [A={'✓' if _crit_a else '✗'}"
                                f" B={'✓' if _crit_b else '✗'}"
                                f" C={'✓' if _crit_c else '✗'}]")
            else:
                _wmean = float(np.mean(l2s_arr)) if l2s_arr else float('inf')
                _wcv   = float('inf')

            if compute_l2_time:
                pbar.set_postfix(loss=f"{l_val:.2e}",L2s=f"{l2s:.2e}",
                                 wμ=f"{_wmean:.1e}",cv=f"{_wcv:.2f}",
                                 best=f"{best_l2s:.2e}",cur=cur_str)
            else:
                pbar.set_postfix(loss=f"{l_val:.2e}",L2s=f"{l2s:.2e}",
                                 wμ=f"{_wmean:.1e}",cv=f"{_wcv:.2f}",
                                 best=f"{best_l2s:.2e}",cur=cur_str)

            if _converged:
                print(f"\n  ✅ CONVERGED [{tag}]: {_conv_reason}")
                print(f"     best_L2s={best_l2s:.2e}  window_mean={_wmean:.2e} < {l2_target:.0e}")
                break
        _record_step(ep)
        if ep%checkpoint_every==0 and ep>start_ep:
            hist={"epochs":ep_arr,"loss":loss_arr,"l2_shape":l2s_arr,"l2_time":l2t_arr,"l2_epochs":l2_ep}
            _save_ckpt(ep,model,opt,sched,best_l2s,best_l2t,best_l2s<l2_target,hist)
        if ep%plot_every==0 and ep>start_ep and len(l2_ep)>0:
            _,pred=l2_time_and_pred()
            stage=2 if use_causal else 3
            _redraw(ep,l_val,l2s_arr[-1],l2t_arr[-1],pred,stage)
    plt.close(fig)
    _record_step(ep, force=True)   # final epoch, whatever it was (n_epochs or a `break`)

    # ── final save ─────────────────────────────────────────────────────── #
    hist={"epochs":ep_arr,"loss":loss_arr,"l2_shape":l2s_arr,"l2_time":l2t_arr,"l2_epochs":l2_ep}
    _save_ckpt(n_epochs,model,opt,sched,best_l2s,best_l2t,best_l2s<l2_target,hist)
    bp=os.path.join(ckpt_dir,_BEST)
    bp_both=os.path.join(ckpt_dir,_BEST_BOTH)
    if os.path.exists(bp_both):
        model.load_state_dict(torch.load(bp_both,map_location=dev,weights_only=True))
        l2t_check,_=l2_time_and_pred()
        l2s_check=l2_shape()
        print(f"  Joint-best checkpoint: L2s={l2s_check:.3e}  L2t={l2t_check:.3e}")
        best_l2t=min(best_l2t,l2t_check)
        best_l2s=min(best_l2s,l2s_check)
    elif os.path.exists(bp):
        model.load_state_dict(torch.load(bp,map_location=dev,weights_only=True))
    l2t_fin,pred_t=l2_time_and_pred()
    _pred_shape_arr, _pred_upl = _pred_shape()          # [PATCH]
    try:
        _rec.final.update(best_u=_pred_upl[:,0].astype(np.float32),
                          best_pr=_pred_upl[:,1].astype(np.float32),
                          best_L=_pred_upl[:,2].astype(np.float32),
                          best_l2_shape=float(l2_shape()))
        _rec.save()
    except Exception as _e:
        print(f"  (plot-record final-best snapshot skipped: {_e})")
    _ref_shape_arr = np.stack([qx_r,qy_r,px_r,py_r],1)  # [PATCH]
    np.savez(os.path.join(ckpt_dir,_ANPZ),t_eval=t_eval,pr0=pr0,ref_2pn_cons=sol_2pn.y.T,
             ref_target=ref2,pred_best_t=pred_t,phi_grid=phi_grid,
             dissipative=dissipative,L_ref_ang=L_ref,
             # ── [PATCH] everything needed for offline figures ──
             pred_shape=_pred_shape_arr, ref_shape=_ref_shape_arr,
             pred_u_pr_L=_pred_upl, u_ref=u_ref, pr_ref=pr_ref,
             ref_1pn=(sol_1pn_ref.y.T if sol_1pn_ref is not None else np.zeros((0,4))),
             ref_0pn=(sol_0pn_ref.y.T if sol_0pn_ref is not None else np.zeros((0,4))),
             u_lo=u_lo, u_hi=u_hi, phi_max=phi_max, n_orbits=n_orbits,
             ecc=ecc, nu=nu, p0_factor=p0_factor, r0_km=r0_km, pr0_factor=pr0,
             n_harmonic=n_harmonic, n_epochs=n_epochs, lr=lr, Phi_r=_Phi_r,
             lbfgs_rounds=lbfgs_rounds, lbfgs_cycles=lbfgs_cycles,
             l2_epochs=np.array(l2_ep),l2_shape=np.array(l2s_arr),l2_time=np.array(l2t_arr))
    with open(os.path.join(ckpt_dir,_ANJSON),'w') as f:
        json.dump(dict(regime=dict(nu=nu,p0_factor=p0_factor,r0_km=r0_km,q0=q0,
                  ecc=ecc,T_orb=T,L0=L0,phi_max=phi_max,dissipative=dissipative),
                  training=dict(n_epochs=n_epochs,best_l2_shape=best_l2s,
                  best_l2_time=best_l2t,l2_target=l2_target,
                  target_reached=(best_l2s<l2_target),
                  both_targets=(best_l2s<l2_target and best_l2t<l2_target),
                  transfer_from_2pn=transfer_from_2pn)),f,indent=2)
    print("-"*70)
    print(f"  DONE [{label}] {tag}  L2_shape={best_l2s:.3e}  L2_time={best_l2t:.3e}")
    if best_l2s<l2_target: print("  *** L2_shape TARGET REACHED ***")
    if best_l2t<l2_target: print("  *** L2_time  TARGET REACHED ***")
    print(f"  Saved to: {ckpt_dir}")
    # ── L-BFGS POLISH ────────────────────────────────────────────────
    if n_epochs > 0 and lbfgs_rounds > 0:
        # ── [PATCH] second-order polish, done correctly ──────────────────
        _n_lb = max(n_colloc, 8000)
        _best_sd_lb = {k: v.detach().clone() for k, v in model.state_dict().items()}
        _best_l2_lb = l2_shape()
        print(f"  L-BFGS polish: {lbfgs_rounds} rounds x {lbfgs_cycles} cycles, "
              f"frozen {_n_lb}-pt batches (start L2={_best_l2_lb:.4e})")
        for _rd in range(lbfgs_rounds):
            _phi_frozen = (torch.rand(_n_lb, 1, device=dev) * phi_max)
            _phi_frozen, _ = torch.sort(_phi_frozen, dim=0)
            lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                      history_size=100,
                                      line_search_fn='strong_wolfe',
                                      tolerance_grad=1e-12,
                                      tolerance_change=1e-16)
            def _closure(_pf=_phi_frozen, _opt=lbfgs):
                _opt.zero_grad()
                phi_l = _pf.clone().requires_grad_(True)
                R_u_l, R_pr_l, _, _, H_l = phi_residual(phi_l)
                l_e = ((H_l.view(-1) - E_target_t)**2).mean()
                loss_l = (R_u_l**2).mean() + (R_pr_l**2).mean() + w_energy*l_e
                loss_l.backward()
                return loss_l
            for _c in range(lbfgs_cycles):
                lbfgs.step(_closure)
            _l2_now = l2_shape()
            if _l2_now < _best_l2_lb:
                _best_l2_lb = _l2_now
                _best_sd_lb = {k: v.detach().clone()
                               for k, v in model.state_dict().items()}
            print(f"    round {_rd+1}/{lbfgs_rounds}: L2={_l2_now:.4e}"
                  f"  (best {_best_l2_lb:.4e})")
        model.load_state_dict(_best_sd_lb)          # never degrade
        l2s_polish = _best_l2_lb
        try:
            _op = _net_out()
            _rec.final.update(polish_u=_op[:,0].astype(np.float32),
                              polish_pr=_op[:,1].astype(np.float32),
                              polish_L=_op[:,2].astype(np.float32),
                              polish_l2_shape=float(l2s_polish))
            _rec.save()
        except Exception as _e:
            print(f"  (plot-record polish snapshot skipped: {_e})")
        print(f"  Post-polish L²={l2s_polish:.4e}  (was best={best_l2s:.4e})")
        if l2s_polish < best_l2s:
            best_l2s=l2s_polish
            torch.save(model.state_dict(), os.path.join(ckpt_dir,_BEST))
            print(f"  L-BFGS improved best L² to {best_l2s:.4e}")

    return None, best_l2s, None
