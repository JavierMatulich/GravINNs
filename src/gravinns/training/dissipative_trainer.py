"""Dissipative 2.5PN single-orbit trainer.

This dissipative protocol adds angular-momentum and secular energy-balance constraints
and supports the physics-only provenance audit used in the paper.
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
from ..recording.energy_plot_record import (PLOTREC_FILENAME as _PLOTREC,
                                            EnergyPlotRecorder as _PlotRecorder)
from .checkpointing import CheckpointStore
from .reference import build_reference_orbits, select_warmstart_orbit
from .diagnostics import OrbitDiagnostics
from .residuals import angle_residual
from .schedules import curriculum_weights, marching_window, causal_weights
from .runtime import initialize_adam_cosine
from .live_plot import TrainingPlotter
from .audit import prepare_dissipative_audit
from .recorders import EnergyRecordController, configure_energy_static
from .artifacts import (ANALYSIS_JSON as _ANJSON, ANALYSIS_NPZ as _ANPZ,
                        BEST as _BEST, BEST_BOTH as _BEST_BOTH, CONFIG as _CONFIG)

def train_dissipative(nu, p0_factor, r0_km,
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
                         causal_fraction    = 'auto',  # 'auto' scales with n_orbits;
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
                         lr                 = 3e-3,
                         n_fourier          = 8,
                         fourier_sigma      = 2.0,
                         n_fourier_hi       = 16,   # high-freq Fourier bank for periapsis (0=off)
                         fourier_sigma_hi   = 8.0,  # bandwidth; periapsis residual is 100× apoapsis
                         n_harmonic         = 'auto',# radial-harmonic bank; 'auto' scales
                         hidden             = 64,
                         depth              = 3,
                         checkpoint_every   = 5_000,
                         plot_every         = 2_000,
                         log_every          = 2_000,
                         # ── plot record (re-plot without retraining) ─────
                         record_every_loss  = 500,     # loss flush cadence (0 = off);
                         record_every_l2    = 500,     # L2 flush cadence   (0 = off), same note
                         record_every_energy= 5_000,   # H(phi)+orbit snapshot cadence
                         record_file        = None,    # None → <ckpt_dir>/plot_record.npz
                         w_L                = 1.0,     # L residual weight
                         w_secular          = 50.0,    # ← energy-BALANCE weight
                         w_energy           = 5.0,     # target-PN energy conservation weight
                         w_anchor           = 10.0,
                         anchor_floor       = 0.05,   # dissipative: min anchor
                         no_25pn_data       = True,   # ← PHYSICS-ONLY MODE.
                         l2_target          = 1e-3,
                         patience           = 5,       # kept for compat (ignored if window active)
                         convergence_window = 20,      # W: L2 measurements in sliding window
                         convergence_cv     = 0.5,     # max allowed std/mean (coeff of variation)
                         dev                = None,
                         # ── repo addition (does not change training) ────
                         live_plot          = None):   # None=auto (Jupyter only)

    dev=dev or ("cuda" if torch.cuda.is_available() else "cpu")
    CSQ=_const.C_SQ; CQD=_const.C_QD

    # ── Dissipative physics consistency ──────────────────────────────────
    if dissipative and int(target_pn_order) != 2:
        print(f"  [dissipative] coercing target_pn_order {target_pn_order} -> 2 "
              f"(2.5PN = 2PN Hamiltonian + 2.5PN radiation reaction).")
        target_pn_order = 2

    # ── PHYSICS-ONLY MODE enforcement ────────────────────────────────────
    if dissipative and no_25pn_data and anchor_floor != 0.0:
        print(f"  [physics-only] anchor_floor {anchor_floor} -> 0.0 "
              f"(anchor must fully switch off; no residual data pull)")
        anchor_floor = 0.0

    _do_causal = dissipative if use_causal_training is None else bool(use_causal_training)

    # ── AUTO-SCALING GUARD: causal_fraction & n_harmonic vs n_orbits ──────
    def _clip(v, lo, hi): return max(lo, min(hi, v))
    _cf_rec = round(_clip(0.4 + 0.1*float(n_orbits), 0.5, 0.9), 3)
    _nh_rec = int(_clip(2*int(n_orbits), 8, 32))
    if isinstance(causal_fraction, str) and causal_fraction.lower().strip()=='auto':
        causal_fraction = _cf_rec if _do_causal else 0.5
        if _do_causal:
            print(f"  [auto] causal_fraction -> {causal_fraction:.2f}  "
                  f"(n_orbits={n_orbits}; scales causal phase with horizon)")
    else:
        causal_fraction = float(causal_fraction)
        if _do_causal and causal_fraction < _cf_rec:
            print(f"  [auto] causal_fraction {causal_fraction:.2f} -> {_cf_rec:.2f} "
                  f"(raised to recommended floor for n_orbits={n_orbits}; "
                  f"outer orbits need a longer causal phase)")
            causal_fraction = _cf_rec
    if isinstance(n_harmonic, str) and n_harmonic.lower().strip()=='auto':
        n_harmonic = _nh_rec if dissipative else 0
        if dissipative:
            print(f"  [auto] n_harmonic -> {n_harmonic}  "
                  f"(n_orbits={n_orbits}; ~2 radial harmonics per orbit)")
    else:
        n_harmonic = int(n_harmonic)
        if dissipative and 0 < n_harmonic < _nh_rec:
            print(f"  [auto] n_harmonic {n_harmonic} -> {_nh_rec} "
                  f"(raised to recommended floor for n_orbits={n_orbits})")
            n_harmonic = _nh_rec
    # ── end auto-scaling guard ────────────────────────────────────────────

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
        s1=f"Stage1=2PN-warmstart-transfer({'ON' if transfer_from_2pn else 'OFF'})"
        _causal_str=f"causal(eps={causal_eps},frac={causal_fraction})" if _do_causal else "causal=OFF"
        s2=f"Stage2=curriculum({curriculum_epochs}ep,start={curriculum_start})+{_causal_str}"
        s3="Stage3=standard | PDE=2PN H + 2.5PN F_RR"
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
    _n_ref = max(4000, int(2000*n_orbits))
    t_max = n_orbits*T
    _refs = build_reference_orbits(
        y0=y0, nu=nu, csq=CSQ, cqd=CQD, c_scalar=C_SCALAR,
        t_max=t_max, n_ref=_n_ref, dissipative=dissipative,
        target_pn_order=target_pn_order, pn_valid_thresh=pn_valid_thresh,
        a=a, ecc=ecc, L0=L0, l2_target=l2_target,
        two_pn_rtol=1e-10, two_pn_atol=1e-12,
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
    if _u_max_ref > 0.95*u_hi:
        u_hi_new=1.30*_u_max_ref
        print(f"  Adaptive u_hi: {u_hi:.2f} -> {u_hi_new:.2f} "
              f"(ref orbit reaches u={_u_max_ref:.2f})")
        u_hi=u_hi_new
    _pr_ref=(ref2[:,0]*ref2[:,2]+ref2[:,1]*ref2[:,3])/_r_ref
    _pr_max_ref=float(np.abs(_pr_ref).max())
    _pr_max_heuristic=max(abs(float(pr0))*2.0+1.5, 2.5)
    pr_max_eff=max(_pr_max_heuristic, 1.20*_pr_max_ref)
    if 1.20*_pr_max_ref > _pr_max_heuristic:
        print(f"  Adaptive pr_max: {_pr_max_heuristic:.2f} -> {pr_max_eff:.2f} "
              f"(ref orbit reaches |pr|={_pr_max_ref:.2f})")
    # ── Radial period Phi_r from the reference orbit's periapsis spacing ── #
    _phi_ref_full=np.unwrap(np.arctan2(ref2[:,1],ref2[:,0])); _phi_ref_full-=_phi_ref_full[0]
    _r_ref_full=np.hypot(ref2[:,0],ref2[:,1])
    _peri_idx=[_i for _i in range(1,len(_r_ref_full)-1)
               if _r_ref_full[_i]<_r_ref_full[_i-1] and _r_ref_full[_i]<_r_ref_full[_i+1]]
    if len(_peri_idx)>=2:
        _Phi_r=float(np.mean(np.diff(_phi_ref_full[_peri_idx])))
    else:
        _Phi_r=2.0*np.pi   # fallback: assume ~Newtonian period
    omega_radial_eff=_Phi_r   # passed as omega_radial_init (period, converted inside)
    if n_harmonic > 0:
        print(f"  Radial-harmonic bank: {n_harmonic} harmonics at radial period "
              f"Phi_r={_Phi_r:.4f} rad (freq {2*np.pi/_Phi_r:.4f})")
    print("done.")

    phi_ref=np.unwrap(np.arctan2(ref2[:,1],ref2[:,0]))
    phi_max=float(phi_ref[-1]-phi_ref[0])
    phi_grid=np.linspace(0,phi_max,_n_ref)
    print(f"  phi spans {phi_max:.3f} rad = {phi_max/(2*np.pi):.2f} revolutions"
          +(f"  | inspiral: r {np.hypot(ref2[0,0],ref2[0,1]):.3f}→"
            f"{np.hypot(ref2[-1,0],ref2[-1,1]):.3f}" if dissipative else ""))

    # ── INSPIRAL-STRENGTH DIAGNOSTIC (dissipative only) ──────────────────
    if dissipative:
        def _l2_2p5_gap():
            _p2 =np.unwrap(np.arctan2(sol_2pn.y[1], sol_2pn.y[0])); _p2-=_p2[0]
            _p25=np.unwrap(np.arctan2(sol_25pn.y[1],sol_25pn.y[0])); _p25-=_p25[0]
            _r2 =np.hypot(sol_2pn.y[0], sol_2pn.y[1])
            _r25=np.hypot(sol_25pn.y[0],sol_25pn.y[1])
            _pr2 =(sol_2pn.y[0]*sol_2pn.y[2]+sol_2pn.y[1]*sol_2pn.y[3])/_r2
            _pr25=(sol_25pn.y[0]*sol_25pn.y[2]+sol_25pn.y[1]*sol_25pn.y[3])/_r25
            _pg=np.linspace(0,min(_p2[-1],_p25[-1]),1500)
            _u2 =np.interp(_pg,_p2,1/_r2);  _u25=np.interp(_pg,_p25,1/_r25)
            _pa2=np.interp(_pg,_p2,_pr2);   _pa25=np.interp(_pg,_p25,_pr25)
            _qx2=np.cos(_pg)/_u2; _qy2=np.sin(_pg)/_u2
            _qx25=np.cos(_pg)/_u25; _qy25=np.sin(_pg)/_u25
            _px2=_pa2*np.cos(_pg)-(float(L0)*_u2)*np.sin(_pg)
            _py2=_pa2*np.sin(_pg)+(float(L0)*_u2)*np.cos(_pg)
            _px25=_pa25*np.cos(_pg)-(float(L0)*_u25)*np.sin(_pg)
            _py25=_pa25*np.sin(_pg)+(float(L0)*_u25)*np.cos(_pg)
            _rn=np.linalg.norm(np.stack([_qx25,_qy25,_px25,_py25]))+1e-16
            return float(np.linalg.norm(np.stack([_qx2-_qx25,_qy2-_qy25,
                                                  _px2-_px25,_py2-_py25]))/_rn)
        _r_start=np.hypot(sol_25pn.y[0,0], sol_25pn.y[1,0])
        _r_final=np.hypot(sol_25pn.y[0,-1],sol_25pn.y[1,-1])
        _decay_frac=(_r_start-_r_final)/max(_r_start,1e-12)
        _gap_2p5=_l2_2p5_gap()
        print(f"  inspiral-strength: 2PN↔2.5PN 4D-L2 gap = {_gap_2p5:.3e}  "
              f"(l2_target {l2_target:.0e});  orbital decay over window = "
              f"{_decay_frac*100:+.1f}%")
        _low_decay  = _decay_frac < 0.05          # orbit barely shrinks
        _tight_tgt  = l2_target > 0.25*_gap_2p5   # target eats >25% of the signal
        if _low_decay:
            print(f"  ⚠ WEAK-INSPIRAL (low decay) ──────────────────────────────")
            print(f"     The orbit shrinks only {_decay_frac*100:.1f}% over these "
                  f"{n_orbits} orbits, so")
            print(f"     radiation reaction is a tiny perturbation on the 2PN orbit")
            print(f"     and the network correctly stays near the 2PN warm-start.")
            print(f"     For a visible inspiral, strengthen the field / lengthen it:")
            print(f"       • smaller r0_km  (e.g. 90–120, not 180)")
            print(f"       • higher eccentricity (lower p0_factor, ~0.5)")
            print(f"       • more orbits (n_orbits ↑) to accumulate decay")
            print(f"  ──────────────────────────────────────────────────────────")
        if _tight_tgt:
            print(f"  ⚠ TIGHT-TARGET NOTE ──────────────────────────────────────")
            print(f"     l2_target={l2_target:.0e} is a large fraction of the whole")
            print(f"     2PN→2.5PN signal ({_gap_2p5:.1e}); the net must capture "
                  f"{100*(1-l2_target/_gap_2p5):.0f}% of")
            print(f"     the entire radiation-reaction correction to reach it. If it")
            print(f"     plateaus somewhat above target, that is expected here — the")
            print(f"     result is still a good 2.5PN orbit. Relax l2_target (e.g.")
            print(f"     {max(2e-3,0.3*_gap_2p5):.0e}) or strengthen the inspiral to widen the gap.")
            print(f"  ──────────────────────────────────────────────────────────")

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

    # ── 2.5PN inspiral anchor (THE fix for the long dissipative case) ─────
    u_25pn_ang =torch.tensor(u_ref, dtype=torch.float32,device=dev)
    pr_25pn_ang=torch.tensor(pr_ref,dtype=torch.float32,device=dev)
    L_25pn_ang =torch.tensor(L_ref, dtype=torch.float32,device=dev)

    # ── CLEAN (physics-only) anchor: the CONSERVATIVE 2PN orbit ──────────
    u_2pn_ang = pr_2pn_ang = L_2pn_ang = None
    if dissipative:
        try:
            _p2=np.unwrap(np.arctan2(sol_2pn.y[1],sol_2pn.y[0])); _p2=_p2-_p2[0]
            _r2c=np.hypot(sol_2pn.y[0],sol_2pn.y[1])
            _pr2c=(sol_2pn.y[0]*sol_2pn.y[2]+sol_2pn.y[1]*sol_2pn.y[3])/_r2c
            _L2c =sol_2pn.y[0]*sol_2pn.y[3]-sol_2pn.y[1]*sol_2pn.y[2]
            _u2c =np.interp(phi_grid,_p2,1.0/_r2c)
            _pr2i=np.interp(phi_grid,_p2,_pr2c)
            _L2i =np.interp(phi_grid,_p2,_L2c)
            u_2pn_ang =torch.tensor(_u2c ,dtype=torch.float32,device=dev)
            pr_2pn_ang=torch.tensor(_pr2i,dtype=torch.float32,device=dev)
            L_2pn_ang =torch.tensor(_L2i ,dtype=torch.float32,device=dev)
        except Exception as _e:
            print(f"  (clean 2PN anchor unavailable: {_e})")

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
    train_dissipative._best_joint=float('inf')  # reset for this run
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
    ep_arr=[]; loss_arr=[]

    # ── physics residual ─────────────────────────────────────────────── #
    def phi_residual(phi):
        return angle_residual(
            model, phi, dissipative=dissipative, target_pn_order=target_pn_order,
            nu=nu, csq=CSQ, cqd=CQD, c_scalar=C_SCALAR,
            include_energy_balance=True,
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
    l2_time_and_pred = _diag.time_error_and_prediction
    _net_energy = _diag.energy

    _H_ref_ang=None
    if dissipative:
        try:
            _qr=np.stack([qx_r,qy_r],1); _pr_=np.stack([px_r,py_r],1)
            _qt=torch.tensor(_qr,dtype=torch.float64); _pt=torch.tensor(_pr_,dtype=torch.float64)
            with torch.no_grad():
                _H_ref_ang=compute_hamiltonian_r(_qt,_pt,target_pn_order,
                                                 nu,CSQ,CQD).cpu().numpy().reshape(-1)
        except Exception:
            _H_ref_ang=None

    # ── PLOT RECORD: static part + helpers ────────────────────────────── #
    _rec_path = record_file or os.path.join(ckpt_dir, _PLOTREC)
    _rec = _PlotRecorder(_rec_path, start_ep)
    configure_energy_static(
        _rec, label=label, tag=tag, warmstart_label=_warmstart_label,
        target_pn_order=target_pn_order, transfer_from_2pn=transfer_from_2pn,
        compute_l2_time=compute_l2_time, dissipative=dissipative,
        nu=nu, p0_factor=p0_factor, r0_km=r0_km, pr0=pr0, ecc=ecc, L0=L0,
        n_orbits=n_orbits, l2_target=l2_target, E_target=E_target,
        phi_max=phi_max, phi_grid=phi_grid, q_min=q_min, q_max=q_max,
        H_ref=_H_ref_ang, sol_0pn=sol_0pn_ref, sol_1pn=sol_1pn_ref,
        sol_2pn=sol_2pn, sol_25pn=sol_25pn, sol_3pn=sol_3pn,
        warm_phi=_phi0_1, warm_r=_r1pn, record_every_loss=record_every_loss,
        record_every_l2=record_every_l2, record_every_energy=record_every_energy,
    )
    print(f"  Plot record → {_rec_path}  (loss/{record_every_loss}, "
          f"L2/{record_every_l2}, energy/{record_every_energy} ep)")
    _record_controller = EnergyRecordController(
        _rec, net_out=_net_out, net_energy=_net_energy, scheduler=sched,
        ep_arr=ep_arr, loss_arr=loss_arr, l2_ep=l2_ep, l2s_arr=l2s_arr,
        l2t_arr=l2t_arr, record_every_loss=record_every_loss,
        record_every_l2=record_every_l2, record_every_energy=record_every_energy,
    )
    _record_step = _record_controller.record_step

    # ── figure ────────────────────────────────────────────────────────── #
    _ncols = 4 if dissipative else 3
    fig,axes=plt.subplots(1,_ncols,figsize=(6*_ncols,5))
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
        net_out=_net_out, ckpt_dir=ckpt_dir,
        save_every_redraw=True, show_dissipative_target=True,
        energy_fn=_net_energy, energy_ref=_H_ref_ang, energy_target=E_target,
    )

    def _redraw(ep, l_val, l2s, l2t, pred, stage=0):
        _plotter.redraw(
            ep, l_val, l2s, l2t, pred, stage,
            best_l2s=best_l2s, best_l2t=best_l2t,
        )

    _audit_tensors, _audit_done = prepare_dissipative_audit(
        dissipative=dissipative, no_25pn_data=no_25pn_data,
        anchor_floor=anchor_floor, tensors=(u_25pn_ang, pr_25pn_ang, L_25pn_ang),
    )

    # ── STAGE 1: TRANSFER LEARNING (conservative OR dissipative) ────────── #
    if transfer_from_2pn and pretrain_epochs > 0 and start_ep == 0:
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  STAGE 1: Transfer learning warm-start              │")
        print(f"  │  Fits to {_warmstart_label} (same phi range)             │")
        print(f"  │  Stage2 PDE: {target_pn_order}PN Hamiltonian enforces physics.  │")
        print("  └─────────────────────────────────────────────────────┘")
        opt_pre=torch.optim.Adam(model.parameters(),lr=pretrain_lr)
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
        _record_step(0,1,force=True)               # flushes l2 (incl. ep 0) + energy
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
    _last_loop_ep=None; _stage_now=3
    for ep in pbar:
        _last_loop_ep=ep
        opt.zero_grad()

        # ── Curriculum weights ────────────────────────────────────────────
        alpha_cur, beta_cur = curriculum_weights(
            ep, dissipative=dissipative, target_pn_order=target_pn_order,
            curriculum_start=curriculum_start, curriculum_epochs=curriculum_epochs,
            anchor_floor=anchor_floor,
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
            R_u,R_pr,R_L,_,H_col,R_E=phi_residual(phi)
            r_per_pt=(R_u**2+R_pr**2).squeeze()
            w = causal_weights(r_per_pt, eps=causal_eps)
            l_pde=(w*r_per_pt).mean()
            l_L  =(w*(R_L**2).squeeze()).mean()
            l_secular=(w*(R_E**2).squeeze()).mean() if dissipative else torch.zeros((),device=dev)
        else:
            phi=(torch.rand(n_colloc,1,device=dev)*phi_window).requires_grad_(True)
            R_u,R_pr,R_L,_,H_col,R_E=phi_residual(phi)
            l_pde=(R_u**2).mean()+(R_pr**2).mean()
            l_L  =(R_L**2).mean() if dissipative else torch.zeros((),device=dev)
            l_secular=(R_E**2).mean() if dissipative else torch.zeros((),device=dev)

        # ── Anchor loss ────────────────────────────────────────────────
        if dissipative and alpha_cur > 0.0:
            out_c = model(phi_grid_t.view(-1,1))
            if no_25pn_data:
                if u_2pn_ang is not None:
                    l_anchor = (nn.functional.mse_loss(out_c[:,0], u_2pn_ang)
                              + nn.functional.mse_loss(out_c[:,1], pr_2pn_ang)
                              + nn.functional.mse_loss(out_c[:,2], L_2pn_ang))
                else:
                    l_anchor = torch.zeros((), device=dev)
            else:
                l_anchor = (nn.functional.mse_loss(out_c[:,0], u_25pn_ang)
                          + nn.functional.mse_loss(out_c[:,1], pr_25pn_ang)
                          + nn.functional.mse_loss(out_c[:,2], L_25pn_ang))
        elif (not dissipative) and transfer_from_2pn and alpha_cur > 0.0:
            out_a = model(phi_grid_t.view(-1,1))
            l_anchor = (nn.functional.mse_loss(out_a[:,0], u_1pn_ang)
                      + nn.functional.mse_loss(out_a[:,1], pr_1pn_ang))
        else:
            l_anchor = torch.zeros((), device=dev)

        # ── Energy conservation loss (target-PN H = E*) ────────────────
        l_energy = ((H_col.view(-1) - E_target_t)**2).mean()

        # ── Total loss ──────────────────────────────────────────────────
        _w_energy_eff = 0.0 if dissipative else w_energy
        _l_secular_term = (w_secular*l_secular) if dissipative else 0.0
        loss = (w_anchor * alpha_cur * l_anchor
                + beta_cur * (l_pde + w_L*l_L + _w_energy_eff*l_energy + _l_secular_term)
                + (0.0 if dissipative else 0.5*l_energy))  # always-on energy anchor (cons. only)

        if torch.isfinite(loss):
            loss.backward()
            # ── one-shot provenance verification on the first backward pass ──
            if dissipative and not _audit_done:
                _leaked=[n for n,t in zip(('u_25pn','pr_25pn','L_25pn'),_audit_tensors)
                         if (t is not None and getattr(t,'grad',None) is not None
                             and torch.any(t.grad!=0))]
                if no_25pn_data and _leaked:
                    raise RuntimeError(
                        "PROVENANCE VIOLATION: no_25pn_data=True but the loss "
                        f"used 2.5PN RK data via {_leaked}. Aborting.")
                print("  [audit] first backward pass: 2.5PN RK tensors received "
                      + ("NO gradient — loss is physics-only ✓" if not _leaked
                         else f"GRADIENT via {_leaked} (2.5PN data in use)"))
                _audit_done=True
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

        _stage_now=2 if use_causal else 3
        if ep%log_every==0:
            l2s=l2_shape(); l2t,pred=l2_time_and_pred()
            l2s_arr.append(l2s); l2t_arr.append(l2t); l2_ep.append(ep)
            hist={"loss":loss_arr,"l2_shape":l2s_arr,"l2_time":l2t_arr,"l2_epochs":l2_ep}
            if l2s<best_l2s:
                best_l2s=l2s; torch.save(model.state_dict(),os.path.join(ckpt_dir,_BEST))
            if l2t<best_l2t: best_l2t=l2t
            if l2s < l2_target:
                _joint=l2s*l2t
                if not hasattr(train_dissipative,'_best_joint'):
                    train_dissipative._best_joint=float('inf')
                if _joint<train_dissipative._best_joint:
                    train_dissipative._best_joint=_joint
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
        _record_step(ep,_stage_now)
        if ep%checkpoint_every==0 and ep>start_ep:
            hist={"loss":loss_arr,"l2_shape":l2s_arr,"l2_time":l2t_arr,"l2_epochs":l2_ep}
            _save_ckpt(ep,model,opt,sched,best_l2s,best_l2t,best_l2s<l2_target,hist)
            _rec.save()
        if ep%plot_every==0 and ep>start_ep and len(l2_ep)>0:
            _,pred=l2_time_and_pred()
            stage=2 if use_causal else 3
            _redraw(ep,l_val,l2s_arr[-1],l2t_arr[-1],pred,stage)
    plt.close(fig)
    if _last_loop_ep is not None:
        _record_step(_last_loop_ep,_stage_now,force=True)

    # ── final save ─────────────────────────────────────────────────────── #
    hist={"loss":loss_arr,"l2_shape":l2s_arr,"l2_time":l2t_arr,"l2_epochs":l2_ep}
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
    try:
        _ob=_net_out(); _Hb=np.asarray(_net_energy(_ob),dtype=np.float64)
        _rec.final.update(best_u=_ob[:,0].astype(np.float32),best_pr=_ob[:,1].astype(np.float32),
                          best_L=_ob[:,2].astype(np.float32),best_H=_Hb,
                          best_l2_shape=float(l2_shape()),best_l2_time=float(l2t_fin))
        _rec.save()
    except Exception as _e:
        print(f"  (plot-record final-best snapshot skipped: {_e})")
    np.savez(os.path.join(ckpt_dir,_ANPZ),t_eval=t_eval,pr0=pr0,ref_2pn_cons=sol_2pn.y.T,
             ref_target=ref2,pred_best_t=pred_t,phi_grid=phi_grid,
             dissipative=dissipative,L_ref_ang=L_ref,
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
    # ── PHYSICAL VALIDITY: energy monotonicity (dissipative only) ────────
    if dissipative:
        try:
            _Hn=_net_energy(); _dH=np.diff(_Hn)
            _frac_up=float((_dH>0).mean()); _dEtot=float(_Hn[-1]-_Hn[0])
            _rel=abs(_dEtot)/max(abs(_Hn[0]),1e-16)
            print(f"  energy check: ΔH={_dEtot:+.4e} ({100*_rel:.2f}% of |H0|), "
                  f"rising steps={100*_frac_up:.1f}%")
            if _dEtot<0 and _frac_up<0.05:
                print("  *** ENERGY MONOTONICALLY DECREASING — physical inspiral ✓ ***")
            elif _dEtot>=0:
                print("  ⚠ energy did NOT decrease — solution is not radiating "
                      "(likely still on the conservative orbit).")
            else:
                print(f"  ⚠ energy decreases overall but is non-monotonic "
                      f"({100*_frac_up:.1f}% rising steps); consider raising w_secular.")
        except Exception as _e:
            print(f"  (energy check unavailable: {_e})")
    if best_l2s<l2_target: print("  *** L2_shape TARGET REACHED ***")
    if best_l2t<l2_target: print("  *** L2_time  TARGET REACHED ***")
    print(f"  Saved to: {ckpt_dir}")
    # ── L-BFGS POLISH ────────────────────────────────────────────────
    if n_epochs > 0:
        print("  Running L-BFGS polish (up to 200 steps)...")
        lbfgs=torch.optim.LBFGS(model.parameters(),lr=0.1,max_iter=20,
                                history_size=50,line_search_fn='strong_wolfe')
        def _closure():
            lbfgs.zero_grad()
            phi_l=(torch.rand(n_colloc,1,device=dev)*phi_max).requires_grad_(True)
            R_u_l,R_pr_l,_,_,H_l,R_E_l=phi_residual(phi_l)
            l_e=((H_l.view(-1)-E_target_t)**2).mean()
            _we=0.0 if dissipative else w_energy   # no energy pin for inspiral
            _lsec=(w_secular*(R_E_l**2).mean()) if dissipative else 0.0
            loss_l=(R_u_l**2).mean()+(R_pr_l**2).mean()+_we*l_e+_lsec
            loss_l.backward(); return loss_l
        for _ in range(10):   # 10 × max_iter=20 = up to 200 function evals
            lbfgs.step(_closure)
        l2s_polish=l2_shape()
        print(f"  Post-polish L²={l2s_polish:.4e}  (was best={best_l2s:.4e})")
        try:
            _op=_net_out(); _Hp=np.asarray(_net_energy(_op),dtype=np.float64)
            _rec.final.update(polish_u=_op[:,0].astype(np.float32),
                              polish_pr=_op[:,1].astype(np.float32),
                              polish_L=_op[:,2].astype(np.float32),polish_H=_Hp,
                              polish_l2_shape=float(l2s_polish))
            _rec.save()
        except Exception as _e:
            print(f"  (plot-record polish snapshot skipped: {_e})")
        if l2s_polish < best_l2s:
            best_l2s=l2s_polish
            torch.save(model.state_dict(), os.path.join(ckpt_dir,_BEST))
            print(f"  L-BFGS improved best L² to {best_l2s:.4e}")

    return None, best_l2s, None
