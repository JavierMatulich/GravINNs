"""Adaptive PN-regime diagnosis and training dispatch.

``diagnose_pn_regime`` selects the physically meaningful transfer route from
the initial conditions. ``train_orbit_adaptive`` dispatches to the dissipative
training protocol when radiation reaction is requested.
"""
import numpy as np
import torch
from scipy.integrate import solve_ivp

from .. import constants as _const
from ..physics.hamiltonian import compute_hamiltonian_r
from .dissipative_trainer import train_dissipative

def diagnose_pn_regime(nu, p0_factor, r0_km, pr0_factor=0.0,
                       n_orbits=3, l2_target=1e-3,
                       gap_factor=10.0,      # require gap >= gap_factor*l2_target
                       apo_safe_mult=2.5,    # "bounded" = r stays < apo_safe_mult*r_apo
                       dissipative=False,    # route to the 2.5PN prescription
                       verbose=True):
    """
    Diagnose which post-Newtonian transfer regime applies to the given
    initial conditions. Returns a dict with at least:
        regime ∈ {'A','B','STOP'}
        warmstart_pn_order, target_pn_order   (for A and B)
        reason                                 (human-readable)
    plus orbital diagnostics (ecc, rmin, x_rel, bnd_1pn, gap_12, gap_23).
    """
    CSQ=_const.C_SQ; CQD=_const.C_QD
    q0=r0_km/200.0; p0t=p0_factor/np.sqrt(q0); pr0=pr0_factor
    y0=[q0,0.0,pr0,p0t]; L0=q0*p0t
    E_newt=0.5*(pr0**2+p0t**2)-1.0/q0
    out=dict(q0=q0,p0t=p0t,pr0=pr0,L0=L0,E_newt=E_newt)

    if E_newt>=0:
        out.update(regime='STOP', reason='Newtonian orbit already unbound (E>=0); not a bound orbit.')
        if verbose: _print_regime(out)
        return out

    # ── DISSIPATIVE ROUTE ────────────────────────────────────────────────
    # For the dissipative (radiation-reaction) case the physics is FIXED by
    # the standard 2.5PN prescription:  2PN conservative Hamiltonian  +
    # 2.5PN reactive force  F_RR.  It is NOT a 2PN→3PN conservative transfer.
    # The 2PN RK orbit is the warm-start/anchor; the PDE order is 2 and the
    # radiation reaction is added on top (dissipative=True in the trainer).
    # We therefore short-circuit the conservative A/B/STOP logic here so the
    # trainer is never mis-routed to target_pn_order=3.
    if dissipative:
        a=-0.5/E_newt; T=2*np.pi*a**1.5
        ecc=np.sqrt(max(0.0,1+2*E_newt*L0**2)); rmin=a*(1-ecc); rapo=a*(1+ecc)
        x_rel=(2.0/rmin-1.0/a)/CSQ
        out.update(a=a,T=T,ecc=ecc,rmin=rmin,rapo=rapo,x_rel=x_rel,
                   bnd_1pn=None, gap_12=float('nan'), gap_23=float('nan'),
                   regime='DISS', warmstart_pn_order='2pn', target_pn_order=2,
                   reason=('DISSIPATIVE: 2PN conservative Hamiltonian + 2.5PN '
                           'radiation-reaction force. Warm-start/anchor = 2PN RK '
                           'orbit; PDE order = 2 with F_RR ON.'))
        if verbose: _print_regime(out)
        return out

    a=-0.5/E_newt; T=2*np.pi*a**1.5
    ecc=np.sqrt(max(0.0,1+2*E_newt*L0**2)); rmin=a*(1-ecc); rapo=a*(1+ecc)
    x_rel=(2.0/rmin-1.0/a)/CSQ                     # ~ (v/c)^2 at periapsis
    out.update(a=a,T=T,ecc=ecc,rmin=rmin,rapo=rapo,x_rel=x_rel)

    def _rhs(pn):
        def f(t, Y):
            q = torch.tensor([[Y[0], Y[1]]], dtype=torch.float64, requires_grad=True)
            p = torch.tensor([[Y[2], Y[3]]], dtype=torch.float64, requires_grad=True)
            H = compute_hamiltonian_r(q, p, pn, nu, CSQ, CQD)
            dq = torch.autograd.grad(H.sum(), q, retain_graph=True)[0][0].numpy()
            dp = torch.autograd.grad(H.sum(), p)[0][0].numpy()
            return [dp[0], dp[1], -dq[0], -dq[1]]
        return f
    t_max=n_orbits*T; te=np.linspace(0,t_max,4000)
    def _integ(pn): return solve_ivp(_rhs(pn),[0,t_max],y0,t_eval=te,rtol=1e-9,atol=1e-11)
    def _phi(s): ph=np.unwrap(np.arctan2(s.y[1],s.y[0])); return ph-ph[0]
    def _bounded(s):
        r=np.hypot(s.y[0],s.y[1])
        return bool((r < rapo*apo_safe_mult).all() and _phi(s)[-1] > 2*np.pi*0.8)
    def _l2_4d(sa,sb):
        pa,pb=_phi(sa),_phi(sb)
        ra=np.hypot(sa.y[0],sa.y[1]); rb=np.hypot(sb.y[0],sb.y[1])
        pra=(sa.y[0]*sa.y[2]+sa.y[1]*sa.y[3])/ra; prb=(sb.y[0]*sb.y[2]+sb.y[1]*sb.y[3])/rb
        pg=np.linspace(0,min(pa[-1],pb[-1]),1500)
        ua=np.interp(pg,pa,1/ra); ub=np.interp(pg,pb,1/rb)
        pai=np.interp(pg,pa,pra); pbi=np.interp(pg,pb,prb)
        qxa=np.cos(pg)/ua; qya=np.sin(pg)/ua; qxb=np.cos(pg)/ub; qyb=np.sin(pg)/ub
        pxa=pai*np.cos(pg)-(L0*ua)*np.sin(pg); pya=pai*np.sin(pg)+(L0*ua)*np.cos(pg)
        pxb=pbi*np.cos(pg)-(L0*ub)*np.sin(pg); pyb=pbi*np.sin(pg)+(L0*ub)*np.cos(pg)
        refn=np.linalg.norm(np.stack([qxb,qyb,pxb,pyb]))+1e-16
        return float(np.linalg.norm(np.stack([qxa-qxb,qya-qyb,pxa-pxb,pya-pyb]))/refn)

    s1=_integ(1); s2=_integ(2); s3=_integ(3)
    bnd1=_bounded(s1)
    gap_12=_l2_4d(s1,s2) if bnd1 else float('nan')
    gap_23=_l2_4d(s2,s3)
    out.update(bnd_1pn=bnd1, gap_12=gap_12, gap_23=gap_23)
    gap_min=gap_factor*l2_target

    if bnd1:
        # CASE A: 1PN is a valid bound orbit → use it as warm-start/anchor for 2PN
        out.update(regime='A', warmstart_pn_order='1pn', target_pn_order=2,
                   reason=f"1PN orbit is BOUNDED → transfer 1PN→2PN (gap_12={gap_12:.2e}).")
        if gap_12 < gap_min:
            out['note']=(f"NOTE: 1PN-2PN gap ({gap_12:.2e}) < {gap_min:.0e}; the 2PN "
                         f"correction is small — transfer is easy but may be near-trivial.")
    else:
        if gap_23 >= gap_min:
            # CASE B: 1PN unusable, but 3PN materially differs from 2PN
            out.update(regime='B', warmstart_pn_order='2pn', target_pn_order=3,
                       reason=(f"1PN orbit UNBOUNDED, and 2PN-3PN gap is significant "
                               f"({gap_23:.2e} ≥ {gap_min:.0e}) → transfer 2PN→3PN."))
        else:
            # STOP: neither transfer is meaningful
            out.update(regime='STOP',
                       reason=(f"1PN orbit UNBOUNDED *and* 2PN-3PN gap negligible "
                               f"({gap_23:.2e} < {gap_min:.0e}). No meaningful PN "
                               f"transfer is possible for these initial conditions."))
    if verbose: _print_regime(out)
    return out


def _print_regime(d):
    print("="*70)
    print("  PN-REGIME DIAGNOSIS")
    print("="*70)
    if 'ecc' in d:
        print(f"   q0={d['q0']:.4f}  ecc={d['ecc']:.3f}  r_min={d['rmin']:.4f}  "
              f"r_apo={d['rapo']:.4f}")
        print(f"   relativistic parameter x = v^2/c^2|_peri = {d['x_rel']:.4e}")
        if d.get('regime')=='DISS':
            print(f"   [dissipative] radiation reaction ON — 2PN H + 2.5PN F_RR")
        else:
            print(f"   1PN bounded: {d.get('bnd_1pn')}")
            if d.get('bnd_1pn'):
                print(f"   gap_12 (1PN→2PN, 4D L2) = {d.get('gap_12'):.4e}")
            print(f"   gap_23 (2PN→3PN, 4D L2) = {d.get('gap_23'):.4e}")
    print(f"   REGIME: {d['regime']}")
    print(f"   {d['reason']}")
    if 'note' in d: print(f"   {d['note']}")
    print("="*70)


def train_orbit_adaptive(nu, p0_factor, r0_km, pr0_factor=0.0,
                         l2_target=1e-3, gap_factor=10.0, n_orbits=3,
                         dissipative=False,        # ← NEW: explicit, forwarded
                         dev=None, **train_kwargs):
    """
    Diagnose the PN regime from the initial conditions, then train the
    appropriate transfer (1PN→2PN or 2PN→3PN). Raises/returns cleanly on STOP.

    Extra **train_kwargs are forwarded verbatim to train_dissipative,
    EXCEPT warmstart_pn_order / target_pn_order / transfer_from_2pn, which
    are set by the diagnosis (an explicit value in train_kwargs overrides).
    """
    diag=diagnose_pn_regime(nu,p0_factor,r0_km,pr0_factor,
                            n_orbits=n_orbits,l2_target=l2_target,
                            gap_factor=gap_factor,dissipative=dissipative,
                            verbose=True)
    if diag['regime']=='STOP':
        print("\n  *** TRAINING ABORTED ***")
        print("  The initial conditions admit no meaningful post-Newtonian transfer.")
        print("  (1PN is unbound and the 2PN→3PN difference is below the target.)")
        return diag, None, None

    # Regime-appropriate settings (caller can override explicitly)
    ws  = train_kwargs.pop('warmstart_pn_order', diag['warmstart_pn_order'])
    tgt = train_kwargs.pop('target_pn_order',    diag['target_pn_order'])
    train_kwargs.pop('transfer_from_2pn', None)   # always True for transfer

    print(f"\n  → Dispatching: warmstart='{ws}', target_pn_order={tgt} "
          f"(regime {diag['regime']})\n")

    hist, best_l2_shape, best_l2_time = train_dissipative(
        nu=nu, p0_factor=p0_factor, r0_km=r0_km, pr0_factor=pr0_factor,
        dissipative=dissipative,          # ← forward radiation-reaction flag
        transfer_from_2pn=train_kwargs.pop('transfer_from_2pn', True),
        warmstart_pn_order=ws,
        target_pn_order=tgt,
        l2_target=l2_target,
        n_orbits=n_orbits,
        dev=dev,
        **train_kwargs)
    return diag, best_l2_shape, best_l2_time
