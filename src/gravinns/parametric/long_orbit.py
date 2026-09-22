"""Long-time orbit generation from a parametric model.

* ``generate_long_orbit``          chained restarts with energy projection (cell 2)
* ``generate_long_orbit_section``  Poincare-section restarts + analytic
                                   action-angle phase clock (cell 1)

Origin: GitHub_parametric.ipynb, cells 1-2.
"""
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    from IPython.display import display as _ipy_display
except ImportError:  # IPython is optional outside notebooks.
    def _ipy_display(_obj):
        return None
from scipy.optimize import brentq, minimize_scalar

from .orbits import C_QD, C_SQ, _GRADS, integrate_orbit
from .training import _energy_target, _measure_l2
from ..physics.hamiltonian import compute_hamiltonian_polar_numpy



# ── Unsupervised long-orbit generation by segment chaining ────────────────
def _project_theta_to_energy(theta_new, E0, pn_order, tol=1e-13, max_iter=60):
    """Project a restart theta onto the EXACT energy shell E0, at fixed L.

    The chain's restart map conserves L identically (L' = r*.p_t* = L), but NOT
    the energy: the network's (u, pr) at the restart point carry a small error,
    so E(theta') drifts from E0 and the chain hops onto a NEIGHBOURING member of
    the orbit family. Over many segments that hop is the dominant accumulated
    error (measured: |dE|/E ~ 1e-4 per junction, end-to-end L2 ~ 1e-2 over 116
    revolutions, with a saturating envelope characteristic of a slow phase
    offset rather than a geometric divergence).

    This restores the second invariant. Keeping r* (the restart geometry) and
    L (already exact), the energy constraint

        H_pn(r*, pr, L; nu) = E0

    is ONE scalar equation in ONE unknown, pr. Solving it makes the restart
    state satisfy BOTH conserved quantities exactly, so the only surviving
    error is in r* itself. No reference orbit is involved -- H is the analytic
    Hamiltonian and E0 is fixed by the chain's own initial condition.

    Implementation: bisection on f(pr) = H(r*,pr,L) - E0. H is monotone in
    |pr| at fixed (r, L) (the kinetic term dominates), so f is monotone on
    each branch and bisection is unconditionally robust -- unlike Newton,
    which is ill-conditioned near a turning point (pr -> 0).

    The sign branch of pr is PRESERVED (the physical in/outbound choice made by
    the restart-window selection); only its magnitude is corrected.

    Returns (theta_projected, info_dict). If no root exists on the branch --
    i.e. the requested energy is unreachable at radius r* with this L, which
    happens if the network's r* is badly wrong -- the input theta is returned
    unchanged with info["ok"]=False, and the caller should treat that candidate
    as invalid.
    """
    import numpy as _np
    p0f, r0km, pr0f, nu = (float(x) for x in theta_new)
    q0 = r0km/200.0                      # restart radius r*
    p0t = p0f/_np.sqrt(q0)               # tangential momentum (fixes L)
    L = q0*p0t

    def _H_of_pr(pr):
        th = torch.tensor([[p0t*_np.sqrt(q0), r0km, pr, nu]],
                          dtype=torch.float64)
        return float(_energy_target(th, pn_order, C_SQ, C_QD))

    f = lambda pr: _H_of_pr(pr) - E0
    f0 = f(pr0f)
    if abs(f0) < tol:
        return theta_new, dict(ok=True, iters=0, dE_before=f0, dE_after=f0,
                               dpr=0.0)

    sgn = 1.0 if pr0f >= 0 else -1.0
    # bracket on the branch sgn*[0, pr_max]: |pr| grows -> H grows
    a = 0.0
    fa = f(0.0)
    b = max(abs(pr0f), 0.1)
    fb = f(sgn*b)
    grow = 0
    while fa*fb > 0 and grow < 60:
        b *= 1.5
        fb = f(sgn*b)
        grow += 1
    if fa*fb > 0:
        # E0 unreachable at this radius on this branch
        return theta_new, dict(ok=False, iters=grow, dE_before=f0,
                               dE_after=f0, dpr=0.0)

    for it in range(max_iter):
        m = 0.5*(a+b)
        fm = f(sgn*m)
        if abs(fm) < tol or (b-a) < 1e-15:
            break
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    pr_new = sgn*0.5*(a+b)
    th_p = (theta_new[0], theta_new[1], float(pr_new), theta_new[3])
    return th_p, dict(ok=True, iters=it+1, dE_before=f0,
                      dE_after=f(pr_new), dpr=float(pr_new-pr0f))


def generate_long_orbit(model, param_ranges, theta0,
                        n_segments=5,
                        n_pts=2000,
                        restart_window=(0.40, 0.95),
                        n_candidates=60,
                        overlap_frac=0.20,
                        tol=1e-3,
                        energy_weight=1.0,
                        error_net=None, error_net_weight=1.0,
                        full_rk_check=False,
                        project_energy=True,
                        pde_pn_order=2, n_orbits=3,
                        validate_rk=False,
                        device="cpu", verbose=True, plot=True):
    """Generate an arbitrarily long orbit by CHAINING parametric-PINN segments.

    The dynamics are autonomous, so any point of an orbit is a valid initial
    condition for a fresh segment. Segment k is evaluated on the model's own
    angle domain [0, model.phi_max]; a restart point phi* is then chosen on it,
    the state there is mapped back to a NEW theta

        r* = 1/u(phi*),  p_t* = L*u(phi*),  ->
        r0_km'      = 200 * r*
        p0_factor'  = p_t* * sqrt(r*)
        pr0_factor' = pr(phi*)
        nu'         = nu            (conserved)

    and segment k+1 continues from there. The mapping was validated with pure
    RK restarts (overlap L2 ~ 6e-6, i.e. exact to integrator tolerance).

    EVERYTHING IS UNSUPERVISED -- no RK reference is used to drive any choice:

    * VALIDITY: a candidate phi* is admissible only if its mapped theta lies
      INSIDE `param_ranges` (the trained box). For a typical eccentric orbit
      this admits ~one window per revolution (the outgoing mid-radius arc);
      near periapsis the mapped p0/pr0 leave the box and are rejected.

    * SELECTION (the L2 < tol criterion, without ground truth): for each
      admissible candidate the NEW segment is evaluated over an overlap window
      [0, W] and compared with the CURRENT segment over [phi*, phi*+W]. Both
      describe the same physical arc, so their relative L2 mismatch is a
      self-consistency error estimate. The candidate with the smallest
      mismatch is chosen; if none is below `tol` the chain stops with a
      warning rather than silently degrading.

    * ENERGY DRIFT (second RK-free signal, needs NO calibration): the energy
      E(theta) = H(IC(theta)) is an ANALYTIC function of theta -- closed-form
      Hamiltonian, no network, no integrator. Since L is conserved exactly by
      the restart mapping, an exact chain would keep E(theta_k) = E(theta_0)
      for every segment; any drift is caused purely by network error in the
      restart state. Unlike the overlap score it does NOT suffer shared-bias
      cancellation (E is not a network output), so it is an ABSOLUTE error
      measure at each join. Verified on a test chain: |dE|/|E0| grew
      0 -> 3.0e-2 -> 5.5e-2 -> 7.3e-2 while the true per-segment L2 grew
      2.1e-2 -> 4.0e-2 -- same order, same trend. The selection score is
          score = overlap_L2 + energy_weight * |E(theta_new)-E(theta_0)|/|E0|
      and `tol` applies to this combined score. Set energy_weight=0 to select
      on overlap alone. Note the honest caveat: overlap
      consistency measures CONTINUATION error, not absolute truth -- a shared
      bias of the network at both thetas cancels in the comparison. It is a
      lower bound on (and in practice tracks) the true per-segment error.

    * The restart state itself is read from the PINN's own outputs (u, pr),
      not from any reference orbit.

    Segment errors compound along the chain (roughly linearly in the number
    of segments), so the tail of a very long chain is less accurate than the
    head. Set validate_rk=True to integrate one RK orbit per segment and
    report TRUE per-segment L2 -- for diagnostics only; nothing in the chain
    uses it.

    Parameters
    ----------
    theta0          : (p0_factor, r0_km, pr0_factor, nu) of the first segment;
                      must lie inside param_ranges.
    n_segments      : how many segments to chain (total ~ n_segments*n_orbits
                      revolutions... in phi: n_segments * phi_max, minus the
                      overlap consumed by each restart).
    restart_window  : (lo, hi) fractions of the segment span in which to search
                      for phi*. Keep hi < 1-overlap_frac so the overlap window
                      fits inside the segment.
    n_candidates    : how many phi* candidates to score per segment.
    overlap_frac    : overlap window W = overlap_frac * phi_max used for the
                      consistency test.
    tol             : unsupervised acceptance threshold on the overlap L2.

    full_rk_check : integrate ONE long RK orbit from theta0, spanning the whole
      chain, and report the END-TO-END L2 of the stitched orbit against it.
      This is the only honest accuracy measure of the chain: the per-segment
      `rk_l2` values (validate_rk) each re-seed their reference from the
      ALREADY-DRIFTED restart state, so they are blind to accumulated error by
      construction. Adds one integration (seconds with the symbolic RHS) and is
      VALIDATION ONLY -- nothing in the chain uses it. Results land in the
      returned dict under "full_rk".

    Returns
    -------
    dict with:
      phi_global, u, pr : the stitched orbit (global azimuth, 1/r, radial mom.)
      full_rk           : (if full_rk_check) dict with l2, l2_u, l2_pr,
                          phi_span, n_rev, and the reference arrays
      x, y              : cartesian trajectory
      segments          : per-segment dicts (theta, phi_offset, phi_star,
                          overlap_l2, rk_l2 [if validate_rk])
      ok                : True if all requested segments passed tol
    """
    import numpy as _np
    keys = ("p0_factor", "r0_km", "pr0_factor", "nu")
    for k in keys:
        if k not in param_ranges:
            raise KeyError(f"param_ranges missing '{k}'")
    th = tuple(float(x) for x in theta0)
    for k, v in zip(keys, th):
        lo, hi = param_ranges[k]
        if not (lo - 1e-12 <= v <= hi + 1e-12):
            raise ValueError(f"theta0[{k}]={v} outside param_ranges[{k}]={param_ranges[k]}")

    PM = float(model.phi_max)
    W = overlap_frac * PM
    if restart_window[1]*PM + W > PM + 1e-9:
        restart_window = (restart_window[0], 1.0 - overlap_frac - 0.01)

    def _eval_seg(theta, phis):
        P = torch.tensor(phis, dtype=torch.float32, device=device).view(-1, 1)
        T = torch.tensor([list(theta)], dtype=torch.float32,
                         device=device).repeat(len(phis), 1)
        with torch.no_grad():
            out = model(P, T).cpu().numpy()
        return out[:, 0], out[:, 1], float(out[0, 2])   # u, pr, L (constant)

    phis = _np.linspace(0.0, PM, n_pts)
    dphi = phis[1] - phis[0]

    def _E(theta):
        # analytic energy of the IC at theta (closed-form Hamiltonian; RK-free)
        return float(_energy_target(
            torch.tensor([list(theta)], dtype=torch.float32, device=device),
            pde_pn_order, C_SQ, C_QD))

    E0 = _E(th)
    if project_energy and verbose:
        print(f"  energy projection ON: every restart state is solved onto "
              f"H = E0 = {E0:.6f} at fixed L (analytic; no reference orbit).")

    all_phi, all_u, all_pr = [], [], []
    segments = []
    offset = 0.0
    ok = True

    for seg in range(n_segments):
        u_s, pr_s, L_s = _eval_seg(th, phis)

        # candidate restart points
        i_lo = int(restart_window[0]*(n_pts-1))
        i_hi = int(restart_window[1]*(n_pts-1))
        cand_idx = _np.unique(_np.linspace(i_lo, i_hi, n_candidates).astype(int))

        best = None
        n_valid = 0
        for i in cand_idx:
            r_star = 1.0/u_s[i]
            th_new = (float(L_s*u_s[i]*_np.sqrt(r_star)),   # p0_factor'
                      float(200.0*r_star),                  # r0_km'
                      float(pr_s[i]),                       # pr0_factor'
                      th[3])                                # nu
            if not all(param_ranges[k][0] - 1e-12 <= v <= param_ranges[k][1] + 1e-12
                       for k, v in zip(keys, th_new)):
                continue
            # Project onto the exact energy shell E0 at fixed L. The restart
            # map conserves L identically but not E; restoring E is what stops
            # the chain drifting onto neighbouring members of the orbit family.
            # Done BEFORE the in-box recheck and scoring, so the candidate is
            # judged in the state it will actually be used in.
            if project_energy:
                th_p, _pinfo = _project_theta_to_energy(th_new, E0, pde_pn_order)
                if not _pinfo["ok"]:
                    continue          # E0 unreachable here: bad restart radius
                th_new = th_p
                if not all(param_ranges[k][0] - 1e-12 <= v <= param_ranges[k][1] + 1e-12
                           for k, v in zip(keys, th_new)):
                    continue          # projection pushed it out of the box
            n_valid += 1
            # unsupervised overlap-consistency score
            n_w = max(8, int(W/dphi))
            if i + n_w >= n_pts:
                continue
            d = phis[:n_w]                                  # new-segment phi
            u_n, pr_n, _ = _eval_seg(th_new, d)
            u_o = u_s[i:i+n_w]; pr_o = pr_s[i:i+n_w]
            l2 = float(_np.sqrt(
                (_np.linalg.norm(u_n-u_o)**2 + _np.linalg.norm(pr_n-pr_o)**2)
                / (_np.linalg.norm(u_o)**2 + _np.linalg.norm(pr_o)**2 + 1e-30)))
            # RK-free absolute signal: analytic energy drift of the mapped IC
            # relative to the chain's initial energy (exact chain => zero).
            e_drift = abs(_E(th_new) - E0)/(abs(E0) + 1e-30)
            score = l2 + energy_weight*e_drift
            # optional learned error filter: a small net trained OFFLINE on
            # (theta, true L2) pairs predicts the absolute error of the NEXT
            # segment at th_new; steers restarts toward low-error regions of
            # the box. The chain itself remains RK-free at generation time.
            pred_l2 = None
            if error_net is not None:
                pred_l2 = error_net.predict_l2(th_new)
                score = score + error_net_weight*pred_l2
            if best is None or score < best[0]:
                best = (score, i, th_new, l2, e_drift, pred_l2)

        is_last = (seg == n_segments - 1)
        if best is None and not is_last:
            print(f"  [seg {seg}] NO admissible restart candidate maps inside the "
                  f"trained box (of {len(cand_idx)} scanned, {n_valid} were "
                  f"in-box but none fit the overlap window). Chain stops here.")
            ok = False

        # commit this segment up to the restart point (or fully, if last/stopped)
        cut = n_pts if (is_last or best is None) else best[1]
        all_phi.append(offset + phis[:cut])
        all_u.append(u_s[:cut]); all_pr.append(pr_s[:cut])

        rec = dict(theta=th, phi_offset=offset, L=L_s)
        if validate_rk:
            rec["rk_l2"] = _measure_l2(model, th, pde_pn_order, n_orbits,
                                       n_pts, PM, device)
        if best is not None and not is_last:
            rec["phi_star"] = float(phis[best[1]])
            rec["score"] = best[0]
            rec["overlap_l2"] = best[3]
            rec["energy_drift"] = best[4]
            if best[5] is not None:
                rec["pred_l2"] = best[5]
            if verbose:
                msg = (f"  [seg {seg}] theta=({th[0]:.3f},{th[1]:.1f},"
                       f"{th[2]:.3f},{th[3]:.3f})  restart phi*={rec['phi_star']:.2f} "
                       f"overlap={best[3]:.2e} dE/E={best[4]:.2e} "
                       + (f"predL2={best[5]:.2e} " if best[5] is not None else "")
                       + f"score={best[0]:.2e}"
                       + (f"  true-L2={rec['rk_l2']:.2e}" if validate_rk else ""))
                print(msg)
            if best[0] > tol:
                print(f"  [seg {seg}] score {best[0]:.2e} (overlap {best[3]:.2e} "
                      f"+ {energy_weight}*dE {best[4]:.2e}) > tol {tol:.0e}: "
                      f"no restart met the unsupervised criterion. Chain stops.")
                ok = False
                segments.append(rec)
                break
            offset += phis[best[1]]
            th = best[2]
        elif verbose:
            print(f"  [seg {seg}] theta=({th[0]:.3f},{th[1]:.1f},"
                  f"{th[2]:.3f},{th[3]:.3f})  (final segment)"
                  + (f"  true-L2={rec['rk_l2']:.2e}" if validate_rk else ""))
        segments.append(rec)
        if best is None and not is_last:
            break

    phi_g = _np.concatenate(all_phi)
    u_g = _np.concatenate(all_u)
    pr_g = _np.concatenate(all_pr)
    x = _np.cos(phi_g)/u_g; y = _np.sin(phi_g)/u_g

    if verbose:
        print(f"  total span: {phi_g[-1]:.1f} rad = "
              f"{phi_g[-1]/(2*_np.pi):.1f} revolutions over "
              f"{len(segments)} segment(s)")

    # ── END-TO-END check: one long RK orbit from theta0 over the chain's span
    full_rk = None
    if full_rk_check:
        th0 = tuple(float(x) for x in theta0)
        span_needed = float(phi_g[-1])
        # integrate_orbit takes a number of RADIAL periods, not a phi target.
        # One period advances phi by ~2*pi (plus the PN periapsis advance), so
        # estimate the count, then overshoot 15% and truncate at span_needed.
        n_orb_est = int(_np.ceil(span_needed/(2*_np.pi)*1.15)) + 1
        if verbose:
            print(f"  [full_rk_check] integrating one RK orbit from theta0 "
                  f"over {span_needed:.1f} rad (~{n_orb_est} radial periods)...")
        ref = integrate_orbit(*th0, pde_pn_order, n_orb_est,
                              max(4000, 4*len(phi_g)))
        if ref["phi"][-1] < span_needed - 1e-6:
            # the estimate fell short (strong precession): retry once, bigger
            grow = 1.3*span_needed/max(ref["phi"][-1], 1e-9)
            n_orb_est = int(_np.ceil(n_orb_est*grow)) + 1
            ref = integrate_orbit(*th0, pde_pn_order, n_orb_est,
                                  max(4000, 4*len(phi_g)))
        m = phi_g <= min(span_needed, ref["phi"][-1])
        u_ref = _np.interp(phi_g[m], ref["phi"], ref["u"])
        pr_ref = _np.interp(phi_g[m], ref["phi"], ref["pr"])
        du = _np.linalg.norm(u_g[m]-u_ref); dp = _np.linalg.norm(pr_g[m]-pr_ref)
        nu_ = _np.linalg.norm(u_ref); np_ = _np.linalg.norm(pr_ref)
        full_rk = dict(
            l2=float(_np.sqrt((du**2+dp**2)/(nu_**2+np_**2+1e-30))),
            l2_u=float(du/(nu_+1e-30)), l2_pr=float(dp/(np_+1e-30)),
            phi_span=float(phi_g[m][-1]), n_rev=float(phi_g[m][-1]/(2*_np.pi)),
            phi=ref["phi"], u=ref["u"], pr=ref["pr"], mask=m)
        if verbose:
            print(f"  [full_rk_check] END-TO-END L2 vs a single RK orbit from "
                  f"theta0 over {full_rk['n_rev']:.1f} rev: "
                  f"{full_rk['l2']:.3e}  (u {full_rk['l2_u']:.2e}, "
                  f"pr {full_rk['l2_pr']:.2e})")
            per_seg = [s.get("rk_l2") for s in segments if s.get("rk_l2")]
            if per_seg:
                print(f"                   for comparison, per-segment rk_l2 "
                      f"median {_np.median(per_seg):.2e} -- these CANNOT see "
                      f"accumulated drift")

    if plot:
        _fig, _axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
        _axs[0].plot(x, y, lw=0.7, color="crimson")
        for s in segments[1:]:
            po = s["phi_offset"]
            j = int(_np.searchsorted(phi_g, po))
            if j < len(x):
                _axs[0].plot(x[j], y[j], "o", ms=5, mfc="none", mec="navy")
        _axs[0].plot(0, 0, "k*", ms=9)
        _axs[0].set_aspect("equal")
        _axs[0].set_title(f"chained PINN orbit: {len(segments)} segments, "
                          f"{phi_g[-1]/(2*_np.pi):.1f} rev\n"
                          f"(circles = restart points)", fontsize=9)
        if full_rk is not None:
            _axs[0].plot(_np.cos(full_rk["phi"])/full_rk["u"],
                         _np.sin(full_rk["phi"])/full_rk["u"],
                         lw=0.6, color="0.55", zorder=0, label="RK (from theta0)")
            _axs[0].legend(fontsize=7, loc="upper right")
        _axs[1].plot(phi_g, 1.0/u_g, lw=0.7, label="PINN chain")
        if full_rk is not None:
            _axs[1].plot(full_rk["phi"], 1.0/full_rk["u"], lw=0.6,
                         color="0.55", zorder=0, label="RK reference")
            _axs[1].legend(fontsize=7)
        for s in segments[1:]:
            _axs[1].axvline(s["phi_offset"], color="0.7", ls=":", lw=0.8)
        _axs[1].set_xlabel("global phi [rad]"); _axs[1].set_ylabel("r")
        _axs[1].set_title("radius vs global azimuth (dotted = segment joins)",
                          fontsize=9)
        _fig.tight_layout()
        try:
            _ipy_display(_fig)
        except Exception:
            pass

    return dict(phi_global=phi_g, u=u_g, pr=pr_g, x=x, y=y,
                segments=segments, ok=ok, full_rk=full_rk)


# ── shared PN Hamiltonian in polar form ─────────────────────────────────
def _H_polar(r, pr, L, nu, pn_order):
    """Polar wrapper around the canonical Hamiltonian implementation."""
    return compute_hamiltonian_polar_numpy(
        r, pr, L, nu, pn_order, C_SQ, C_QD
    )


def _theta_to_polar(theta):
    p0f, r0km, pr0f, nu = (float(x) for x in theta)
    r0 = r0km/200.0
    L = r0*(p0f/math.sqrt(r0))
    return r0, pr0f, L, nu


def _polar_to_theta(r, pr, L, nu):
    return (L/math.sqrt(r), 200.0*r, float(pr), float(nu))


# ── invariant-curve geometry and the action-angle phase clock ─────────────
class OrbitClock:
    """Everything the chain needs about the exact orbit, from (E0, L0) only.

    psi in [0, Phi_r) is the azimuth measured from the last periapsis passage.
    It advances EXACTLY one-for-one with phi, and wraps every Phi_r.
    """
    def __init__(self, theta0, pn_order=2, n_quad=128):
        self.pn = int(pn_order); self.n_quad = int(n_quad)
        r0, pr0, L, nu = _theta_to_polar(theta0)
        self.L, self.nu = L, nu
        self.E = float(_H_polar(r0, pr0, L, nu, self.pn))
        if self.E >= 0:
            raise ValueError("unbound orbit (E >= 0)")
        self.r_p, self.r_a = self._turning_points()
        self.Phi_r = 2.0*self.psi_outbound(self.r_a)
        self.psi0 = self.psi_of_state(r0, pr0)

    def _Veff(self, r):
        return float(_H_polar(r, 0.0, self.L, self.nu, self.pn))

    def _turning_points(self):
        E, L = self.E, self.L
        a = -0.5/E; e = math.sqrt(max(1.0 + 2.0*E*L*L, 0.0))
        rp0, ra0 = a*(1-e), a*(1+e)
        rc = minimize_scalar(self._Veff, bounds=(0.5*rp0, 1.5*ra0),
                             method="bounded", options=dict(xatol=1e-13)).x
        f = lambda r: self._Veff(r) - E
        if f(rc) >= 0:
            raise ValueError("E0 below effective-potential minimum (circular/invalid)")
        hi = max(1.5*ra0, 1.01*rc)
        while f(hi) < 0: hi *= 1.5
        lo = 0.5*rp0
        while f(lo) < 0 and lo > 1e-3: lo *= 0.8
        ra = brentq(f, rc, hi, xtol=1e-15, rtol=1e-15)
        rp = brentq(f, lo, rc, xtol=1e-15, rtol=1e-15)
        return rp, ra

    def pr_on_shell(self, r, sign=+1.0):
        f = lambda p: float(_H_polar(r, p, self.L, self.nu, self.pn)) - self.E
        if f(0.0) >= 0:
            return 0.0
        hi = 0.5
        while f(hi) < 0: hi *= 2.0
        return math.copysign(brentq(f, 0.0, hi, xtol=1e-16, rtol=1e-15), sign)

    def _dphi_dr(self, r):
        pr = self.pr_on_shell(r, +1.0)
        g = _GRADS[self.pn](r, 0.0, pr, self.L/r, self.nu, C_SQ, C_QD)
        return (g[3]/r)/g[2]              # (dH/dL)/(dH/dpr)

    def psi_outbound(self, r):
        """Azimuth swept from periapsis to radius r on the OUTBOUND branch.
        Chebyshev-type substitution removes the 1/sqrt endpoint singularity,
        Gauss-Legendre is then spectrally convergent."""
        rm = 0.5*(self.r_a + self.r_p); h = 0.5*(self.r_a - self.r_p)
        chi_to = math.acos(min(1.0, max(-1.0, (rm - r)/h)))
        x, w = np.polynomial.legendre.leggauss(self.n_quad)
        chi = 0.5*chi_to*(x + 1.0); w = 0.5*chi_to*w
        return float(sum(wi*self._dphi_dr(rm - h*math.cos(ci))*h*math.sin(ci)
                         for ci, wi in zip(chi, w)))

    def psi_of_state(self, r, pr):
        r = min(max(r, self.r_p), self.r_a)
        out = self.psi_outbound(r)
        if pr > 0:  return out
        if pr < 0:  return self.Phi_r - out
        return 0.0 if abs(r - self.r_p) < abs(r - self.r_a) else 0.5*self.Phi_r

    def section_state(self, frac=1.0, sign=+1.0):
        """Point of the invariant curve at r = r_p + frac*(r_a - r_p).
        frac=1 -> apoapsis (pr=0). Returns (r, pr, psi)."""
        r = self.r_p + frac*(self.r_a - self.r_p)
        pr = 0.0 if frac >= 1.0 else self.pr_on_shell(r, sign)
        return r, pr, self.psi_of_state(r, pr)


# ── the generator ──────────────────────────────────────────────────────────
def generate_long_orbit_section(model, param_ranges, theta0, n_rev=100.0,
                                pn_order=2,
                                section_fracs=(1.0, 0.9, 0.8, 0.7),
                                section_signs=(+1.0, -1.0),
                                pts_per_rad=60,
                                net_eval=None,
                                full_rk_check=False,
                                device="cpu", verbose=True):
    """Arbitrarily long orbit from theta0 with bounded (non-accumulating) error.

    Section selection is UNSUPERVISED: every candidate section state is exact
    by construction, so candidates only differ in how well the NETWORK
    reproduces one period from them. Each in-box candidate is scored on
        periodicity defect  : network state at local phi=Phi_r vs its own IC
        energy defect       : mean |H(1/u, pr, L) - E0|/|E0| along the period
    (both use only network outputs + the analytic Hamiltonian) and the best is
    kept. The same arc is then replayed every Phi_r.

    net_eval(theta, phis) -> (u, pr) lets you swap in another evaluator;
    by default it calls the parametric PINN.
    """
    keys = ("p0_factor", "r0_km", "pr0_factor", "nu")
    in_box = lambda th: all(param_ranges[k][0] - 1e-12 <= v <= param_ranges[k][1] + 1e-12
                            for k, v in zip(keys, th))
    PM = float(model.phi_max) if model is not None else np.inf

    if net_eval is None:
        import torch
        def net_eval(theta, phis):
            out = []
            for s in range(0, len(phis), 200_000):
                P = torch.tensor(phis[s:s+200_000], dtype=torch.float32,
                                 device=device).view(-1, 1)
                T = torch.tensor([list(theta)], dtype=torch.float32,
                                 device=device).expand(len(P), 4)
                with torch.no_grad():
                    out.append(model(P, T)[:, :2].cpu().numpy())
            o = np.concatenate(out)
            return o[:, 0].astype(np.float64), o[:, 1].astype(np.float64)

    clk = OrbitClock(theta0, pn_order)
    L, nu, E0, Phi = clk.L, clk.nu, clk.E, clk.Phi_r
    if Phi > PM:
        raise ValueError(f"Phi_r={Phi:.3f} exceeds model.phi_max={PM:.3f}")
    if verbose:
        print(f"  invariants : E0={E0:.12f}  L0={L:.12f}  nu={nu}")
        print(f"  turning pts: r_p={clk.r_p:.6f} ({200*clk.r_p:.1f} km)  "
              f"r_a={clk.r_a:.6f} ({200*clk.r_a:.1f} km)")
        print(f"  Phi_r      : {Phi:.12f} rad  (periapsis advance "
              f"{Phi-2*math.pi:.6f} rad / radial period)")

    # ── score candidate sections (unsupervised) ─────────────────────────────
    n_chk = 2000
    loc_chk = np.linspace(0.0, Phi, n_chk)
    best = None
    for fr in section_fracs:
        for sg in (section_signs if fr < 1.0 else (+1.0,)):
            r_s, pr_s, psi_s = clk.section_state(fr, sg)
            th_s = _polar_to_theta(r_s, pr_s, L, nu)
            if not in_box(th_s):
                continue
            u, pr = net_eval(th_s, loc_chk)
            per = math.sqrt(((u[-1] - u[0])/u[0])**2 + (pr[-1] - pr[0])**2)
            dE = float(np.mean(np.abs(_H_polar(1.0/u, pr, L, nu, pn_order) - E0))/abs(E0))
            score = per + dE
            if verbose:
                print(f"   section r={200*r_s:7.2f} km pr={pr_s:+.4f}: "
                      f"periodicity {per:.2e}  energy {dE:.2e}")
            if best is None or score < best[0]:
                best = (score, th_s, psi_s, per, dE)
    if best is None:
        raise RuntimeError("no candidate section state maps inside param_ranges; "
                           "add fractions to section_fracs")
    _, th_s, psi_s, per, dE = best

    # ── assemble ──────────────────────────────────────────────────────────
    phi_total = 2*math.pi*float(n_rev)
    phi_s0 = (psi_s - clk.psi0) % Phi           # azimuth of the first crossing
    phi_g = np.linspace(0.0, phi_total, int(pts_per_rad*phi_total) + 1)
    u_g = np.empty_like(phi_g); pr_g = np.empty_like(phi_g)
    pre = phi_g < phi_s0
    if pre.any():                                 # arc before the first crossing
        u_g[pre], pr_g[pre] = net_eval(tuple(theta0), phi_g[pre])
    loc = (phi_g[~pre] - phi_s0) % Phi
    u_g[~pre], pr_g[~pre] = net_eval(th_s, loc)
    x = np.cos(phi_g)/u_g; y = np.sin(phi_g)/u_g
    if verbose:
        print(f"  chosen section theta_s={tuple(round(v, 6) for v in th_s)}  "
              f"first crossing at phi={phi_s0:.6f}")
        print(f"  {n_rev} rev = {phi_total/Phi:.1f} radial periods, "
              f"{len(phi_g)} points")

    res = dict(phi_global=phi_g, u=u_g, pr=pr_g, x=x, y=y, theta_section=th_s,
               Phi_r=Phi, phi_first_section=phi_s0, E0=E0, L0=L,
               periodicity_defect=per, energy_defect=dE, full_rk=None)

    # ── validation only: same metric as generate_long_orbit(full_rk_check) ─
    if full_rk_check:
        n_orb = int(math.ceil(phi_total/(2*math.pi)*1.15)) + 1
        ref = integrate_orbit(*[float(v) for v in theta0], pn_order, n_orb,
                              max(4000, 4*len(phi_g)))
        m = phi_g <= ref["phi"][-1]
        u_ref = np.interp(phi_g[m], ref["phi"], ref["u"])
        pr_ref = np.interp(phi_g[m], ref["phi"], ref["pr"])
        du = np.linalg.norm(u_g[m]-u_ref); dp = np.linalg.norm(pr_g[m]-pr_ref)
        nrm = math.sqrt(np.linalg.norm(u_ref)**2 + np.linalg.norm(pr_ref)**2)
        res["full_rk"] = dict(l2=math.sqrt(du**2+dp**2)/nrm,
                              n_rev=float(phi_g[m][-1]/(2*math.pi)), mask=m,
                              phi=ref["phi"], u=ref["u"], pr=ref["pr"])
        if verbose:
            print(f"  [full_rk_check] END-TO-END L2 over "
                  f"{res['full_rk']['n_rev']:.1f} rev: {res['full_rk']['l2']:.3e}")
    return res
