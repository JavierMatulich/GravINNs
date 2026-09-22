"""Initial-condition diagnostics and generators for the single-orbit sweeps.

Origin: sections B.1-B.2 of notebook cell 3 ("SINGLE-ORBIT STATISTICS SWEEP").
"""
import numpy as np

from .. import constants as _const


# ---------------------------------------------------------------------------
#  B.1  Physical diagnostics for a candidate initial condition
# ---------------------------------------------------------------------------
#  These use ONLY the Newtonian closed forms already used elsewhere in the
#  notebook (same convention as diagnose_pn_regime), so the numbers are
#  directly comparable to the draft's Sec. V-C stratification.
#
#  r_g in code units: the notebook rescales r -> r/(GM), so r_g = GM/c^2 maps
#  to 1/c^2 = 1/C_SQ. Hence  r_p/r_g = r_p * C_SQ.
# ---------------------------------------------------------------------------

def ic_diagnostics(nu, p0_factor, r0_km, pr0_factor=0.0):
    """Closed-form orbital diagnostics for one IC. No integration."""
    CSQ = _const.C_SQ
    q0  = r0_km / _const.R0_REF_KM
    p0t = p0_factor / np.sqrt(q0)
    pr0 = float(pr0_factor)
    L0  = q0 * p0t
    E   = 0.5 * (pr0**2 + p0t**2) - 1.0 / q0

    d = dict(nu=float(nu), p0_factor=float(p0_factor), r0_km=float(r0_km),
             pr0_factor=pr0, q0=float(q0), p0t=float(p0t), L0=float(L0),
             E_newt=float(E))

    if E >= 0:
        d.update(bound=False, ecc=float("nan"), a=float("inf"),
                 r_peri=float("nan"), r_apo=float("inf"),
                 rp_over_rg=float("nan"), x_peri=float("nan"),
                 T_orb=float("nan"), pn_class="unbound", isco_flag=True)
        return d

    a    = -0.5 / E
    ecc  = float(np.sqrt(max(0.0, 1.0 + 2.0 * E * L0**2)))
    rp   = a * (1.0 - ecc)
    ra   = a * (1.0 + ecc)
    T    = 2.0 * np.pi * a**1.5
    # (v/c)^2 at periapsis, Newtonian vis-viva
    x    = (2.0 / rp - 1.0 / a) / CSQ
    rp_rg = rp * CSQ          # periapsis in gravitational radii

    # Physical validity classification, matching the draft's thresholds:
    #   r_ISCO = 6 r_g ;  2PN quantitatively reliable above ~20 r_g
    if   rp_rg < 6.0:  cls = "inside_isco"
    elif rp_rg < 10.0: cls = "strong_field"
    elif rp_rg < 20.0: cls = "transition"
    else:              cls = "pn_valid"

    d.update(bound=True, ecc=ecc, a=float(a), r_peri=float(rp),
             r_apo=float(ra), rp_over_rg=float(rp_rg), x_peri=float(x),
             T_orb=float(T), pn_class=cls, isco_flag=bool(rp_rg < 6.0))
    return d


def ecc_bin(e):
    """Eccentricity strata. The first bin isolates the near-circular regime
    where the local-minimum degeneracy is expected."""
    if not np.isfinite(e):      return "nan"
    if e < 0.05:                return "quasi_circular"
    if e < 0.20:                return "low"
    if e < 0.40:                return "moderate"
    if e < 0.60:                return "high"
    return "very_high"


def rp_bin(rp_rg):
    if not np.isfinite(rp_rg):  return "nan"
    if rp_rg < 10.0:            return "[0,10)"
    if rp_rg < 20.0:            return "[10,20)"
    if rp_rg < 40.0:            return "[20,40)"
    return "[40,inf)"


# ---------------------------------------------------------------------------
#  B.2  IC generation
# ---------------------------------------------------------------------------
#  Two modes:
#    'grid'   - deterministic tensor product; reproducible, good for figures
#               that need a filled (p0,r0) plane.
#    'lhs'    - Latin hypercube over the 4D box; better statistics per run.
#  Both are filtered to bound orbits and (optionally) to a minimum periapsis,
#  so the sweep does not waste GPU time on ICs where the 2PN Hamiltonian is
#  not a faithful description in the first place. Rejected ICs are RECORDED
#  (not silently dropped) so the paper can quote how much of the box is
#  physically inadmissible.
# ---------------------------------------------------------------------------

def _lhs(n, dim, rng):
    """Latin hypercube sample in [0,1]^dim."""
    out = np.empty((n, dim))
    for j in range(dim):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / n
    return out


def generate_ics(mode="lhs", n=64, seed=0,
                 nu_range=(0.10, 0.25),
                 p0_range=(0.55, 1.00),
                 r0_range=(150.0, 350.0),
                 pr0_range=(0.0, 0.0),
                 min_rp_rg=10.0,
                 require_bound=True,
                 grid_shape=None):
    """Return (accepted, rejected) lists of diagnostic dicts."""
    rng = np.random.default_rng(seed)

    if mode == "grid":
        gs = grid_shape or (4, 4, 3, 1)
        axes = [np.linspace(*nu_range,  gs[0]),
                np.linspace(*p0_range,  gs[1]),
                np.linspace(*r0_range,  gs[2]),
                np.linspace(*pr0_range, gs[3])]
        pts = np.array(np.meshgrid(*axes, indexing="ij")).reshape(4, -1).T
    elif mode == "lhs":
        U = _lhs(n, 4, rng)
        lo = np.array([nu_range[0], p0_range[0], r0_range[0], pr0_range[0]])
        hi = np.array([nu_range[1], p0_range[1], r0_range[1], pr0_range[1]])
        pts = lo + U * (hi - lo)
    else:
        raise ValueError("mode must be 'grid' or 'lhs'")

    acc, rej = [], []
    for k, (nu, p0f, r0, pr0) in enumerate(pts):
        d = ic_diagnostics(nu, p0f, r0, pr0)
        d["ic_index"] = int(k)
        d["ecc_bin"]  = ecc_bin(d["ecc"])
        d["rp_bin"]   = rp_bin(d["rp_over_rg"])
        bad = []
        if require_bound and not d["bound"]:
            bad.append("unbound")
        if d["bound"] and np.isfinite(min_rp_rg) and d["rp_over_rg"] < min_rp_rg:
            bad.append(f"rp<{min_rp_rg:g}rg")
        if bad:
            d["reject_reason"] = ",".join(bad)
            rej.append(d)
        else:
            acc.append(d)
    return acc, rej


def _p0_for_ecc(nu, r0_km, pr0, e_target, tol=1e-10, max_iter=200):
    """Solve for the p0_factor giving eccentricity e_target at this pr0.

    For pr0 = 0 the IC is at an apsis and the Newtonian relation is exact,
        e = |1 - p0_factor^2|,
    so p0_factor = sqrt(1-e) on the sub-circular (apoapsis-start) branch.

    For pr0 != 0 the IC is OFF-apsis: E_0 picks up the pr0^2/2 term while L_0
    does not, so e is no longer 1-p0^2 and the closed form mislabels the
    stratum (e.g. at p0=0.9, r0=250km, pr0=0.3 the true e is 0.357, not 0.19).
    We therefore bisect on the exact expression e(p0) = sqrt(1+2 E_0 L_0^2),
    which is monotonically decreasing in p0 on the sub-circular branch.

    Returns None if the target is unreachable at this (nu, r0, pr0).
    """
    q0 = r0_km / _const.R0_REF_KM

    def _e(p0f):
        p0t = p0f / np.sqrt(q0)
        L0  = q0 * p0t
        E   = 0.5 * (pr0**2 + p0t**2) - 1.0 / q0
        if E >= 0:
            return None                      # unbound
        return float(np.sqrt(max(0.0, 1.0 + 2.0 * E * L0**2)))

    if abs(pr0) < 1e-12:
        return float(np.sqrt(max(1e-6, 1.0 - e_target)))

    # Sub-circular branch: e decreases as p0 grows. Bracket [lo, hi].
    lo, hi = 1e-3, 1.0
    e_lo, e_hi = _e(lo), _e(hi)
    if e_lo is None or e_hi is None:
        return None
    if not (min(e_lo, e_hi) - 1e-9 <= e_target <= max(e_lo, e_hi) + 1e-9):
        return None                          # target outside reachable range
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        e_m = _e(mid)
        if e_m is None:
            return None
        if abs(e_m - e_target) < tol:
            return float(mid)
        # e is decreasing in p0
        if e_m > e_target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def generate_ics_stratified(n_per_bin=8, seed=0,
                            nu_range=(0.10, 0.25),
                            r0_range=(150.0, 350.0),
                            pr0_range=(0.0, 0.0),
                            ecc_targets=(0.02, 0.10, 0.30, 0.50, 0.65),
                            ecc_tol=0.02, min_rp_rg=10.0, max_tries=20000,
                            verbose=True):
    """Sample ICs with GUARANTEED coverage of each eccentricity stratum.

    Uniform sampling of the (nu, p0, r0) box under-populates the near-circular
    corner (e < 0.05 is a thin slice), yet that corner carries one of the two
    physical claims. Here we sample (nu, r0, pr0) freely and SOLVE for the
    p0_factor realising each target eccentricity, so every stratum receives the
    same number of runs and the comparison between strata is balanced.

    WHY THERE IS NO p0_range ARGUMENT
    ---------------------------------
    At pr0 = 0, p0_factor and eccentricity are the SAME degree of freedom
    (e = 1 - p0^2). Specifying both would be redundant at best and
    contradictory at worst. Eccentricity is the better control variable: it is
    the physically meaningful label, it is what the stratification tables are
    binned by, and it is comparable across different r0. p0_factor is a
    coordinate artefact of the IC parametrisation. If you genuinely want a
    p0-box sample instead, use generate_ics(mode='lhs', p0_range=...), which
    samples the raw box and lets eccentricity fall where it may.

    WHY pr0_range IS DIFFERENT
    --------------------------
    pr0 is NOT redundant with eccentricity. At fixed e it moves the starting
    point along the orbit: pr0 = 0 starts at an apsis, pr0 != 0 starts off-apsis
    (infalling for pr0 < 0, outgoing for pr0 > 0). This changes where the
    network's exactly-enforced initial condition sits relative to the periapsis
    passage, which is the phase anchor of the phase-blindness argument, so it is
    a genuinely independent axis worth sampling.
    """
    rng = np.random.default_rng(seed)
    acc, rej = [], []
    for e_t in ecc_targets:
        got, tries, n_unreach = 0, 0, 0
        while got < n_per_bin and tries < max_tries:
            tries += 1
            nu  = rng.uniform(*nu_range)
            r0  = rng.uniform(*r0_range)
            pr0 = rng.uniform(*pr0_range)
            e_j = float(np.clip(e_t + rng.normal(0, ecc_tol), 0.0, 0.95))

            p0f = _p0_for_ecc(nu, r0, pr0, e_j)
            if p0f is None:
                n_unreach += 1
                continue

            d = ic_diagnostics(nu, p0f, r0, pr0)
            if not d["bound"]:
                continue
            # Guard: verify the solve actually hit the requested stratum.
            if not np.isfinite(d["ecc"]) or abs(d["ecc"] - e_j) > 5e-3:
                n_unreach += 1
                continue
            if np.isfinite(min_rp_rg) and d["rp_over_rg"] < min_rp_rg:
                d["reject_reason"] = f"rp<{min_rp_rg:g}rg"
                rej.append(d)
                continue

            d["ic_index"]   = len(acc)
            d["ecc_bin"]    = ecc_bin(d["ecc"])
            d["rp_bin"]     = rp_bin(d["rp_over_rg"])
            d["ecc_target"] = float(e_t)
            acc.append(d)
            got += 1

        if verbose and got < n_per_bin:
            print(f"  [warn] ecc target {e_t}: only {got}/{n_per_bin} ICs "
                  f"({n_unreach} unreachable at the sampled pr0, "
                  f"rest below rp>{min_rp_rg} r_g in this box)")
    return acc, rej
