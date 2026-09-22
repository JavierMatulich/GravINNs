"""Physical constants and the global code-unit normalisation.

Origin: notebook cell 0 ("REPRODUCIBILITY LOCK / DEVICE SETUP / USER-CONTROLLABLE
PHYSICAL PARAMETERS") and the "Block-level normalization constants" at the end
of that cell.

Everything downstream (the Hamiltonian, the trainer, the IC diagnostics) works
in code units where lengths are measured in ``R0`` and velocities in
``V_char = sqrt(G M / R0)``.  In those units the speed of light is
``c_norm = c / V_char`` and ``C_SQ = c_norm**2``.

The values are module-level attributes so that library code reads them at
*call time* (``gravinns.constants.C_SQ``).  A case that needs a different binary
calls :func:`configure` once, before training::

    import gravinns.constants as const
    const.configure(R0=3.0e5, p0_factor=0.9)

With no call to :func:`configure` the defaults reproduce the notebook exactly
(2 x 1.4 M_sun, R0 = 200 km, p0_factor = 0.8).
"""
from __future__ import annotations

import numpy as np

# ── fundamental constants (SI) ────────────────────────────────────────────
G = 6.67430e-11
c_si = 299792458.0
M_sun = 1.989e30

# ── defaults of the physical set-up (cell 0) ─────────────────────────────
DEFAULTS = dict(
    m1=1.4 * M_sun,
    m2=1.4 * M_sun,
    R0=2.0e5,          # initial separation / reference length (m)
    p0_factor=0.8,     # 1.0 circular, <1 sub-circular, >1 super-circular
    nu_user=None,      # set to override the physical symmetric mass ratio
    t_max=12.0,
    num_points=2000,
)

_KEEP = object()

# Declared here for readers and linters; the values are set by configure()
# (called by reset() at the bottom of this module).
m1 = m2 = R0 = p0_factor = nu_user = t_max = num_points = None
M_char = L_char = V_char = T_char = None
q0_norm = v0_si = p0_norm = None
c_norm = C_SQ = C_QD = None
nu_physical = NU = None
q0 = p0 = y0 = t_eval = None
R0_REF_M = R0_REF_KM = R_G_KM = R_ISCO_KM = R_PN_KM = None


def configure(m1=_KEEP, m2=_KEEP, R0=_KEEP, p0_factor=_KEEP,
              nu_user=_KEEP, t_max=_KEEP, num_points=_KEEP) -> None:
    """(Re)compute every derived constant.  Arguments that are not given keep
    their current value.  Masses in kg, ``R0`` in metres."""
    g = globals()
    new = dict(m1=m1, m2=m2, R0=R0, p0_factor=p0_factor, nu_user=nu_user,
               t_max=t_max, num_points=num_points)
    for k, v in new.items():
        if v is not _KEEP:
            g[k] = v

    M = g["m1"] + g["m2"]
    L = g["R0"]
    V = float(np.sqrt(G * M / L))
    g.update(M_char=M, L_char=L, V_char=V, T_char=L / V)

    # dimensionless initial state
    g["q0_norm"] = 1.0
    g["v0_si"] = g["p0_factor"] * V
    g["p0_norm"] = g["v0_si"] / V

    # relativistic normalisation: velocities are in units of V_char
    g["c_norm"] = c_si / V
    g["C_SQ"] = g["c_norm"] ** 2
    g["C_QD"] = g["c_norm"] ** 4

    g["nu_physical"] = (g["m1"] * g["m2"]) / (M ** 2)
    g["NU"] = g["nu_physical"] if g["nu_user"] is None else g["nu_user"]

    g["q0"] = [g["q0_norm"], 0.0]
    g["p0"] = [0.0, g["p0_norm"]]
    g["y0"] = g["q0"] + g["p0"]
    g["t_eval"] = np.linspace(0, g["t_max"], g["num_points"])

    # block-level aliases used by the trainer and the sweep
    g["R0_REF_M"] = L
    g["R0_REF_KM"] = L / 1e3
    g["R_G_KM"] = G * M / (c_si ** 2) / 1e3
    g["R_ISCO_KM"] = 6.0 * g["R_G_KM"]
    g["R_PN_KM"] = 20.0 * g["R_G_KM"]


def reset() -> None:
    """Restore the notebook defaults."""
    configure(**DEFAULTS)


def print_normalization_summary() -> None:
    """The '--- NORMALIZATION SUMMARY ---' printout of cell 0."""
    print("\n--- NORMALIZATION SUMMARY ---")
    print(f"Total mass                : {M_char / M_sun:.4f} M_sun")
    print(f"Initial separation R0     : {R0/1000:.3f} km")
    print(f"Characteristic velocity   : {V_char:.6e} m/s")
    print(f"Characteristic timescale  : {T_char:.6e} s")
    print(f"Normalized momentum p0    : {p0_norm:.6f}")
    print(f"Normalized speed of light : {c_norm:.6f}")
    print(f"C_SQ                      : {C_SQ:.6e}")
    print(f"Symmetric mass ratio NU   : {NU:.6f}")
    print("--------------------------------")


def check_relativistic_safety(M_char_=None, R0_=None, p0_factor_=None,
                              G_=G, c_si_=c_si) -> None:
    """ISCO / 2PN-validity check of cell 0 (defaults to the current set-up)."""
    M_char_ = M_char if M_char_ is None else M_char_
    R0_ = R0 if R0_ is None else R0_
    p0_factor_ = p0_factor if p0_factor_ is None else p0_factor_

    r_g = G_ * M_char_ / (c_si_ ** 2)
    r_isco = 6.0 * r_g
    r_pn_breakdown = 20.0 * r_g

    if p0_factor_ < 1.0:
        r_periapsis = R0_ * (p0_factor_ ** 2) / (2.0 - p0_factor_ ** 2)
    elif p0_factor_ < 1.414:
        r_periapsis = R0_
    else:
        print("\n[WARNING] Unbound orbit.")
        return

    print("\n--- Relativistic Diagnostics ---")
    print(f"Gravitational radius     : {r_g/1000:.3f} km")
    print(f"ISCO radius              : {r_isco/1000:.3f} km")
    print(f"2PN breakdown threshold  : {r_pn_breakdown/1000:.3f} km")
    print(f"Predicted periapsis      : {r_periapsis/1000:.3f} km")

    if r_periapsis <= r_isco:
        print("WARNING: Orbit crosses ISCO.")
    elif r_periapsis <= r_pn_breakdown:
        print("WARNING: Orbit enters weak 2PN validity regime.")
    else:
        print("Orbit safely inside 2PN regime.")
    print("--------------------------------")


def print_block_summary() -> None:
    """The 'Block 11 ... global normalization' printout of cell 0."""
    print("─" * 56)
    print("  Block 11 (improved, 2PN-data-free) — global normalization")
    print(f"  R0_ref = {R0_REF_KM:.1f} km   V_char = {V_char:.4e} m/s")
    print(f"  C_SQ   = {C_SQ:.6e}")
    print(f"  r_g    = {R_G_KM:.4f} km   r_PN = {R_PN_KM:.3f} km")
    print("─" * 56)


def print_setup() -> None:
    """Everything cell 0 printed, in the same order."""
    print_normalization_summary()
    check_relativistic_safety()
    print_block_summary()


reset()
