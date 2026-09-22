"""Data-provenance checks for dissipative physics-only training."""
from __future__ import annotations


def prepare_dissipative_audit(
    *,
    dissipative: bool,
    no_25pn_data: bool,
    anchor_floor: float,
    tensors,
):
    """Print the provenance contract and make reference tensors auditable."""
    if not dissipative:
        return [], True

    print("  " + "─" * 66)
    print(f"  DATA-PROVENANCE AUDIT  (no_25pn_data={no_25pn_data})")
    print("    loss terms in use:")
    print("      l_pde     (PDE residual: 2PN H + F_RR)         : ALLOWED (equations)")
    print("      l_L       (ang.-mom. balance, F_RR torque)     : ALLOWED (equations)")
    print("      l_secular (energy balance dH/dphi = F_RR.v/phidot): ALLOWED (equations)")
    if no_25pn_data:
        print("      l_anchor  (2PN conservative orbit only)        : ALLOWED (2PN warm-start)")
        print("      2.5PN RK orbit (u/pr/L_25pn_ang)              : BLOCKED  <-- NOT in loss")
        print(f"      anchor_floor = {anchor_floor} (alpha -> 0 after curriculum)")
    else:
        print("      l_anchor  (2.5PN RK INSPIRAL u/pr/L)          : ALLOWED  <-- 2.5PN DATA IN LOSS!")
        print("      *** WARNING: this run USES 2.5PN RK data as supervision. ***")
        print("      *** Set no_25pn_data=True for a physics-only result.    ***")
    print("    reference 2.5PN orbit is used ONLY for: L2 metric, plots, diagnostics.")
    print("  " + "─" * 66)

    audit_tensors = list(tensors)
    for tensor in audit_tensors:
        try:
            tensor.requires_grad_(True)
        except Exception:
            pass
    return audit_tensors, False
