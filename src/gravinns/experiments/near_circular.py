"""Shared configuration of the near-circular (NC) cold-vs-transfer study.

Origin: notebook cell 2.  Every NC sweep (the original sweep, the top-up and
the n = 30 extension) uses the same fixed-order machinery and the same FULL
training budget; only the IC set and the output folder differ.
"""

PN_ORDER_NC = 2

# ── same fixed-order machinery and FULL budget for every NC sweep ─────────
COMMON_NC = dict(
    target_pn_order    = PN_ORDER_NC,
    n_orbits           = 10,
    compute_l2_time    = False,
    log_every          = 500,
    plot_every         = 1000,
    checkpoint_every   = 5_000,
    hidden = 128, depth = 5,
    n_fourier = 10, fourier_sigma = 3.0,
    n_fourier_hi = 16, fourier_sigma_hi = 8.0,
    lr = 3e-3, n_colloc = 4000,
    w_anchor = 10.0, w_energy = 5.0,
    convergence_window = 20, convergence_cv = 0.5,
    n_epochs           = 20_000,           # FULL budget - do not shorten
)

# The two arms of the comparison.
ARMS_NC = {
    "transfer": dict(transfer_from_2pn=True, pretrain_epochs=10_000,
                     curriculum_start=2_000, curriculum_epochs=8_000),
    "cold":     dict(transfer_from_2pn=False, pretrain_epochs=0,
                     curriculum_start=0, curriculum_epochs=0),
}
