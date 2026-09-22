"""Small helpers that were inline code in GitHub_parametric.ipynb.

* ``build_ref_data_from_cache``  cell 5  -- anchors from a reference cache
* ``anchors_and_probes_from_cache``  cell 5  -- anchor thetas + 3 seen / 3 unseen probes
* ``predict_orbit``              cell 7  -- orbit of any theta from a trained model
* ``ecc_bin_table`` / ``ecc_bin_boxplot``  cell 16 -- per-eccentricity statistics
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


def build_ref_data_from_cache(cache, indices, phi_max_global, n_pts):
    """Supervised data for ``train_parametric_pinn(ref_data=...)``: every
    anchor orbit resampled on one common phi grid [0, phi_max_global]."""
    phi_grid = np.linspace(0, phi_max_global, n_pts)
    thetas = cache['thetas'][indices]
    U = np.array([np.interp(phi_grid, cache['PHI'][i], cache['U'][i]) for i in indices])
    PR = np.array([np.interp(phi_grid, cache['PHI'][i], cache['PR'][i]) for i in indices])
    return dict(
        THETA=torch.tensor(thetas, dtype=torch.float32),
        U=torch.tensor(U, dtype=torch.float32),
        PR=torch.tensor(PR, dtype=torch.float32),
        PHI=torch.tensor(phi_grid, dtype=torch.float32)
    )


def anchors_and_probes_from_cache(cache, n_anchors, n_seen=3, n_unseen=3):
    """First ``n_anchors`` orbits of the cache are the anchors; the probes are
    the first ``n_seen`` anchors plus the ``n_unseen`` orbits that follow them."""
    anchor_indices = range(n_anchors)
    anchor_thetas = [tuple(row) for row in cache['thetas'][anchor_indices]]
    seen_thetas = [anchor_thetas[i] for i in range(n_seen)]
    unseen_thetas = [tuple(cache['thetas'][n_anchors + i]) for i in range(n_unseen)]
    probe_thetas = seen_thetas + unseen_thetas
    probe_is_seen = [True]*n_seen + [False]*n_unseen
    return anchor_indices, anchor_thetas, probe_thetas, probe_is_seen


# ── Predict an UNSEEN orbit ──────────────────────────────────────────────
# theta must lie inside param_ranges. Returns u(phi), pr(phi), L on a grid.
def predict_orbit(model, p0_factor, r0_km, pr0_factor, nu, n_pts=2000, device=None):
    """Example::

        orb = predict_orbit(model, 0.51, 118.0, 0.24, 0.235)
        plt.plot(orb["x"], orb["y"]); plt.gca().set_aspect("equal")
    """
    if device is None:
        device = next(model.parameters()).device
    phi = torch.linspace(0, model.phi_max, n_pts, device=device).view(-1, 1)
    th  = torch.tensor([[p0_factor, r0_km, pr0_factor, nu]],
                       dtype=torch.float32, device=device).repeat(n_pts, 1)
    with torch.no_grad():
        out = model(phi, th).cpu().numpy()
    phi_np = phi.cpu().numpy().ravel()
    u, pr = out[:, 0], out[:, 1]
    return dict(phi=phi_np, u=u, pr=pr,
                x=np.cos(phi_np)/u, y=np.sin(phi_np)/u)


def ecc_bin_table(ecc, l2, ecc_bins=(0.0, 0.3, 0.5, 0.65, 0.8), tol=1e-3):
    """'Eccentricity bin statistics (single model)' table of cell 16."""
    ecc = np.asarray(ecc); l2 = np.asarray(l2)
    print("\nEccentricity bin statistics (single model):")
    print(f"{'ecc bin':<15} {'n':>6} {'median L2':>12} {'frac < tol':>12}")
    print("-" * 50)
    for i in range(len(ecc_bins)-1):
        lo, hi = ecc_bins[i], ecc_bins[i+1]
        mask = (ecc >= lo) & (ecc < hi)
        n = np.sum(mask)
        if n > 0:
            med = np.median(l2[mask])
            frac = np.mean(l2[mask] < tol)
            print(f"[{lo:.2f},{hi:.2f}) {n:>6} {med:>12.2e} {frac*100:>11.1f}%")
        else:
            print(f"[{lo:.2f},{hi:.2f}) {n:>6} {'--':>12} {'--':>11}")


def ecc_bin_boxplot(ecc, l2, ecc_bins=(0.0, 0.3, 0.5, 0.65, 0.8)):
    """L2 boxplots per eccentricity bin (cell 16, optional visualisation)."""
    ecc = np.asarray(ecc); l2 = np.asarray(l2)
    plt.figure(figsize=(10,6))
    data_by_bin = []
    bin_centers = 0.5 * (np.array(ecc_bins[1:]) + np.array(ecc_bins[:-1]))
    for i in range(len(ecc_bins)-1):
        lo, hi = ecc_bins[i], ecc_bins[i+1]
        mask = (ecc >= lo) & (ecc < hi)
        data_by_bin.append(l2[mask] if np.any(mask) else [])
    plt.boxplot(data_by_bin, positions=bin_centers, widths=0.02*(ecc_bins[-1]-ecc_bins[0]))
    plt.xlabel("Eccentricity")
    plt.ylabel("L2 error")
    plt.title("L2 error distribution per eccentricity bin")
    plt.grid(True, alpha=0.3)
    plt.show()
