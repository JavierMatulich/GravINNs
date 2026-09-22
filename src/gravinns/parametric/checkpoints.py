"""Loading trained parametric models.

Origin: GitHub_parametric.ipynb, cell 2.  The notebook defined
``load_parametric_model`` three times; only the last definition (the one the
notebook actually used) is kept.
"""
import torch

from .model import ParametricAnglePINN



def load_parametric_model(base_dir, which="best", phi_max=None,
                          device="cpu", verbose=True):
    """Load a saved parametric-PINN checkpoint, choosing WHICH one.

    which : "final" -> param_pinn.pt       (last-epoch weights)
            "best"  -> param_pinn_best.pt  (lowest heatmap-median L2)
            "last"  -> param_pinn_last.pt  (latest refresh checkpoint)
            or an explicit filename.

    The checkpoints are bare state_dicts, but almost the entire architecture
    can be INFERRED from them: theta ranges and the u-band live in registered
    buffers; n_fourier/n_fourier_hi/n_harmonic from the B/B_hi/harm_n buffer
    lengths; hidden/depth/omega_hidden from the layer shapes. The ONE thing a
    state_dict does not carry is phi_max (a plain float) -- and it matters:
    it scales the Fourier features and the IC gate, so a wrong value silently
    distorts every prediction. Newer runs write param_pinn_config.json next to
    the weights and this loader reads it automatically; for older runs pass
    phi_max= the value printed at training time ("phi_max_global=... rad").

    Returns the reconstructed model in eval mode, ready to be passed to
    refine_parametric_pinn / l2_sweep_1d / generate_long_orbit.
    """
    import os, json
    fname = {"final": "param_pinn.pt", "best": "param_pinn_best.pt",
             "last": "param_pinn_last.pt",
             "refined": "param_pinn_refined.pt"}.get(which, which)
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        avail = [f for f in os.listdir(base_dir) if f.endswith(".pt")]             if os.path.isdir(base_dir) else []
        raise FileNotFoundError(f"{path} not found. Available: {avail}")
    sd = torch.load(path, map_location=device)

    cfgp = os.path.join(base_dir, "param_pinn_config.json")
    cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else None
    if phi_max is None:
        if cfg is not None:
            phi_max = cfg["phi_max_global"]
        else:
            raise ValueError(
                "phi_max is not stored inside this (older) checkpoint and no "
                "param_pinn_config.json was found. Pass phi_max= the value "
                "printed at training time, e.g. 'phi_max_global=17.63 rad' -> "
                "load_parametric_model(..., phi_max=17.63). Runs made with the "
                "current trainer write the config file automatically.")

    keys = ("p0_factor", "r0_km", "pr0_factor", "nu")
    ranges = {k: (float(sd["theta_lo"][i]), float(sd["theta_hi"][i]))
              for i, k in enumerate(keys)}
    n_fourier = int(sd["B"].numel())
    n_fourier_hi = int(sd["B_hi"].numel())
    n_harmonic = int(sd["harm_n"].numel()) if "harm_n" in sd else 0
    hidden = int(sd["net.0.weight"].shape[0])
    n_lin = len([k for k in sd if k.startswith("net.") and k.endswith(".weight")])
    depth = n_lin - 1
    omega_hidden = (int(sd["omega_net.0.weight"].shape[0])
                    if "omega_net.0.weight" in sd else 32)

    model = ParametricAnglePINN(
        phi_max=float(phi_max), param_ranges=ranges,
        u_lo=float(sd["u_lo"]), u_hi=float(sd["u_hi"]),
        n_fourier=n_fourier, n_fourier_hi=n_fourier_hi,
        n_harmonic=n_harmonic, hidden=hidden, depth=depth,
        pr_max=float(sd["pr_max"]), omega_hidden=omega_hidden).to(device)
    model.load_state_dict(sd)     # overwrites B/B_hi too -> exact restoration
    model.eval()
    if verbose:
        print(f"loaded {path}")
        print(f"  arch: hidden={hidden} depth={depth} n_fourier={n_fourier}/"
              f"{n_fourier_hi} n_harmonic={n_harmonic} omega_hidden={omega_hidden}")
        print(f"  phi_max={float(phi_max):.4f} rad | box: "
              + ", ".join(f"{k}=({v[0]:g},{v[1]:g})" for k, v in ranges.items()))
        if cfg is not None and "n_orbits" in cfg:
            print(f"  run settings: n_orbits={cfg['n_orbits']} "
                  f"n_pts={cfg['n_pts']} (a refinement of this model MUST "
                  f"use the same two values)")
    return model


def get_param_ranges(model):
    """Recover the param_ranges dict a model was trained on (from its
    registered theta_lo/theta_hi buffers) -- so after a kernel restart you do
    not have to retype the box, and cannot mistype it."""
    keys = ("p0_factor", "r0_km", "pr0_factor", "nu")
    return {k: (float(model.theta_lo[i]), float(model.theta_hi[i]))
            for i, k in enumerate(keys)}
