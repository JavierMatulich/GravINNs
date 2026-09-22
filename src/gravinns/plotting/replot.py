"""Redraw the dissipative training figures from a saved plot record.

Origin: GitHub_Dissipative.ipynb, cell 6.  Reads ``plot_record.npz`` written by
the v4 trainer -- no retraining, no GPU.  (``_save_panels_separately`` there was
identical to the shared helper and is imported from it.)
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .panels import save_panels_separately as _save_panels_separately

PLOTREC = "plot_record.npz"
_STAGE_NAMES = {1: "Stage1:Transfer", 2: "Stage2:Causal", 3: "Stage3:Standard"}


def load_plot_record(path):
    """Return the record as a plain dict of numpy arrays (no pickle)."""
    if os.path.isdir(path):
        path = os.path.join(path, "plot_record.npz")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}
def _R(rec):
    return load_plot_record(rec) if isinstance(rec, str) else rec
def _st(R, k, default=None):
    a = R.get("static_" + k)
    if a is None:
        return default
    return a.item() if a.ndim == 0 else a
def list_energy_snapshots(rec):
    R = _R(rec)
    ep = R.get("energy_epoch", np.array([]))
    print(f"{'#':>3} {'epoch':>9} {'stage':<16} {'L2 shape':>10} {'ΔH':>12} {'rising %':>9}")
    for i, e in enumerate(ep):
        print(f"{i:>3} {int(e):>9,} {_STAGE_NAMES.get(int(R['energy_stage'][i]), ''):<16} "
              f"{R['energy_l2_shape'][i]:>10.3e} {R['energy_dE_tot'][i]:>+12.3e} "
              f"{100*R['energy_frac_up'][i]:>8.1f}%")
    for tag in ("best", "polish"):
        if f"final_{tag}_H" in R:
            H = R[f"final_{tag}_H"]
            print(f"  final '{tag}': L2 shape={float(R[f'final_{tag}_l2_shape']):.3e}  "
                  f"ΔH={H[-1]-H[0]:+.3e}  rising={100*(np.diff(H) > 0).mean():.1f}%")
    return ep
def _snapshot(R, epoch):
    """-> dict(ep, stage, u, H, lr, l2s, dE, frac_up, name)."""
    if epoch in ("best", "polish"):
        k = f"final_{epoch}_"
        if k + "H" not in R:
            raise KeyError(f"no '{epoch}' snapshot in this record")
        H = R[k + "H"]; dH = np.diff(H)
        ep_last = int(R["energy_epoch"][-1]) if R.get("energy_epoch", np.array([])).size else 0
        return dict(ep=ep_last, stage=0, u=R[k + "u"], H=H, lr=np.nan,
                    l2s=float(R[k + "l2_shape"]), dE=float(H[-1] - H[0]),
                    frac_up=float((dH > 0).mean()), name=epoch)
    eps = R["energy_epoch"]
    i = len(eps) - 1 if epoch is None else int(np.argmin(np.abs(eps - epoch)))
    if epoch is not None and eps[i] != epoch:
        print(f"  (no snapshot at epoch {epoch:,}; using nearest {int(eps[i]):,})")
    return dict(ep=int(eps[i]), stage=int(R["energy_stage"][i]), u=R["energy_u"][i],
                H=R["energy_H"][i], lr=float(R["energy_lr"][i]),
                l2s=float(R["energy_l2_shape"][i]), dE=float(R["energy_dE_tot"][i]),
                frac_up=float(R["energy_frac_up"][i]), name=None)
def replot_training_figure(rec, epoch=None, truncate=True, save=None,
                            save_dir=None, dpi=300):
    R = _R(rec)
    S = _snapshot(R, epoch)
    diss = bool(_st(R, "dissipative"))
    phi = _st(R, "phi_grid"); l2_target = float(_st(R, "l2_target"))
    ep_cut = S["ep"] if (truncate and S["name"] is None) else np.inf

    ncols = 4 if diss else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    # ── panel 1: orbit ────────────────────────────────────────────────────
    ax = axes[0]
    def _xy(k):
        a = _st(R, k); return a if (a is not None and a.size) else None
    for k, kw in (("orbit_0pn", dict(color="grey", ls=":", lw=1.2, alpha=.7, label="0PN RK45")),
                  ("orbit_1pn", dict(color="purple", ls="-.", lw=1.2, alpha=.6, label="1PN RK45")),
                  ("orbit_2pn", dict(color="steelblue", ls="--", lw=1.6, alpha=.85,
                                     label="2PN RK45 [pretrain src]")),
                  ("orbit_target", dict(color="seagreen", ls="-", lw=1.9, alpha=.95,
                                        label=str(_st(R, "target_label"))))):
        a = _xy(k)
        if a is not None:
            ax.plot(a[0], a[1], **kw)
    a = _xy("orbit_warmstart")
    if a is not None:
        ax.plot(a[0], a[1], color="darkorange", ls=":", lw=1.3, alpha=.6,
                label=f"warm-start: {_st(R, 'warmstart_label')}")
    r_p = 1.0 / S["u"]
    ax.plot(r_p * np.cos(phi), r_p * np.sin(phi), color="crimson", lw=2, alpha=.95, label="PINN")
    ax.scatter([0], [0], color="k", marker="*", s=120, zorder=5)
    qmin, qmax = _st(R, "q_lim")
    ax.set_xlim(qmin, qmax); ax.set_ylim(qmin, qmax); ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    sl = f" [{_STAGE_NAMES[S['stage']]}]" if S["stage"] in _STAGE_NAMES else ""
    if S["name"]:
        sl = f" [final: {S['name']}]"
    ws_short = str(_st(R, "warmstart_label")).split(" ")[0]
    scheme = (f" | {ws_short}→{int(_st(R, 'target_pn_order'))}PN transfer"
              if _st(R, "transfer_from_2pn") else "")
    ax.set_title(f"Orbit — ep {S['ep']:,}{sl}{scheme}" + ("  (inspiral)" if diss else ""))

    # ── panel 2: loss — EXACT same arrays/drawing as the live figure ───────
    ax = axes[1]
    if "full_loss_epoch" in R and R["full_loss_epoch"].size:
        e_all = R["full_loss_epoch"]; v_all = R["full_loss_value"]
        m = e_all <= ep_cut
        e = e_all[m]; v = v_all[m]
        if len(e) > 1:
            ax.semilogy(e, v, color="steelblue", lw=.7, alpha=.5)
            w = min(50, len(v))
            ax.semilogy(e[w - 1:], np.convolve(v, np.ones(w) / w, "valid"),
                        color="navy", lw=2, label=f"mean(w={w})")
            ax.legend(fontsize=8)
    ax.set_title("Loss"); ax.grid(which="both", alpha=.3); ax.set_xlabel("epoch")

    # ── panel 3: L2 — EXACT same arrays/drawing as the live figure ─────────
    ax = axes[2]
    best_l2s = best_l2t = np.inf
    if "full_l2_epoch" in R and R["full_l2_epoch"].size:
        e_all = R["full_l2_epoch"]; l2s_all = R["full_l2_shape"]; l2t_all = R["full_l2_time"]
        m = e_all <= ep_cut
        e = e_all[m]; l2s = l2s_all[m]; l2t = l2t_all[m]
        if len(e) > 1:
            ax.semilogy(e, l2s, color="crimson", lw=2, label="L² shape")
            ax.fill_between(e, l2s, alpha=.12, color="crimson")
            if _st(R, "compute_l2_time") and np.isfinite(l2t).any():
                ax.semilogy(e, l2t, color="royalblue", lw=1.5, ls="--", label="L² time")
        if len(e) > 0:
            best_l2s = float(np.min(l2s))                      # running min, same as training's best_l2s
            _ft = l2t[np.isfinite(l2t)]
            best_l2t = float(np.min(_ft)) if _ft.size else np.inf
    ax.axhline(l2_target, color="crimson", ls=":", lw=1.2, label=f"target {l2_target:.0e}")
    if _st(R, "compute_l2_time") and best_l2t < 1.0:
        ax.axhline(best_l2t, color="seagreen", ls="--", lw=1.0, alpha=.7,
                   label=f"best L²t={best_l2t:.1e}")
    both = best_l2s < l2_target and best_l2t < l2_target
    ax.set_title("L² convergence" + (" ★ BOTH" if both else " ★ SHAPE" if best_l2s < l2_target else ""),
                 color="seagreen" if both else "black")
    ax.legend(fontsize=7); ax.grid(which="both", alpha=.3); ax.set_xlabel("epoch")

    # ── panel 4: energy ───────────────────────────────────────────────────
    if diss:
        ax = axes[3]
        E_t = float(_st(R, "E_target")); H_ref = _st(R, "H_ref")
        ax.plot(phi, S["H"], color="crimson", lw=1.8, label="H(φ) — PINN")
        if H_ref is not None and H_ref.size:
            ax.plot(phi, H_ref, color="seagreen", lw=1.5, ls="--", alpha=.9, label="H(φ) — 2.5PN ref")
        ax.axhline(E_t, color="steelblue", ls=":", lw=1.2, label=f"E* (conservative) = {E_t:.4f}")
        ok = (S["frac_up"] < 0.05) and (S["dE"] < 0)
        ax.set_title("Energy H(φ)  ✓ monotonic decay" if ok else "Energy H(φ)  ⚠ not monotonic",
                     color="seagreen" if ok else "darkorange")
        ax.set_xlabel("φ [rad]"); ax.set_ylabel("H"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="best")
        ax.text(0.02, 0.02, f"ΔH = {S['dE']:+.3e}\nrising steps = {100*S['frac_up']:.1f}%",
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=.7))

    lr_str = f" lr {S['lr']:.1e}" if np.isfinite(S["lr"]) else ""
    fig.suptitle(f"{_st(R, 'label')} [{_st(R, 'tag')}] | ep {S['ep']:,} | L²s {S['l2s']:.2e}{lr_str}",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if save_dir:
        panel_names = ["orbit", "loss", "l2"] + (["energy"] if diss else [])
        prefix = f"ep{S['ep']:06d}" if S["name"] is None else f"final_{S['name']}"
        paths = _save_panels_separately(fig, axes, panel_names, save_dir, prefix, dpi=dpi)
        print(f"  saved {len(paths)} panel PNGs to {save_dir}/")
    return fig
def plot_energy_evolution(rec, epochs=None, n_default=6, include_final=True,
                          save=None, save_dir=None, dpi=300):
    """H(φ) of the network at several training stages, its deviation from the
    2.5PN reference, and the monotonicity metrics vs epoch."""
    R = _R(rec)
    phi = _st(R, "phi_grid"); H_ref = _st(R, "H_ref"); E_t = float(_st(R, "E_target"))
    all_ep = R["energy_epoch"]
    if epochs is None:
        idx = np.unique(np.linspace(0, len(all_ep) - 1, min(n_default, len(all_ep))).round().astype(int))
    else:
        idx = np.unique([int(np.argmin(np.abs(all_ep - e))) for e in epochs])
    has_ref = H_ref is not None and H_ref.size > 0

    fig, axes = plt.subplots(1, 3 if has_ref else 2, figsize=(18 if has_ref else 12, 5))
    cols = cm.viridis(np.linspace(0, 0.9, len(idx)))
    for c, i in zip(cols, idx):
        stg = _STAGE_NAMES.get(int(R["energy_stage"][i]), "")
        lab = f"ep {int(all_ep[i]):,}" + (" (post-Stage1)" if int(R["energy_stage"][i]) == 1 else "")
        axes[0].plot(phi, R["energy_H"][i], color=c, lw=1.4, label=lab)
        if has_ref:
            axes[1].plot(phi, R["energy_H"][i] - H_ref, color=c, lw=1.4, label=lab)
    if include_final and "final_best_H" in R:
        axes[0].plot(phi, R["final_best_H"], color="crimson", lw=2, label="final (best ckpt)")
        if has_ref:
            axes[1].plot(phi, R["final_best_H"] - H_ref, color="crimson", lw=2, label="final (best ckpt)")
    if has_ref:
        axes[0].plot(phi, H_ref, color="k", ls="--", lw=1.5, label="2.5PN ref")
        axes[1].axhline(0, color="k", ls="--", lw=1)
        axes[1].set_title("H_PINN(φ) − H_ref(φ)"); axes[1].set_xlabel("φ [rad]")
        axes[1].grid(alpha=.3); axes[1].legend(fontsize=7)
    axes[0].axhline(E_t, color="steelblue", ls=":", lw=1.2, label=f"E* = {E_t:.4f}")
    axes[0].set_title("Energy H(φ) across training"); axes[0].set_xlabel("φ [rad]"); axes[0].set_ylabel("H")
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=7)

    ax = axes[-1]
    ax.plot(all_ep, R["energy_dE_tot"], "o-", color="crimson", ms=3, label="ΔH = H(φ_max) − H(0)")
    if has_ref:
        ax.axhline(H_ref[-1] - H_ref[0], color="seagreen", ls="--", label="ΔH reference")
    ax.set_xlabel("epoch"); ax.set_ylabel("ΔH"); ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot(all_ep, 100 * R["energy_frac_up"], "s--", color="darkorange", ms=3, label="rising steps %")
    ax2.axhline(5, color="darkorange", ls=":", lw=1); ax2.set_ylabel("rising steps [%]")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
    ax.set_title("Energy-loss diagnostics vs epoch")
    fig.suptitle(f"{_st(R, 'label')} [{_st(R, 'tag')}] — energy evolution", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if save_dir:
        names = ["H_vs_phi", "H_minus_ref", "diagnostics"] if has_ref else ["H_vs_phi", "diagnostics"]
        paths = _save_panels_separately(fig, axes, names, save_dir, "energy_evolution", dpi=dpi)
        print(f"  saved {len(paths)} panel PNGs to {save_dir}/")
    return fig
