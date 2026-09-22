"""Post-training refinement of a parametric model (Adam and/or L-BFGS).

Origin: GitHub_parametric.ipynb, cell 2.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ..plotting.panels import live_display
from .orbits import C_QD, C_SQ, build_reference_set, integrate_orbit, integrate_orbit_cached, phi_residual_param
from .training import _grid_thetas, _measure_l2, _pretrain_omega_net, _sample_theta, make_anchor_thetas, make_probe_thetas



def refine_parametric_pinn(
        model, param_ranges,
        anchor_thetas,                # the ORIGINAL training anchors
        probe_thetas=None, probe_is_seen=None,
        heatmap_axes=("p0_factor", "r0_km"),
        pde_pn_order=2, data_pn_order=2,
        n_orbits=3, n_pts=2000,
        # ── the refinement set: DENSE new anchors that pin the phase ──
        n_refine_anchors=96,          # extra RK orbits (~0.25 s each)
        refine_seed=7,
        # Re-pinning omega_net on the denser anchor set MOVES the harmonic
        # frequencies out from under a network whose weights were tuned to the
        # OLD ones. Measured: the re-pin ALONE degrades L2 by ~2.5x (1.70e-2 ->
        # 4.32e-2) before a single refinement step runs, and Adam then burns
        # ~15k epochs just crawling back. Worse, with more anchors the small MLP
        # must INTERPOLATE rather than memorise, so its fit error actually GROWS
        # (2.3e-3 at 32 anchors -> 3.7e-3 at 128). Off by default.
        repin_omega=False,
        repin_omega_epochs=4000, repin_omega_lr=5e-3,
        # ── phase 1: Adam with a STRONG supervised term ──
        adam_epochs=40000, adam_lr=3e-4,
        # Supervised points per Adam step. None = full batch over ALL anchors
        # (K*n_pts points every step: fine for K<~100, but 1500 anchors x 2000
        # pts = 3M points/step will OOM or crawl). Set e.g. 8000 to draw a
        # random subsample of the (anchor, point) cloud each step, exactly as
        # train_parametric_pinn's n_sup_batch does -- epoch cost then stays
        # flat no matter how many anchors there are. The estimator matches the
        # full-batch loss to within a few percent.
        n_sup_batch=None,
        w_sup_refine=1.0,             # supervised weight (was alpha_floor=0.05)
        w_pde_refine=1.0,
        n_colloc=8000,
        # ── phase 2: L-BFGS polish ──
        lbfgs_steps=300, lbfgs_lr=0.5,
        lbfgs_colloc=4000, lbfgs_sup=8000,
        # Re-draw the frozen L-BFGS batches every this many steps. L-BFGS needs
        # a FIXED loss for its line search, but 300 steps x 20 inner iterations
        # on ONE frozen sample overfits it: the frozen loss keeps dropping while
        # the true objective rises (observed: probe L2 1.04e-3 -> 1.61e-3 over
        # 300 steps). Refreezing every ~25 steps keeps the line search valid
        # within each block while preventing the overfit. 0 = never refreeze.
        lbfgs_refreeze_every=25,
        heatmap_side=11, plot_every=5000, log_every=1000,
        base_dir="parametric_pinn_refine", device="cpu"):
    """Refine an ALREADY-TRAINED parametric PINN to push L2 below 1e-3.

    WHY NOT JUST L-BFGS ON THE PDE?
    --------------------------------
    The angle-domain residual R_u, R_pr contains no explicit phi -- it is
    AUTONOMOUS. If (u(phi), pr(phi)) solves it, so does (u(phi+d), pr(phi+d))
    for any constant d. Measured on a true orbit: shifting it by 0.1 rad leaves
    l_pde at 6.6e-8 (unchanged) while the L2 error rises to 2.7e-2. The PDE
    residual is COMPLETELY BLIND to a phase offset.

    What breaks the degeneracy is the initial condition at phi=0, which the IC
    gate enforces exactly -- but only pointwise. Over several revolutions the
    network can satisfy u(0)=u0 and still settle onto a phase-shifted branch of
    the same solution family.

    That is exactly the observed failure: at an ANCHOR theta the supervised
    term pins the whole orbit (SEEN L2 ~ 3e-4), while at an UNSEEN theta only
    the phase-blind PDE and the pointwise IC apply (UNSEEN L2 ~ 2.5e-3). The
    PDE residual is already 2.9e-6, so there is nothing left for L-BFGS to win
    there. Driving it lower will NOT fix the phase.

    THE FIX
    -------
    Supervise the phase across the whole box:
      1. Add `n_refine_anchors` NEW RK orbits, Latin-Hypercube spread. They are
         cheap now (~0.25 s each with the symbolic RHS), so ~100 costs ~25 s.
      2. Run Adam at a small LR with a FULL-strength supervised term
         (w_sup_refine=1.0 instead of alpha_floor=0.05 -- the anchors were
         contributing only ~2% of the gradient before).
      3. Finish with L-BFGS on (PDE + supervised). L-BFGS is second-order and
         excels at the final descent once the loss is smooth and near a minimum,
         but it must optimise a loss that SEES phase -- hence the supervised
         term must be present and strong.

    The anchors are ground truth, so weighting them heavily cannot bias the
    solution -- it can only constrain it.

    Returns (model, hist, heat_hist).
    """
    import os
    os.makedirs(base_dir, exist_ok=True)
    dev = device

    # ── 1. dense refinement anchor set (original + new) ────────────────────
    print("Building refinement anchor set ...")
    extra = make_anchor_thetas(param_ranges, n_anchors=n_refine_anchors,
                               include_corners=True, seed=refine_seed,
                               verbose=False)
    seen = {tuple(round(float(x), 9) for x in a) for a in anchor_thetas}
    extra = [t for t in extra if tuple(round(float(x), 9) for x in t) not in seen]
    all_anchors = list(anchor_thetas) + extra
    print(f"  {len(anchor_thetas)} original + {len(extra)} new = "
          f"{len(all_anchors)} supervised orbits")

    metas = [integrate_orbit(*th, data_pn_order, n_orbits, n_pts)
             for th in all_anchors]
    phi_max_global = float(model.phi_max)      # keep the trained domain
    THETA, U, PR, PHI, _ = build_reference_set(
        all_anchors, data_pn_order, n_orbits, n_pts, phi_max_global)
    THETA = THETA.to(dev); U = U.to(dev); PR = PR.to(dev); PHI = PHI.to(dev)
    K = THETA.shape[0]

    if repin_omega:
        # Only do this if you accept the disruption (see the note in the
        # signature). It changes the harmonic frequency at every theta.
        _pretrain_omega_net(model, all_anchors, metas, dev,
                            epochs=repin_omega_epochs, lr=repin_omega_lr)
    else:
        print("  omega_net left UNCHANGED (repin_omega=False): the trained "
              "weights are tuned to its current frequencies.")

    if probe_thetas is None:
        probe_thetas, probe_is_seen = make_probe_thetas(
            param_ranges, anchor_thetas, n_seen=2, n_unseen=2, verbose=False)
    if probe_is_seen is None:
        probe_is_seen = [False]*len(probe_thetas)

    hist = dict(ep=[], l_sup=[], l_pde=[], l_energy=[], l_total=[])
    heat_hist = []

    # live figure: probe orbits (PINN vs RK) + refinement loss
    n_probe = len(probe_thetas)
    fig = plt.figure(figsize=(4*max(n_probe, 2), 7))
    gs = fig.add_gridspec(2, max(n_probe, 2))
    ax_orb = [fig.add_subplot(gs[0, i]) for i in range(n_probe)]
    ax_loss = fig.add_subplot(gs[1, :])
    disp = live_display(fig, None)   # [repo] live in Jupyter only

    def _probe_report(tag, draw=True):
        msg = []
        errs = []
        for th, s in zip(probe_thetas, probe_is_seen):
            e = _measure_l2(model, th, pde_pn_order, n_orbits, n_pts,
                            phi_max_global, dev)
            errs.append(e)
            msg.append(f"{'SEEN' if s else 'UNSEEN'}={e:.2e}")
        print(f"  [{tag}] " + "  ".join(msg))
        if not draw:
            return errs
        for a, th, s, e in zip(ax_orb, probe_thetas, probe_is_seen, errs):
            a.clear()
            o = integrate_orbit_cached(*th, pde_pn_order, n_orbits, n_pts)
            pg = np.linspace(0, min(o["phi_max"], phi_max_global), n_pts)
            with torch.no_grad():
                P = torch.tensor(pg, dtype=torch.float32, device=dev).view(-1, 1)
                T = torch.tensor([list(th)], dtype=torch.float32,
                                 device=dev).repeat(len(pg), 1)
                out = model(P, T).cpu().numpy()
            ur = np.interp(pg, o["phi"], o["u"])
            a.plot(np.cos(pg)/ur, np.sin(pg)/ur, color="0.6", lw=1.2,
                   label=f"{pde_pn_order}PN RK")
            a.plot(np.cos(pg)/out[:, 0], np.sin(pg)/out[:, 0], "r", lw=0.9,
                   label="PINN")
            a.plot(0, 0, "k*", ms=8); a.set_aspect("equal")
            a.set_title(f"{'SEEN (anchor)' if s else 'UNSEEN (held-out)'}"
                        f"   L2={e:.2e}\n"
                        f"p0={th[0]:.2f} r0={th[1]:.0f} "
                        f"prr={th[2]:.2f} nu={th[3]:.2f}",
                        fontsize=7, color=("black" if s else "darkred"))
            a.tick_params(labelsize=6)
        ax_orb[0].legend(fontsize=6, loc="upper right")
        ax_loss.clear()
        if hist["ep"]:
            ax_loss.semilogy(hist["ep"], np.maximum(hist["l_sup"], 1e-14),
                             label="sup")
            ax_loss.semilogy(hist["ep"], np.maximum(hist["l_pde"], 1e-14),
                             label="pde")
            ax_loss.semilogy(hist["ep"], np.maximum(hist["l_total"], 1e-14),
                             "k", label="total")
            ax_loss.legend(fontsize=7)
        ax_loss.set_xlabel("refine epoch", fontsize=8)
        ax_loss.set_title(f"Refinement loss  [{tag}]", fontsize=9)
        ax_loss.tick_params(labelsize=7)
        fig.suptitle("Refinement: probe orbits (PINN vs RK)", fontsize=10)
        fig.tight_layout()
        disp.update(fig)
        return errs

    _probe_report("before refine")

    # ── 2. Adam with a strong supervised term ─────────────────────────────
    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, adam_epochs), eta_min=adam_lr*1e-3)
    PHI_col = PHI.view(1, -1, 1).repeat(K, 1, 1)

    # Pre-flatten the anchor cloud once, so the minibatch path is a cheap
    # gather rather than a rebuild every step.
    Npt_a = PHI.shape[0]
    _phi_all = PHI_col.view(-1, 1)
    _th_all = THETA.view(K, 1, 4).repeat(1, Npt_a, 1).view(-1, 4)
    _u_all = U.reshape(-1); _pr_all = PR.reshape(-1)
    if n_sup_batch is not None and int(n_sup_batch) < K*Npt_a:
        print(f"  supervised minibatch: {int(n_sup_batch)} of {K*Npt_a} "
              f"anchor points per step")
    else:
        print(f"  supervised full batch: {K*Npt_a} anchor points per step"
              + ("" if K*Npt_a < 3e5 else
                 "  [consider n_sup_batch=8000 to cut cost/memory]"))

    pbar = tqdm(range(adam_epochs), desc="refine-Adam")
    for ep in pbar:
        opt.zero_grad()
        # supervised over the anchors (they pin the phase everywhere)
        if n_sup_batch is None or int(n_sup_batch) >= K*Npt_a:
            out = model(_phi_all, _th_all)
            l_sup = (((out[:, 0].view(K, -1) - U)**2).mean()
                     + ((out[:, 1].view(K, -1) - PR)**2).mean())
        else:
            idx = torch.randint(0, K*Npt_a, (int(n_sup_batch),), device=dev)
            out = model(_phi_all[idx], _th_all[idx])
            l_sup = (((out[:, 0] - _u_all[idx])**2).mean()
                     + ((out[:, 1] - _pr_all[idx])**2).mean())
        # PDE over the box (iid theta)
        th_r = _sample_theta(param_ranges, int(n_colloc), dev)
        phi_r = (torch.rand(int(n_colloc), 1, device=dev)
                 * phi_max_global).requires_grad_(True)
        R_u, R_pr, _H = phi_residual_param(
            model, phi_r, th_r, pde_pn_order, C_SQ, C_QD)
        l_pde = (R_u**2).mean() + (R_pr**2).mean()

        loss = w_sup_refine*l_sup + w_pde_refine*l_pde
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if ep % log_every == 0 or ep == adam_epochs-1:
            hist["ep"].append(ep)
            hist["l_sup"].append(float(l_sup))
            hist["l_pde"].append(float(l_pde))
            hist["l_energy"].append(0.0)
            hist["l_total"].append(float(loss))
            pbar.set_postfix(sup=f"{float(l_sup):.1e}", pde=f"{float(l_pde):.1e}",
                             lr=f"{opt.param_groups[0]['lr']:.1e}")
        if plot_every and ep > 0 and ep % plot_every == 0:
            _probe_report(f"adam ep{ep}")

    _probe_report("after Adam")

    # ── 3. L-BFGS polish ──────────────────────────────────────────────────
    # Second-order, full-batch. It needs a FIXED loss (no resampling inside the
    # closure) or the line search is meaningless, so freeze the collocation and
    # supervised subsets once.
    if lbfgs_steps > 0:
        print(f"  L-BFGS polish ({lbfgs_steps} steps) ...")
        torch.manual_seed(0)
        n_sup = min(int(lbfgs_sup), K*PHI.shape[0])
        phi_sup_all = PHI_col.view(-1, 1)
        th_sup_all = THETA.view(K, 1, 4).repeat(1, PHI.shape[0], 1).view(-1, 4)
        u_all = U.view(-1); pr_all = PR.view(-1)

        lb = torch.optim.LBFGS(model.parameters(), lr=lbfgs_lr, max_iter=20,
                               history_size=50, line_search_fn="strong_wolfe")

        def _probe_mean():
            return float(np.mean([_measure_l2(model, th, pde_pn_order,
                                              n_orbits, n_pts,
                                              phi_max_global, dev)
                                  for th in probe_thetas]))

        # The AFTER-ADAM state is candidate #0. Without this, the tracker can
        # only return the least-bad L-BFGS iterate -- even when every single
        # L-BFGS step made things worse (observed), so "keep the best" silently
        # returned a degraded model. Now L-BFGS can never end worse than it
        # started.
        best = _probe_mean()
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_tag = "after-Adam baseline"

        def _draw_batches():
            thf = _sample_theta(param_ranges, int(lbfgs_colloc), dev)
            phf = torch.rand(int(lbfgs_colloc), 1, device=dev)*phi_max_global
            si = torch.randperm(K*PHI.shape[0], device=dev)[:n_sup]
            return (thf, phf, phi_sup_all[si], th_sup_all[si],
                    u_all[si], pr_all[si])

        th_fix, phi_fix, phi_sup, th_sup, u_sup, pr_sup = _draw_batches()

        def closure():
            lb.zero_grad()
            o = model(phi_sup, th_sup)
            ls = ((o[:, 0]-u_sup)**2).mean() + ((o[:, 1]-pr_sup)**2).mean()
            pf = phi_fix.clone().requires_grad_(True)
            ru, rp, _ = phi_residual_param(model, pf, th_fix,
                                           pde_pn_order, C_SQ, C_QD)
            lp = (ru**2).mean() + (rp**2).mean()
            L = w_sup_refine*ls + w_pde_refine*lp
            L.backward()
            return L

        for st in tqdm(range(lbfgs_steps), desc="refine-LBFGS"):
            if lbfgs_refreeze_every and st > 0 and st % lbfgs_refreeze_every == 0:
                th_fix, phi_fix, phi_sup, th_sup, u_sup, pr_sup = _draw_batches()
                lb = torch.optim.LBFGS(model.parameters(), lr=lbfgs_lr,
                                       max_iter=20, history_size=50,
                                       line_search_fn="strong_wolfe")
            lb.step(closure)
            if st % max(1, lbfgs_steps//30) == 0 or st == lbfgs_steps-1:
                m = _probe_mean()
                if m < best:
                    best = m
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                    best_tag = f"L-BFGS step {st}"
        model.load_state_dict(best_state)
        print(f"  L-BFGS kept: {best_tag} (mean probe L2 {best:.3e})")
        _probe_report("after L-BFGS")

    # ── 4. final heatmap over the box ─────────────────────────────────────
    TH_g, v0, v1 = _grid_thetas(param_ranges, heatmap_axes, heatmap_side, dev)
    errs = np.array([
        _measure_l2(model, TH_g[i].cpu(), pde_pn_order, n_orbits, n_pts,
                    phi_max_global, dev) for i in range(TH_g.shape[0])
    ]).reshape(heatmap_side, heatmap_side)
    heat_hist.append((adam_epochs+lbfgs_steps, errs, v0, v1))

    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if hist["ep"]:
        ax[0].semilogy(hist["ep"], np.maximum(hist["l_sup"], 1e-14), label="sup")
        ax[0].semilogy(hist["ep"], np.maximum(hist["l_pde"], 1e-14), label="pde")
        ax[0].semilogy(hist["ep"], np.maximum(hist["l_total"], 1e-14), "k", label="total")
        ax[0].legend(fontsize=7); ax[0].set_xlabel("refine epoch")
        ax[0].set_title("Refinement loss", fontsize=9)
    im = ax[1].imshow(np.log10(errs+1e-12), origin="lower", aspect="auto",
                      extent=[v0[0], v0[-1], v1[0], v1[-1]], cmap="viridis")
    ax[1].set_xlabel(heatmap_axes[0]); ax[1].set_ylabel(heatmap_axes[1])
    ax[1].set_title(f"log10 L2 after refine\nmedian {np.median(errs):.2e}, "
                    f"max {errs.max():.2e}", fontsize=9)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, "refine_summary.png"), dpi=110)
    torch.save(model.state_dict(), os.path.join(base_dir, "param_pinn_refined.pt"))
    # sidecar config so the refined checkpoint is COLD-LOADABLE after a kernel
    # restart (phi_max is not in the state_dict; see load_parametric_model).
    import json as _json
    with open(os.path.join(base_dir, "param_pinn_config.json"), "w") as _f:
        _json.dump(dict(phi_max_global=float(phi_max_global),
                        param_ranges={k: list(v) for k, v in param_ranges.items()},
                        n_orbits=int(n_orbits), n_pts=int(n_pts),
                        pde_pn_order=int(pde_pn_order),
                        data_pn_order=int(data_pn_order)), _f, indent=2)

    frac = float((errs < 1e-3).mean())
    print("-"*66)
    print(f"  heatmap median L2 = {np.median(errs):.3e}   max = {errs.max():.3e}")
    print(f"  fraction of the box below 1e-3: {100*frac:.1f}%")
    print(f"  saved -> {base_dir}/")
    return model, hist, heat_hist
