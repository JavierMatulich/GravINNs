"""Angle-parametrised orbit PINN.

Origin: ``AngleOrbitPINN4`` and ``_reconstruct_qp4`` in notebook cell 1.

The network maps the orbital angle phi to (u, pr, L) with u = 1/r.  Initial
conditions are enforced exactly through a gate ``1 - exp(-phi/tau)``, u is
confined to a band [u_lo, u_hi] with a sigmoid and pr to [-pr_max, pr_max]
with a tanh.  Two random-Fourier-feature banks (low / high bandwidth) are
used: the high bank resolves the sharp periapsis passage.

Optional radial-harmonic bank (``n_harmonic > 0``, added by the long-orbit
case): K harmonics of a *learnable* radial frequency, exactly periodic in the
radial phase, so capacity no longer has to grow with the number of orbits.
With ``n_harmonic=0`` (the default) the class is identical to the version used
by the near-circular sweep: same parameters, same buffers, same random draws.
"""
import torch
import torch.nn as nn


class AngleOrbitPINN4(nn.Module):
    """u(phi), pr(phi), L(phi), t(phi).
    dissipative=False: L frozen to L0 (recovers 2PN trainer exactly).
    dissipative=True:  L(phi) free to evolve, softplus-positive."""
    def __init__(self, phi_max, u0, pr0, L0, u_lo, u_hi,
                 dissipative=False, n_fourier=8, fourier_sigma=2.0,
                 hidden=64, depth=3,
                 n_fourier_hi=0, fourier_sigma_hi=8.0,
                 pr_max_override=None,
                 n_harmonic=0, omega_radial_init=None):
        # Dual Fourier-bank architecture.
        # B    ~ N(0, fourier_sigma)    captures global/smooth shape.
        # B_hi ~ N(0, fourier_sigma_hi) captures the sharp periapsis passage
        # (periapsis PDE residual is ~100× apoapsis for ecc>0.7 orbits).
        # Set n_fourier_hi=0 to disable and reproduce the original behaviour.
        super().__init__()
        self.phi_max=float(phi_max); self.dissipative=bool(dissipative)
        self.n_fourier=int(n_fourier); self.n_fourier_hi=int(n_fourier_hi)
        for k,v in [("u0",u0),("pr0",pr0),("L0",L0),("u_lo",u_lo),("u_hi",u_hi)]:
            self.register_buffer(k, torch.tensor(float(v), dtype=torch.float32))
        # pr is produced via tanh(...)*pr_max, so pr_max is a HARD CEILING on
        # what the network can ever represent. The old heuristic
        # max(|pr0|*2+1.5, 2.5) only looks at the INITIAL pr0, which badly
        # underestimates the true periapsis |pr| when the orbit starts near
        # apoapsis (small pr0) but is highly eccentric. Pass pr_max_override
        # (measured directly from the integrated reference orbit, the same
        # way u_hi is adaptively widened) to avoid silently clipping pr.
        if pr_max_override is not None:
            pr_mag=float(pr_max_override)
        else:
            pr_mag=max(abs(float(pr0))*2.0+1.5, 2.5)
        self.register_buffer("pr_max", torch.tensor(pr_mag, dtype=torch.float32))
        self.register_buffer("B", torch.randn(self.n_fourier)*float(fourier_sigma))
        if self.n_fourier_hi > 0:
            self.register_buffer("B_hi", torch.randn(self.n_fourier_hi)*float(fourier_sigma_hi))
        else:
            self.register_buffer("B_hi", torch.zeros(0))
        # ── Radial-harmonic bank (fixes phase-error accumulation) ──────── #
        # The orbit is EXACTLY periodic in the radial phase (r repeats every
        # radial period Phi_r), with the precession carried by the slow
        # phi(chi) drift. Random Fourier features sit at fixed wrong
        # frequencies, so representing n_orbits near-repeats needs ~n_orbits×
        # more of them — the source of the phase-accumulation floor.
        # Instead we add K harmonics locked to a LEARNABLE radial frequency
        # omega (initialised at 2π/Phi_r, known from the warm-start orbit's
        # periapsis spacing). These are exactly periodic, so ALL orbits are
        # represented by the SAME K harmonics — capacity O(K), independent
        # of n_orbits, which removes the drift floor. omega is trainable so
        # the network can fine-tune the precession-carrying radial frequency.
        self.n_harmonic=int(n_harmonic)
        if self.n_harmonic > 0:
            import numpy as _np
            _w0 = (2.0*_np.pi/float(omega_radial_init)) if (omega_radial_init not in (None,0)) else 1.0
            # omega stored as a raw learnable log-parameter for positivity
            self.log_omega = nn.Parameter(torch.tensor(float(_np.log(max(_w0,1e-6))), dtype=torch.float32))
            self.register_buffer("harm_n", torch.arange(1, self.n_harmonic+1, dtype=torch.float32))
        in_dim = 1 + 2*self.n_fourier + 2*self.n_fourier_hi + 2*self.n_harmonic
        layers=[nn.Linear(in_dim,hidden), nn.Tanh()]
        for _ in range(depth-1): layers+=[nn.Linear(hidden,hidden), nn.Tanh()]
        layers+=[nn.Linear(hidden,3)]          # u_raw, pr_raw, L_raw
        self.net=nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.zero_()

    def _features(self, phi):
        pn=phi/self.phi_max
        proj_lo=pn*self.B.view(1,-1)
        parts=[pn, torch.sin(proj_lo), torch.cos(proj_lo)]
        if self.n_fourier_hi > 0:
            proj_hi=pn*self.B_hi.view(1,-1)
            parts += [torch.sin(proj_hi), torch.cos(proj_hi)]
        if self.n_harmonic > 0:
            # Harmonics of the LEARNABLE radial frequency omega, applied to
            # the RAW phi (not normalised), so the periodicity is physical.
            omega = torch.exp(self.log_omega)
            ang = phi * (omega * self.harm_n).view(1,-1)   # (N, n_harmonic)
            parts += [torch.sin(ang), torch.cos(ang)]
        return torch.cat(parts, dim=1)

    def forward(self, phi):
        feat=self._features(phi); raw=self.net(feat)
        tau=0.02*self.phi_max
        gate=1.0-torch.exp(-phi/tau)
        u_frac0=(self.u0-self.u_lo)/(self.u_hi-self.u_lo+1e-8)
        a0_u=torch.log(u_frac0/(1.0-u_frac0+1e-6)+1e-6)
        u=self.u_lo+(self.u_hi-self.u_lo)*torch.sigmoid(a0_u+gate[:,0]*raw[:,0])
        a0_pr=torch.atanh(torch.clamp(self.pr0/self.pr_max,-0.999,0.999))
        pr=self.pr_max*torch.tanh(a0_pr+gate[:,0]*raw[:,1])
        if self.dissipative:
            L=self.L0*torch.nn.functional.softplus(
                1.0+gate[:,0]*raw[:,2])/torch.nn.functional.softplus(
                torch.ones_like(raw[:,2]))
        else:
            L=self.L0.expand_as(u)
        return torch.stack([u,pr,L],dim=1)


def reconstruct_qp4(u, pr, L, phi):
    r=1.0/u; cs=torch.cos(phi); sn=torch.sin(phi)
    return r*cs, r*sn, pr*cs-(L*u)*sn, pr*sn+(L*u)*cs


# name kept for backwards compatibility with the notebook
_reconstruct_qp4 = reconstruct_qp4
