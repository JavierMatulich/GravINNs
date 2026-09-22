"""ParametricAnglePINN: one network N(phi, theta) -> (u, pr, L) for a box of
initial conditions theta = (p0_factor, r0_km, pr0_factor, nu).

Origin: GitHub_parametric.ipynb, cell 2.
"""
import torch
import torch.nn as nn



class ParametricAnglePINN(nn.Module):
    """u(phi,theta), pr(phi,theta), L(phi,theta).

    theta = (p0_factor, r0_km, pr0_factor, nu), each normalised to [-1,1]
    using the supplied ranges. phi is featurised with random Fourier banks
    plus a radial-harmonic bank whose base frequency is a small MLP of theta
    (the radial period varies across parameter space).
    """
    def __init__(self, phi_max, param_ranges, u_lo, u_hi,
                 n_fourier=10, fourier_sigma=3.0,
                 n_fourier_hi=16, fourier_sigma_hi=8.0,
                 n_harmonic=8, hidden=128, depth=5,
                 pr_max=4.0, omega_hidden=32):
        super().__init__()
        self.phi_max = float(phi_max)
        self.n_fourier = int(n_fourier)
        self.n_fourier_hi = int(n_fourier_hi)
        self.n_harmonic = int(n_harmonic)
        # param ranges: dict name -> (lo, hi); order fixed below
        self._pkeys = ("p0_factor", "r0_km", "pr0_factor", "nu")
        lo = torch.tensor([param_ranges[k][0] for k in self._pkeys], dtype=torch.float32)
        hi = torch.tensor([param_ranges[k][1] for k in self._pkeys], dtype=torch.float32)
        self.register_buffer("theta_lo", lo)
        self.register_buffer("theta_hi", hi)
        self.register_buffer("u_lo", torch.tensor(float(u_lo)))
        self.register_buffer("u_hi", torch.tensor(float(u_hi)))
        self.register_buffer("pr_max", torch.tensor(float(pr_max)))
        # phi Fourier banks
        self.register_buffer("B", torch.randn(self.n_fourier)*float(fourier_sigma))
        if self.n_fourier_hi > 0:
            self.register_buffer("B_hi", torch.randn(self.n_fourier_hi)*float(fourier_sigma_hi))
        else:
            self.register_buffer("B_hi", torch.zeros(0))
        # radial-harmonic bank: base frequency = MLP(theta)
        if self.n_harmonic > 0:
            self.register_buffer("harm_n", torch.arange(1, self.n_harmonic+1, dtype=torch.float32))
            self.omega_net = nn.Sequential(
            # TWO hidden layers. With one hidden layer the radial-frequency
            # map omega(theta) underfits on wide boxes: measured max fit err
            # 1.43e-2 (32x1, big box) -> 3.2e-3 (64x2). A 1.5% frequency error
            # over ~18 rad is a 0.27-rad phase slip, which no amount of main
            # training can undo, so this fit error is a hard floor.
            nn.Linear(4, omega_hidden), nn.Tanh(),
            nn.Linear(omega_hidden, omega_hidden), nn.Tanh(),
            nn.Linear(omega_hidden, 1), nn.Softplus())
        # main trunk: phi-features + normalised theta -> (raw_u, raw_pr).
        # (This block was lost in an earlier edit -- without it the class is
        # unconstructable: forward() calls self.net.)
        in_dim = (1 + 2*self.n_fourier + 2*self.n_fourier_hi
                  + 2*self.n_harmonic + 4)
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.zero_()

    # ── theta helpers ──────────────────────────────────────────────────────
    def norm_theta(self, theta):
        """theta: (N,4) raw -> (N,4) in [-1,1]."""
        return 2.0*(theta - self.theta_lo)/(self.theta_hi - self.theta_lo + 1e-12) - 1.0

    @staticmethod
    def ic_from_theta(theta):
        """theta: (N,4) raw (p0f, r0km, pr0f, nu) -> q0,p0t,pr0,L0,u0 (each (N,))."""
        p0f = theta[:, 0]; r0km = theta[:, 1]; pr0f = theta[:, 2]
        q0 = r0km/200.0
        p0t = p0f/torch.sqrt(q0)
        pr0 = pr0f
        L0 = q0*p0t
        u0 = 1.0/q0
        return q0, p0t, pr0, L0, u0

    @staticmethod
    def band_from_theta(theta):
        """PER-SAMPLE u-band [u_lo, u_hi] from each theta's own orbit geometry.

        WHY PER-SAMPLE: u = u_lo + (u_hi-u_lo)*sigmoid(...). With ONE global
        band covering the whole box, a near-circular orbit (small ecc) occupies
        only a few percent of that band -- measured on this box, the p0=1.0
        orbits used 4% of the global range. The sigmoid then has to land in a
        razor-thin slice, gradients vanish and precision collapses. Deriving
        the band from each theta's own semi-major axis and eccentricity gives
        every orbit 20-55% of its band, which is what the working 4-parameter
        code did.
        """
        p0f = theta[:, 0]; r0km = theta[:, 1]; pr0f = theta[:, 2]
        q0 = r0km/200.0
        p0t = p0f/torch.sqrt(q0)
        L0 = q0*p0t
        u0 = 1.0/q0
        p2 = pr0f**2 + p0t**2
        E = 0.5*p2 - 1.0/q0
        a = torch.where(E < 0, -0.5/E, torch.full_like(E, 1e3))
        ecc = torch.sqrt(torch.clamp(1.0 + 2.0*E*L0**2, min=0.0))
        u_lo = 0.70/(a*(1.0 + ecc) + 1e-6)
        u_hi = 1.60/(a*(1.0 - ecc) + 1e-6)
        u_lo = torch.minimum(u_lo, 0.7*u0)
        u_hi = torch.maximum(u_hi, 1.6*u0)
        return u_lo, u_hi

    # ── features ───────────────────────────────────────────────────────────
    def _features(self, phi, theta):
        pn = phi/self.phi_max                      # (N,1)
        proj_lo = pn*self.B.view(1, -1)
        parts = [pn, torch.sin(proj_lo), torch.cos(proj_lo)]
        if self.n_fourier_hi > 0:
            proj_hi = pn*self.B_hi.view(1, -1)
            parts += [torch.sin(proj_hi), torch.cos(proj_hi)]
        if self.n_harmonic > 0:
            omega = self.omega_net(self.norm_theta(theta))          # (N,1) > 0
            ang = phi * (omega * self.harm_n.view(1, -1))           # (N,n_harmonic)
            parts += [torch.sin(ang), torch.cos(ang)]
        parts.append(self.norm_theta(theta))       # raw normalised theta
        return torch.cat(parts, dim=1)

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, phi, theta):
        """phi: (N,1), theta: (N,4) raw. Returns (N,3): u, pr, L."""
        q0, p0t, pr0, L0, u0 = self.ic_from_theta(theta)
        feat = self._features(phi, theta)
        raw = self.net(feat)
        tau = 0.02*self.phi_max
        gate = 1.0 - torch.exp(-phi/tau)           # (N,1)
        g = gate[:, 0]
        # u via sigmoid band, anchored at u0(theta) at phi=0.
        # The band is PER-SAMPLE (see band_from_theta) so every theta uses a
        # healthy fraction of the sigmoid range instead of a thin slice.
        u_lo_s, u_hi_s = self.band_from_theta(theta)
        u_frac0 = (u0 - u_lo_s)/(u_hi_s - u_lo_s + 1e-8)
        u_frac0 = torch.clamp(u_frac0, 1e-4, 1-1e-4)
        a0_u = torch.log(u_frac0/(1.0 - u_frac0))
        u = u_lo_s + (u_hi_s - u_lo_s)*torch.sigmoid(a0_u + g*raw[:, 0])
        # pr: ADDITIVE hard IC pr(0)=pr0(theta). A global tanh(pr_max) ceiling
        # squashes every theta through one scale; the working 4-parameter code
        # used the unbounded additive form and it is better conditioned here.
        pr = pr0 + g*raw[:, 1]
        # L frozen to L0(theta) (conservative)
        L = L0
        return torch.stack([u, pr, L], dim=1)
