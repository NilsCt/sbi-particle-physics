import torch
from torch import Tensor
import numpy as np
from matplotlib.pylab import RandomState
from pathlib import Path
from sbi_particle_physics.config import (
    MKPI,
    BACKGROUND_CTL_P1, 
    BACKGROUND_CTL_P2, 
    BACKGROUND_CTK_P1, 
    BACKGROUND_CTK_P2, 
    BACKGROUND_PHI_P1, 
    BACKGROUND_PHI_P2, 
    BACKGROUND_MB_MIN, 
    BACKGROUND_MB_MAX, 
    BACKGROUND_TAU_BKG_MB, 
    BACKGROUND_FSIG_MB_WINDOW, 
    RESOLUTION_Q2_MIN, 
    RESOLUTION_Q2_MAX, 
    RESOLUTION_Q2_SIGMA_CORE, 
    RESOLUTION_Q2_SIGMA_SLOPE, 
    RESOLUTION_Q2_SIGMA_TAIL, 
    RESOLUTION_Q2_TAIL_FRACTION, 
    RESOLUTION_PHI_SIGMA, 
    RESOLUTION_COSTHETA_SIGMA
)


class Imperfections:
    """
    Apply detector-like imperfections to ideal EOS samples.s
    """

    def __init__(
        self,
        device: torch.device,
        rng: RandomState,

        use_acceptance: bool = True, # flags
        use_resolution: bool = True,
        use_background: bool = True,

        acceptance_coeffs_path : Path | None = None
    ):
        
        self.device : torch.device = device
        self.rng : RandomState = rng

        self.use_acceptance : bool = use_acceptance
        self.use_resolution : bool = use_resolution
        self.use_background : bool = use_background

        self.mkpi : float = MKPI
        self.q2_min : float = RESOLUTION_Q2_MIN
        self.q2_max : float = RESOLUTION_Q2_MAX

        # acceptance
        self.acceptance_coeffs_path : Path | None = None
        self.acceptance_orders : dict | None = None 
        self.acceptance_ranges_dict : dict | None = None 
        self.acceptance_coeffs : Tensor | None = None
        if acceptance_coeffs_path is not None: self._load_coefs(acceptance_coeffs_path)

        # Resolution
        self.resolution_q2_sigma_core : float = RESOLUTION_Q2_SIGMA_CORE
        self.resolution_q2_sigma_tail : float = RESOLUTION_Q2_SIGMA_TAIL
        self.resolution_q2_tail_fraction : float = RESOLUTION_Q2_TAIL_FRACTION
        self.resolution_q2_sigma_slope : float = RESOLUTION_Q2_SIGMA_SLOPE
        self.resolution_cos_theta_sigma : float = RESOLUTION_COSTHETA_SIGMA
        self.resolution_phi_sigma : float = RESOLUTION_PHI_SIGMA

        # background
        self.background_ctl_p1 : float = BACKGROUND_CTL_P1
        self.background_ctl_p2 : float = BACKGROUND_CTL_P2
        self.background_ctk_p1 : float = BACKGROUND_CTK_P1
        self.background_ctk_p2 : float = BACKGROUND_CTK_P2
        self.background_phi_p1 : float = BACKGROUND_PHI_P1
        self.background_phi_p2 : float = BACKGROUND_PHI_P2
        self.background_tau_bkg_mb : float = BACKGROUND_TAU_BKG_MB
        self.background_mb_min : float = BACKGROUND_MB_MIN
        self.background_mb_max : float = BACKGROUND_MB_MAX
        self.background_fsig_mb_window : float = BACKGROUND_FSIG_MB_WINDOW


    def to_tensor(self, x, dtype=torch.float32) -> Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def apply(self, x: Tensor) -> Tensor:
        """
        Apply imperfections to a batch of events.
        The number of events will probably decrease because of acceptance
        """
        if self.use_acceptance:
            x = self._apply_acceptance(x)
        if self.use_resolution:
            x = self._apply_resolution(x)
        if self.use_background:
            x = self._apply_background(x) # background already includes acceptance
        return x

    # Acceptance
    def _rescale(self, x : Tensor | float, xmin : Tensor | float, xmax : Tensor | float): # map to [-1,1]
        x = self.to_tensor(x)
        xmin = self.to_tensor(xmin)
        xmax = self.to_tensor(xmax)
        new = 2 * (x - xmin) / (xmax - xmin) - 1
        return torch.clamp(new, -1.0, 1.0)
    
    @staticmethod
    def _legendre_all(x: Tensor, nmax: int) -> Tensor:
        # calculate Legendre polynomials of order 0,1,...,nmax evaluated at x
        x = x.unsqueeze(-1)
        P0 = torch.ones_like(x)
        if nmax == 0: return P0
        P1 = x
        Ps = [P0, P1]
        for n in range(1, nmax): # Recurrence: P_{n+1} = (2n+1) x P_n - n P_{n-1}
            Pn = Ps[-1]
            Pnm1 = Ps[-2]
            Pnp1 = ((2*n + 1) * x * Pn - n * Pnm1) / (n + 1)
            Ps.append(Pnp1)
        return torch.cat(Ps, dim=-1)

    def _load_coefs(self, path: Path):
        self.acceptance_coeffs_path = path
        with open(path, "r") as f:
            header = f.readline().removeprefix("#").strip()
        header_vals = np.array(header.split(), dtype=float)
        Nm, Nq2, Nctl, Nctk, Nphi  = header_vals[:5].astype(int) #number of coeffs per observable (max order=1) of the legendre polynomials (mKpi, q2, cosθl, cosθK, phi)
        self.acceptance_orders = {
            "mkpi": Nm-1,
            "q2": Nq2-1,
            "ctl": Nctl-1,
            "ctk": Nctk-1,
            "phi": Nphi-1,
        }
        self.acceptance_ranges_dict = {
            "mkpi": (header_vals[5],  header_vals[6]),
            "q2":   (header_vals[7],  header_vals[8]),
            "ctl":  (header_vals[9],  header_vals[10]),
            "ctk":  (header_vals[11], header_vals[12]),
            "phi":  (header_vals[13], header_vals[14]),
        }

        coeffs = np.loadtxt(path)
        expected_size = Nm * Nq2 * Nctl * Nctk * Nphi
        if coeffs.size != expected_size:
            raise ValueError(f"Coefficient size mismatch: got {coeffs.size}, expected {expected_size}")
        coeffs = self.to_tensor(coeffs.reshape(Nm, Nq2, Nctl, Nctk, Nphi))

        mkpi_min, mkpi_max = self.acceptance_ranges_dict["mkpi"]
        x_mkpi = self._rescale(self.mkpi, mkpi_min, mkpi_max)
        Lmkpi = Imperfections._legendre_all(x_mkpi, self.acceptance_orders["mkpi"]).squeeze(0)

        # Contract mkpi dimension: C4[q2, ctl, ctk, phi] = sum_i L_i(mkpi) * C5[i, ...]
        self.acceptance_coeffs = torch.tensordot(Lmkpi, coeffs, dims=([0], [0]))  # -> (Nq2,Nctl,Nctk,Nphi)

    def _smart_legendre_all(self, x : Tensor, code : str) -> Tensor:
        x_min, x_max = self.acceptance_ranges_dict[code]
        x = self._rescale(x, x_min, x_max)
        return Imperfections._legendre_all(x,  self.acceptance_orders[code])

    def _apply_acceptance(self, x: Tensor) -> Tensor:
        q2, ctl, ctk, phi = x.T
        Lq2 = self._smart_legendre_all(q2, "q2")
        Lctl = self._smart_legendre_all(ctl, "ctl")
        Lctk = self._smart_legendre_all(ctk, "ctk")
        Lphi = self._smart_legendre_all(phi, "phi")
        # eps[b] = sum_{j,k,m,n} C4[j,k,m,n] * Lq2[b,j]*Lctl[b,k]*Lctk[b,m]*Lphi[b,n]
        eps = torch.einsum("bj,bk,bm,bn,jkmn->b", Lq2, Lctl, Lctk, Lphi, self.acceptance_coeffs)
        eps = torch.clamp(eps, 0.0, 1.0)
        u = torch.rand(len(x), device=self.device)
        return x[u < eps] # Accept-reject


    # Resolution
    def _apply_resolution(self, x: Tensor) -> Tensor:
        q2, ctl, ctk, phi = x.T
        n = q2.shape[0]
        is_tail = torch.rand(n, device=self.device) < self.resolution_q2_tail_fraction

        base_sigma = torch.where(is_tail, self.resolution_q2_sigma_tail, self.resolution_q2_sigma_core)
        sigma_q2 = base_sigma * torch.clamp(1.0 + self.resolution_q2_sigma_slope * q2, min=0.0) # make it dependent to q^2
        q2 = q2 + sigma_q2 * torch.randn_like(q2)
        q2 = torch.clamp(q2, self.q2_min, self.q2_max)

        ctl = torch.clamp(ctl + self.resolution_cos_theta_sigma * torch.randn_like(ctl), -1.0, 1.0)
        ctk = torch.clamp(ctk + self.resolution_cos_theta_sigma * torch.randn_like(ctk), -1.0, 1.0)
        phi = torch.clamp(phi + self.resolution_phi_sigma * torch.randn_like(phi), -torch.pi, torch.pi)
        return torch.stack([q2, ctl, ctk, phi], dim=1)
    

    # Background
    def _cheb2_weight(self, x: Tensor, p1: float, p2: float) -> Tensor:
        # Calculate Chebyshev polynomial of order 2
        return 1.0 + p1 * x + p2 * (2.0 * x * x - 1.0)

    def _sample_cheb2(self, n: int, p1: float, p2: float) -> Tensor:
        # Generates n points with Chebyshev polynomial of order 2 as a probability density function in [-1,1]
        x_test = torch.linspace(-1, 1, 1000, device=self.device) # estimate max on a grid
        w_test = self._cheb2_weight(x_test, p1, p2)
        wmax = torch.max(torch.clamp(w_test, min=0.0))
        out = []
        need = n
        while need > 0:
            m = int(need * 1.5) + 32
            x = 2.0 * torch.rand(m, device=self.device) - 1.0
            w = self._cheb2_weight(x, p1, p2)
            w = torch.clamp(w, min=0.0)
            u = torch.rand(m, device=self.device) * wmax
            keep = x[u < w]
            out.append(keep)
            got = sum(t.numel() for t in out)
            need = n - got
        return torch.cat(out)[:n]

    def _sample_trunc_exp(self, n: int, tau: float, xmin: float, xmax: float) -> Tensor:
        # Generates n points with exponential probability density function in [x_min, x_max]
        u = torch.rand(n, device=self.device)
        if abs(tau) < 1e-12:
            return xmin + (xmax - xmin) * u
        delta = xmax - xmin
        ed = torch.exp(self.to_tensor(tau * delta))
        return xmin + (1.0 / tau) * torch.log(1.0 + u * (ed - 1.0))

    def _sample_background_events(self, n: int) -> tuple[Tensor, Tensor]:
        # Generates n background events
        q2 = self.q2_min + (self.q2_max - self.q2_min) * torch.rand(n, device=self.device) # q^2 is uniform in [q2_min, q2_max]
        ctl = self._sample_cheb2(n, self.background_ctl_p1, self.background_ctl_p2)
        ctk = self._sample_cheb2(n, self.background_ctk_p1, self.background_ctk_p2)
        phi = self._sample_cheb2(n, self.background_phi_p1, self.background_phi_p2) * torch.pi # Chebyshev gives phi in [-1,1] instead of [-pi, pi]
        mB = self._sample_trunc_exp(n, self.background_tau_bkg_mb, self.background_mb_min, self.background_mb_max)
        x_bkg = torch.stack([q2, ctl, ctk, phi], dim=1)
        return x_bkg, mB

    def _apply_background(self, x: Tensor) -> Tensor:
        n_sig = len(x)
        n_bkg = int(n_sig * (1.0/self.background_fsig_mb_window - 1)) + 1 # crash if 0
        x_bkg, _ = self._sample_background_events(n=n_bkg)
        x_all = torch.cat([x, x_bkg], dim=0) # doesn't replace real events but add new background events (to optimize time)
        perm = torch.randperm(len(x_all), device=self.device)
        x_all = x_all[perm]
        return x_all