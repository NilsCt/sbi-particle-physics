import torch
from torch import Tensor
import numpy as np
from matplotlib.pylab import RandomState
from sbi_particle_physics.config import (
    IMP_ACC_BETA0_MEAN, IMP_ACC_BETA0_STD, IMP_ACC_A_L_STD, IMP_ACC_A_K_STD, IMP_ACC_A_PHI_STD, IMP_ACC_B_PHI_STD, IMP_ACC_A_Q2_STD,
    IMP_Q2_SIGMA_CORE, IMP_Q2_SIGMA_TAIL, IMP_Q2_TAIL_FRACTION, IMP_Q2_SIGMA_SLOPE, IMP_COS_THETA_SIGMA, IMP_PHI_SIGMA,
    IMP_BACKGROUND_FRACTION, IMP_BACKGROUND_Q2_LAMBDA, IMP_Q2_MIN, IMP_Q2_MAX,
)


class Imperfections:
    """
    Apply detector-like imperfections to ideal EOS samples.
    """

    def __init__(
        self,
        device: torch.device,
        rng: RandomState,

        use_acceptance: bool = True, # flags
        use_resolution: bool = True,
        use_background: bool = True,

        # Acceptance
        acc_beta0_mean: float = IMP_ACC_BETA0_MEAN,
        acc_beta0_std: float = IMP_ACC_BETA0_STD,
        acc_a_l_std: float = IMP_ACC_A_L_STD,
        acc_a_k_std: float = IMP_ACC_A_K_STD,
        acc_a_phi_std: float = IMP_ACC_A_PHI_STD,
        acc_b_phi_std: float = IMP_ACC_B_PHI_STD,
        acc_a_q2_std: float = IMP_ACC_A_Q2_STD,

        # Resolution
        q2_sigma_core: float = IMP_Q2_SIGMA_CORE,
        q2_sigma_tail: float = IMP_Q2_SIGMA_TAIL,
        q2_tail_fraction: float = IMP_Q2_TAIL_FRACTION,
        q2_sigma_slope: float = IMP_Q2_SIGMA_SLOPE,
        cos_theta_sigma: float = IMP_COS_THETA_SIGMA,
        phi_sigma: float = IMP_PHI_SIGMA,

        # Background
        background_fraction: float = IMP_BACKGROUND_FRACTION,
        background_q2_lambda: float = IMP_BACKGROUND_Q2_LAMBDA,

        # Phsysical bounds
        q2_min: float = IMP_Q2_MIN,
        q2_max: float = IMP_Q2_MAX,
    ):
        self.device = device
        self.rng = rng

        self.use_acceptance = use_acceptance
        self.use_resolution = use_resolution
        self.use_background = use_background

        self.acc_beta0_mean = acc_beta0_mean
        self.acc_beta0_std = acc_beta0_std
        self.acc_a_l_std = acc_a_l_std
        self.acc_a_k_std = acc_a_k_std
        self.acc_a_phi_std = acc_a_phi_std
        self.acc_b_phi_std = acc_b_phi_std
        self.acc_a_q2_std = acc_a_q2_std

        self.q2_sigma_core = q2_sigma_core
        self.q2_sigma_tail = q2_sigma_tail
        self.q2_tail_fraction = q2_tail_fraction
        self.q2_sigma_slope = q2_sigma_slope
        self.cos_theta_sigma = cos_theta_sigma
        self.phi_sigma = phi_sigma

        self.background_fraction = background_fraction
        self.background_q2_lambda = background_q2_lambda

        self.q2_min = q2_min
        self.q2_max = q2_max


    def apply(self, x: Tensor) -> Tensor:
        """
        Apply imperfections to a batch of events.
        """
        if self.use_acceptance:
            x = self._apply_acceptance(x)
        if self.use_resolution:
            x = self._apply_resolution(x)
        if self.use_background:
            x = self._apply_background(x)
        return x


    def _sample_acceptance_nuisances(self):
        return (
            self.rng.normal(self.acc_beta0_mean, self.acc_beta0_std),
            self.rng.normal(0.0, self.acc_a_l_std),
            self.rng.normal(0.0, self.acc_a_k_std),
            self.rng.normal(0.0, self.acc_a_phi_std),
            self.rng.normal(0.0, self.acc_b_phi_std),
            self.rng.normal(0.0, self.acc_a_q2_std),
        )

    def _apply_acceptance(self, x: Tensor) -> Tensor:
        beta0, a_l, a_k, a_phi, b_phi, a_q2 = self._sample_acceptance_nuisances()
        q2, ctl, ctk, phi = x.T
        q2_centered = q2 - q2.mean()

        score = (
            beta0
            + a_l * (1.5 * ctl**2 - 0.5)
            + a_k * (1.5 * ctk**2 - 0.5)
            + a_phi * torch.cos(phi)
            + b_phi * torch.sin(phi)
            + a_q2 * q2_centered
        )

        epsilon = torch.sigmoid(score)
        u = torch.rand(len(x), device=self.device)
        return x[u < epsilon]


    def _apply_resolution(self, x: Tensor) -> Tensor:
        q2, ctl, ctk, phi = x.T

        # q^2 smearing (q^2-dependent Gaussian mixture)
        is_tail = torch.rand(len(q2), device=self.device) < self.q2_tail_fraction
        base_sigma = torch.where(
            is_tail,
            torch.tensor(self.q2_sigma_tail, device=self.device),
            torch.tensor(self.q2_sigma_core, device=self.device),
        )
        sigma = base_sigma * (1.0 + self.q2_sigma_slope * q2)
        q2 = q2 + sigma * torch.randn_like(q2)
        q2 = torch.clamp(q2, self.q2_min, self.q2_max) # Enforce physical bounds

        # Angular smearing
        ctl = torch.clamp(ctl + self.cos_theta_sigma * torch.randn_like(ctl), -1.0, 1.0)
        ctk = torch.clamp(ctk + self.cos_theta_sigma * torch.randn_like(ctk), -1.0, 1.0)
        phi = phi + self.phi_sigma * torch.randn_like(phi)

        return torch.stack([q2, ctl, ctk, phi], dim=1)


    def _apply_background(self, x: Tensor) -> Tensor:
        n = len(x)
        n_bkg = int(self.background_fraction * n)
        if n_bkg == 0: return x
        background = self._sample_background(n_bkg)
        indices = torch.randperm(n, device=self.device)[:n_bkg]
        x = x.clone()
        x[indices] = background
        return x

    def _sample_background(self, n: int) -> Tensor:
        u = torch.rand(n, device=self.device) # Exponential-like q^2 (smooth, decreasing)
        q2 = -torch.log(1 - u) / self.background_q2_lambda
        q2 = torch.clamp(q2, self.q2_min, self.q2_max)

        ctl = 2.0 * torch.rand(n, device=self.device) - 1.0
        ctk = 2.0 * torch.rand(n, device=self.device) - 1.0
        phi = 2.0 * np.pi * torch.rand(n, device=self.device)

        return torch.stack([q2, ctl, ctk, phi], dim=1)
