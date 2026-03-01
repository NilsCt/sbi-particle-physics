import torch
from torch import Tensor
import eos
import numpy as np
from tqdm.notebook import tqdm
import logging
from matplotlib.pylab import RandomState
from sbi_particle_physics.config import EOS_DECAY, EOS_PARAMETER, IMPERFECTIONS_OVERSAMPLE_FACTOR, IMPERFECTIONS_MAX_TRIES, LEPTON, MB_MAX, MB_MIN, MODEL, QUARK, Q2_MAX, Q2_MIN, DEFAULT_STRIDE, DEFAULT_PRE_N, DEFAULT_PRERUNS, MB_SIG_CORE_FRAC, MB_MASS, MB_SIGMA1, MB_SIGMA2, MB_ALPHA1, MB_ALPHA2, MB_N1, MB_N2
from sbi_particle_physics.objects.imperfections import Imperfections
import math

class Simulator:
    """
    Simulate data with given parameters

    Deal only with un-normalized ("raw") data and parameters
    """


    def __init__(
            self, 
            device : torch.device,
            rng : RandomState,
            stride : int = DEFAULT_STRIDE,
            pre_N : int = DEFAULT_PRE_N,
            preruns : int = DEFAULT_PRERUNS, 
            q2_min : float = Q2_MIN,
            q2_max : float = Q2_MAX,
            mb_min : float = MB_MIN,
            mb_max : float = MB_MAX,
            lepton : str = LEPTON,
            quark : str = QUARK,
            model : str = MODEL,
            decay : str = EOS_DECAY,
            mb_sig_core_frac : float = MB_SIG_CORE_FRAC,
            mb_mass : float = MB_MASS,
            mb_sigma1 : float = MB_SIGMA1,
            mb_alpha1 : float = MB_ALPHA1,
            mb_n1 : float = MB_N1,
            mb_sigma2 : float = MB_SIGMA2,
            mb_alpha2 : float = MB_ALPHA2,
            mb_n2 : float = MB_N2
        ):
        self.device = device
        self.stride : int = stride
        self.pre_N : int = pre_N
        self.preruns : int = preruns
        self.q2_min : float = q2_min
        self.q2_max : float = q2_max
        self.mb_min : float = mb_min
        self.mb_max : float = mb_max
        self.lepton : str = lepton
        self.quark : str = quark
        self.model : str = model
        self.decay : str = decay
        self.mb_sig_core_frac : float = mb_sig_core_frac
        self.mb_mass : float = mb_mass
        self.mb_sigma1 : float = mb_sigma1
        self.mb_n1 : float = mb_n1
        self.mb_alpha1 : float = mb_alpha1
        self.mb_sigma2 : float = mb_sigma2
        self.mb_alpha2 : float = mb_alpha2
        self.mb_n2 : float = mb_n2
        self.rng : RandomState = rng

        self.eos_kinematics = eos.Kinematics(Simulator._get_kinematics(self.q2_min, self.q2_max))
        self.eos_options = eos.Options(Simulator._get_options(self.lepton, self.quark, self.model))

        self.eos_parameters = eos.Parameters()
        
        self.distributions = eos.SignalPDF.make(
            self.decay,
            self.eos_parameters, # arbitrary value
            self.eos_kinematics,
            self.eos_options
        )

        eos.logger.setLevel(logging.WARNING) # or INFO to get the details
        #handler = logging.StreamHandler(stream=sys.stdout)
        #eos.logger.addHandler(handler)

        self.imperfections : Imperfections | None = None

    @staticmethod
    def _get_kinematics(q2_min : float, q2_max : float) -> dict:
        return {
            's': 2.0,   's_min': q2_min, 's_max' : q2_max,
            'cos(theta_l)^LHCb':  0.0,  'cos(theta_l)^LHCb_min': -1.0,      'cos(theta_l)^LHCb_max': +1.0,
            'cos(theta_k)^LHCb':  0.0,  'cos(theta_k)^LHCb_min': -1.0,      'cos(theta_k)^LHCb_max': +1.0,
            'phi^LHCb':           0.3,  'phi^LHCb_min':           -1.0*np.pi,      'phi^LHCb_max':           1.0 * np.pi,
        }
    
    @staticmethod
    def _get_options(lepton : str, quark : str, model : str) -> dict:
        return {
            'l': lepton,
            'q': quark,
            'model': model,
            'debug': 'false',
            'logging': 'quiet',
            'log-level': 'off',
        }
    
    def get_metadata(self, prior_low_raw : Tensor, prior_high_raw : Tensor) -> dict:
        return {
            'device': str(self.device),
            'prior_low_raw': prior_low_raw.cpu().numpy(),
            'prior_high_raw': prior_high_raw.cpu().numpy(),
            'stride': self.stride, # just for information
            'pre_N': self.pre_N,
            'preruns': self.preruns,
            'q2_min': self.q2_min,
            'q2_max': self.q2_max,
            'mb_min': self.mb_min,
            'mb_max': self.mb_max,
            'lepton': self.lepton,
            'quark': self.quark,
            'model': self.model,
        }

    def to_tensor(self, x, dtype=torch.float32) -> Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)
    
    def set_imperfections(self, **imperfections):
        self.imperfections = Imperfections(device=self.device, rng=self.rng, q2_min=self.q2_min, q2_max=self.q2_max, mb_min=self.mb_min, mb_max=self.mb_max, **imperfections)

    def simulate_a_sample(self, raw_parameter: Tensor, n_points: int) -> Tensor:
        self.set_eos_parameter(raw_parameter)
        n_target = int(n_points)
        n_generated = n_target if self.imperfections is None else int(n_target * IMPERFECTIONS_OVERSAMPLE_FACTOR)
        tries = 0
        collected = None  # will become a Tensor of shape (N_collected, D)

        while True:
            n_collected = 0 if collected is None else collected.shape[0]
            if n_collected >= n_target: 
                break
            if tries >= IMPERFECTIONS_MAX_TRIES:
                raise RuntimeError(f"Could not collect {n_target} accepted events (only {n_collected})")

            raw_sample, _ = self.distributions.sample_mcmc(
                N=n_generated,
                stride=self.stride,
                pre_N=self.pre_N,
                preruns=self.preruns,
                rng=self.rng,
            )
            x = self.to_tensor(raw_sample)
            # m_B is not generated by EOS but from a double Crystall Ball pdf
            m_B = self._sample_double_crystal_ball(x.shape[0])
            x = torch.cat([x, m_B], dim=1)

            if self.imperfections is not None:
                x = self.imperfections.apply(x)

            collected = x if collected is None else torch.cat([collected, x], dim=0)
            tries += 1
        return collected[:n_target] # keep only n_target points


    def simulate_samples(self, raw_parameters : Tensor, n_points : int) -> Tensor:
        raw_data = []
        for raw_parameter in tqdm(raw_parameters, desc="Simulating samples", leave=False):
            raw_data.append(self.to_tensor(self.simulate_a_sample(raw_parameter, n_points)))
        return torch.stack(raw_data)

    def set_eos_parameter(self, raw_parameter : Tensor):
        self.eos_parameters.set(EOS_PARAMETER, raw_parameter[0].item())
        return self.eos_parameters
    
    # Simulate m_B
    def _crystal_ball_pdf(self, x: Tensor, mean: float, sigma: float, alpha: float, n: float,) -> Tensor:
        sigma = torch.as_tensor(sigma, device=self.device)
        alpha = torch.as_tensor(alpha, device=self.device)
        n = torch.as_tensor(n, device=self.device)
        t = (x - mean) / sigma
        abs_alpha = torch.abs(alpha)
        A = (n / abs_alpha) ** n * torch.exp(-abs_alpha**2 / 2) # same coefs as the wikipedia page for a Crystal Ball pdf
        B = n / abs_alpha - abs_alpha
        C = (n / abs_alpha) * (1 / (n - 1)) * torch.exp(-abs_alpha**2 / 2)
        D = math.sqrt(math.pi / 2) * (1 + torch.erf(abs_alpha / math.sqrt(2)))
        N = 1.0 / (sigma * (C + D))
        gaussian = torch.exp(-t**2 / 2)
        powerlaw = A * (B - t) ** (-n)
        if alpha >= 0:
            mask = t > -alpha
        else:
            mask = t < -alpha
        pdf = torch.where(mask, gaussian, powerlaw)
        return N * pdf
    
    def _sample_crystal_ball_truncated(self, n: int, mean: float, sigma: float, alpha: float, n_param: float) -> Tensor:
        # maximum at mean
        x_peak = min(max(mean, self.mb_min), self.mb_max) # in case the max is not in mb_min mb_max (but it should not be the case)
        pdf_max = self._crystal_ball_pdf(self.to_tensor([x_peak]), mean, sigma, alpha, n_param).item()
        samples = []
        need = n
        while need > 0:
            m = int(need * 1.3) + 16
            x = torch.empty(m, device=self.device).uniform_(self.mb_min, self.mb_max)
            y = torch.empty(m, device=self.device).uniform_(0, pdf_max)
            pdf = self._crystal_ball_pdf(x, mean, sigma, alpha, n_param)
            accepted = x[y < pdf]
            samples.append(accepted)
            need -= accepted.shape[0]
        return torch.cat(samples)[:n]
    
    def _sample_double_crystal_ball(self, n: int) -> Tensor:
        mask = torch.rand(n, device=self.device) < self.mb_sig_core_frac
        n1_points = mask.sum().item()
        n2_points = n - n1_points
        samples = torch.empty(n, device=self.device)
        if n1_points > 0:
            samples[mask] = self._sample_crystal_ball_truncated(n1_points, self.mb_mass, self.mb_sigma1, self.mb_alpha1, self.mb_n1)
        if n2_points > 0:
            samples[~mask] = self._sample_crystal_ball_truncated(n2_points, self.mb_mass, self.mb_sigma2, self.mb_alpha2, self.mb_n2)
        return samples.unsqueeze(1)