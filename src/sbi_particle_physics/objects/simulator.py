import torch
from torch import Tensor
import eos
import numpy as np
from tqdm.notebook import tqdm
import logging
from matplotlib.pylab import RandomState
from sbi_particle_physics.config import EOS_DECAY, EOS_PARAMETER, IMPERFECTIONS_OVERSAMPLE_FACTOR, IMPERFECTIONS_MAX_TRIES, LEPTON, MB_MAX, MB_MIN, MB_SIG_MEAN, MB_SIG_SIGMA, MODEL, QUARK, Q2_MAX, Q2_MIN, DEFAULT_STRIDE, DEFAULT_PRE_N, DEFAULT_PRERUNS
from sbi_particle_physics.objects.imperfections import Imperfections

class Simulator:
    """
    Simulate data with given parameters

    Deal only with un-normalized ("raw") data and parameters
    """


    def __init__(
            self, 
            device : torch.device,
            rng : RandomState,
            stride : int | None = None,
            pre_N : int | None = None,
            preruns : int | None = None, 
            q2_min : float | None = None,
            q2_max : float | None = None,
            mb_min : float | None = None,
            mb_max : float | None = None,
            lepton : str | None = None,
            quark : str | None = None,
            model : str | None = None,
            decay : str | None = None,
        ):
        self.device = device
        self.stride : int = DEFAULT_STRIDE if stride is None else stride
        self.pre_N : int = DEFAULT_PRE_N if pre_N is None else pre_N
        self.preruns : int = DEFAULT_PRERUNS if preruns is None else preruns
        self.q2_min : float = Q2_MIN if q2_min is None else q2_min
        self.q2_max : float = Q2_MAX if q2_max is None else q2_max
        self.mb_min : float = MB_MIN if mb_min is None else mb_min
        self.mb_max : float = MB_MAX if mb_max is None else mb_max
        self.lepton : str = LEPTON if lepton is None else lepton
        self.quark : str = QUARK if quark is None else quark
        self.model : str = MODEL if model is None else model
        self.decay : str = EOS_DECAY if decay is None else decay
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
            # m_B is not generated by EOS but from a Gaussian distribution
            m_B = torch.normal(mean=MB_SIG_MEAN, std=MB_SIG_SIGMA, size=(x.shape[0], 1), device=self.device)
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