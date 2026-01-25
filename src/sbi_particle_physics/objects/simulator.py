import torch
from torch import Tensor
import eos
import numpy as np
from tqdm.notebook import tqdm
import logging
from matplotlib.pylab import RandomState
from sbi_particle_physics.config import EOS_KINEMATICS, EOS_OPTIONS, EOS_DECAY, EOS_PARAMETER, IMPERFECTIONS_OVERSAMPLE_FACTOR, IMPERFECTIONS_MAX_TRIES
from sbi_particle_physics.objects.imperfections import Imperfections

class Simulator:
    """
    Simulate data with given parameters

    Deal only with un-normalized ("raw") data and parameters
    """


    def __init__(self, device : torch.device, rng : RandomState, stride : int, pre_N : int, preruns : int):
        self.device = device
        self.stride : int = stride
        self.pre_N : int = pre_N
        self.preruns : int = preruns
        self.rng : RandomState = rng

        self.eos_kinematics = eos.Kinematics(EOS_KINEMATICS)
        self.eos_options = eos.Options(EOS_OPTIONS)

        self.eos_parameters = eos.Parameters()
        
        self.distributions = eos.SignalPDF.make(
            EOS_DECAY,
            self.eos_parameters, # arbitrary value
            self.eos_kinematics,
            self.eos_options
        )

        eos.logger.setLevel(logging.WARNING) # or INFO to get the details
        #handler = logging.StreamHandler(stream=sys.stdout)
        #eos.logger.addHandler(handler)

        self.imperfections : Imperfections | None = None

    def to_tensor(self, x, dtype=torch.float32) -> Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)
    
    def set_imperfections(self, imperfections):
        self.imperfections = imperfections

    def simulate_a_sample(self, raw_parameter: Tensor, n_points: int) -> Tensor:
        self.set_eos_parameter(raw_parameter)
        collected = []
        n_target = n_points
        n_generated = n_points if self.imperfections is None else int(n_points * IMPERFECTIONS_OVERSAMPLE_FACTOR) 
        tries = 0

        while len(collected) < n_target:
            if tries >= IMPERFECTIONS_MAX_TRIES:
                raise RuntimeError(f"Could not collect {n_points} accepted events (only {len(collected)})")

            raw_sample, _ = self.distribution.sample_mcmc(
                N=n_generated,
                stride=self.stride,
                pre_N=self.pre_N,
                preruns=self.preruns,
                rng=self.rng,
            )
            x = self.to_tensor(raw_sample)

            if self.imperfection is not None:
                x = self.imperfection.apply(x)

            collected.append(x)
            collected = [torch.cat(collected, dim=0)]
            tries += 1

        return collected[0][:n_target]

    def simulate_samples(self, raw_parameters : Tensor, n_points : int) -> Tensor:
        raw_data = []
        for raw_parameter in tqdm(raw_parameters, desc="Simulating samples", leave=False):
            raw_data.append(self.to_tensor(self.simulate_a_sample(raw_parameter, n_points)))
        return torch.stack(raw_data)

    def set_eos_parameter(self, raw_parameter : Tensor):
        self.eos_parameters.set(EOS_PARAMETER, raw_parameter[0].item())
        return self.eos_parameters