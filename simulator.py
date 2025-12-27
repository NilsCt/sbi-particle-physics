import torch
import eos
import numpy as np
from tqdm.notebook import tqdm
import logging


class Simulator:
    # the simulator deals with un-normalized data (raw)

    raw_point_dim = 4
    encoded_point_dim = 5
    parameter_dim = 1

    def __init__(self, device, rng, stride, pre_N, preruns):
        self.device = device
        self.stride = stride
        self.pre_N = pre_N
        self.preruns = preruns
        self.rng = rng

        self.eos_kinematics = eos.Kinematics({
            's':             2.0,   's_min':             1,       's_max' :            8.0,
            'cos(theta_l)^LHCb':  0.0,  'cos(theta_l)^LHCb_min': -1.0,      'cos(theta_l)^LHCb_max': +1.0,
            'cos(theta_k)^LHCb':  0.0,  'cos(theta_k)^LHCb_min': -1.0,      'cos(theta_k)^LHCb_max': +1.0,
            'phi^LHCb':           0.3,  'phi^LHCb_min':           -1.0*np.pi,      'phi^LHCb_max':           1.0 * np.pi,
        })

        self.eos_options = eos.Options({
            'l': 'mu',
            'q': 'd',
            'model': 'WET',
            'debug': 'false',
            'logging': 'quiet',
            'log-level': 'off',
            
        })

        self.eos_parameters = eos.Parameters()
        
        self.distribution = eos.SignalPDF.make(
            'B->K^*ll::d^4Gamma@LowRecoil',
            self.eos_parameters, # arbitrary value
            self.eos_kinematics,
            self.eos_options
        )

        eos.logger.setLevel(logging.WARNING) # INFO pour avoir les info
        #handler = logging.StreamHandler(stream=sys.stdout)
        #eos.logger.addHandler(handler)

    def to_tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def simulate_a_sample(self, raw_parameter, n_points):
        self.set_eos_parameter(raw_parameter)
        raw_sample, _ = self.distribution.sample_mcmc(
            N=n_points,
            stride=self.stride,
            pre_N=self.pre_N,
            preruns=self.preruns,
            rng=self.rng
        )
        return self.to_tensor(raw_sample)

    def simulate_samples(self, raw_parameters, n_points):
        n_samples = raw_parameters.shape[0]
        raw_data = torch.zeros((n_samples, n_points, Simulator.raw_point_dim), dtype=torch.float32, device=self.device)
        for i, raw_parameter in enumerate(tqdm(raw_parameters, desc="Simulating samples", leave=True)):
            raw_data[i] = self.simulate_a_sample(raw_parameter, n_points)
        return raw_data

    def set_eos_parameter(self, raw_parameter):
        self.eos_parameters.set("b->smumu::Re{c9}", raw_parameter[0].item())
        return self.eos_parameters