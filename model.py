import numpy as np
import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from simulator import Simulator
from normalizer import Normalizer

class Model:
    # Object containing everything we need
    def __init__(self, device, seed, stop_after_epochs):
        self.device = device
        self.rng = np.random.mtrand.RandomState(seed)
        self.stop_after_epochs = stop_after_epochs # souvent 20
        self.prior = None
        self.simulator = None
        self.normalizer = None
        #self.encoder = None
        self.neural_network = None
        self.posterior = None
        self.point_dim = 4
        self.parameter_dim = 1
        self.data_labels = ["$q^2$", r"$\cos \theta_l$", r"$\cos \theta_d$", r"$\phi$"]
        self.parameters_labels = ["$C_9$"]

        # Configuration du réseau de neurones (peut être modifiée avant set_prior)
        self.nsf_hidden_features = 128
        self.nsf_num_transforms = 10
        self.nsf_num_bins = 20
        self.encoder_output_dim = 64

    def to_tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def set_prior(self, raw_low, raw_high):
        # low and high are raw since the normalization has not been done yet when this function is called
        self.prior = BoxUniform(low=raw_low, high=raw_high, device=self.device)

    def set_simulator(self, stride, pre_N, preruns):
        self.simulator = Simulator(self.device, stride, pre_N, preruns, self.rng, self.point_dim, self.parameter_dim)

    def set_normalizer(self, raw_data, raw_parameters):
        self.normalizer = Normalizer(raw_data, raw_parameters)

    def build(self, phi_hidden_dims, rho_hidden_dims, output_dim): 
        self.neural_network = NPE(prior=self.prior, device=self.device, density_estimator="nsf")

    def draw_raw_parameters_from_prior(self, n_parameters):
        return self.prior.sample((n_parameters,))
    
    def simulate_raw_data(self, n_samples, n_points): # used before creating the normalizer
        raw_parameters = self.draw_raw_parameters_from_prior(n_samples)
        return self.simulator.simulate_samples(raw_parameters, n_points), raw_parameters

    def simulate_data(self, n_samples, n_points):
        raw_data, raw_parameters = self.simulate_raw_data(n_samples, n_points)
        return self.normalizer.normalize_data(raw_data), self.normalizer.normalize_parameters(raw_parameters)
    
    def simulate_data_with_parameters(self, parameters, n_points):
        raw_parameters = self.normalizer.denormalize_parameters(parameters)
        raw_data = self.simulator.simulate_samples(raw_parameters, n_points)
        return self.normalizer.normalize_data(raw_data)

    def train(self, data, parameters):
        self.neural_network.append_simulations(parameters, data) # self.neural_network = ... ?
        self.neural_network.train(stop_after_epochs=self.stop_after_epochs)
        self.posterior = self.neural_network.build_posterior(sample_with='mcmc')

    # draw parameters from the posterior predicted for some observed sample
    def draw_parameters_from_predicted_posterior(self, observed_sample, n_parameters):
        if len(observed_sample.shape) == 2:  # If shape is (n_points, point_dim)
          observed_sample = observed_sample.unsqueeze(0)
        return self.posterior.sample((n_parameters,), x=observed_sample)

    # used to compare an observed sample with samples produced with parameters drawn from the posterior distribution predicted for the observed sample
    def simulate_data_from_predicted_posterior(self, observed_sample, n_samples, n_points):
        parameters = self.draw_parameters_from_predicted_posterior(observed_sample, n_samples)
        return self.simulate_data_with_parameters(parameters, n_points), parameters
    
    def get_random_true_parameter(self, n_points):
        raw_parameter = self.draw_raw_parameters_from_prior(1)
        parameter = self.normalizer.normalize_parameters(raw_parameter)
        data = self.simulate_data_with_parameters(parameter, n_points)
        return parameter.squeeze(0), data.squeeze(0)
    