import torch
from torch import Tensor
import numpy as np
from simulator import Simulator
from normalizer import Normalizer
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.utils import BoxUniform
from matplotlib.pylab import RandomState


class Model:
    """
    Object containing everything about a sbi session
    
    How to create an instance?
    First initialization with the device and optionnaly a seed
    Second set_prior() and set_simulator()
    Then set_normalizer() or set_normalizer_with_data()
    Finally build() or build_default()

    Then it is possible to add simulations with append_data()
    And to train the neural network with it using train()
    """
    encoded_point_dim = 5

    def __init__(self, device, seed : int = None):
        self.device = device
        if seed is None: seed = np.random.randint(0, 10000)
        self.rng : RandomState = np.random.mtrand.RandomState(seed)
        self.prior = None
        self.simulator : Simulator = None
        self.normalizer : Normalizer = None
        self.neural_network : NPE = None
        self.posterior = None

    def to_tensor(self, x, dtype=torch.float32) -> Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)
    
    def set_prior(self, raw_low : Tensor, raw_high : Tensor):
        # low and high are raw since the normalization has not been done yet when this function is called
        self.prior = BoxUniform(low=raw_low, high=raw_high, device=self.device)

    def set_prior_basic(self, raw_low : list[float], raw_high : list[float]):
        low = self.to_tensor(raw_low)
        high = self.to_tensor(raw_high)
        self.set_prior(low, high)

    def set_simulator(self, stride : int, pre_N : int, preruns : int):
        self.simulator = Simulator(self.device, self.rng, stride, pre_N, preruns)

    def set_normalizer(self, data_mean : float, data_std : float):
        self.normalizer = Normalizer(data_mean, data_std)

    def set_normalizer_with_data(self, raw_data : Tensor):
        self.normalizer = Normalizer.create_normalizer(raw_data)

    def build(self, trial_num_layers : int, trial_num_hiddens : int, trial_embedding_dim : int, 
              aggregated_num_layers : int, aggregated_num_hiddens : int, aggregated_output_dim : int,
              nsf_hidden_features : int, nsf_num_transforms : int, nsf_num_bins : int): 
        
        single_trial_net = FCEmbedding(
            input_dim=Model.encoded_point_dim,
            num_layers=trial_num_layers,
            num_hiddens=trial_num_hiddens,
            output_dim=trial_embedding_dim
        )

        embedding_net = PermutationInvariantEmbedding(
            trial_net=single_trial_net,
            trial_net_output_dim=trial_embedding_dim,
            num_layers=aggregated_num_layers,
            num_hiddens=aggregated_num_hiddens,
            output_dim=aggregated_output_dim
        )

        density_estimator = posterior_nn(
            model='nsf',
            hidden_features=nsf_hidden_features,
            num_transforms=nsf_num_transforms,
            num_bins=nsf_num_bins,
            embedding_net=embedding_net,
            z_score_x='none'  #  no normalization since I already do that
        )

        self.neural_network = NPE(
            prior=self.prior,
            device=self.device,
            density_estimator=density_estimator
        )

    def build_default(self):
        self.build(
            trial_num_layers=2,
            trial_num_hiddens=64,
            trial_embedding_dim=64,
            aggregated_num_layers=2,
            aggregated_num_hiddens=64,
            aggregated_output_dim=128,
            nsf_hidden_features=128, 
            nsf_num_transforms=10,
            nsf_num_bins=8
        )

    def draw_raw_parameters_from_prior(self, n_parameters : int) -> Tensor:
        return self.prior.sample((n_parameters,))
    
    def simulate_raw_data(self, n_samples : int, n_points : int) -> tuple[Tensor, Tensor]: # can be used before initializing the normalizer
        raw_parameters = self.draw_raw_parameters_from_prior(n_samples)
        return self.simulator.simulate_samples(raw_parameters, n_points), raw_parameters
    
    def simulate_data(self, n_samples : int, n_points : int) -> tuple[Tensor, Tensor]:
        raw_data, raw_parameters = self.simulate_raw_data(n_samples, n_points)
        return self.normalizer.normalize_data(raw_data), self.normalizer.normalize_parameters(raw_parameters)
    
    def simulate_data_with_parameters(self, parameters : Tensor, n_points) -> Tensor:
        raw_parameters = self.normalizer.denormalize_parameters(parameters)
        raw_data = self.simulator.simulate_samples(raw_parameters, n_points)
        return self.normalizer.normalize_data(raw_data) 
    
    def append_data(self, data : Tensor, parameters : Tensor):
        self.neural_network.append_simulations(parameters, data)

    def train(self, stop_after_epochs : int, max_num_epochs : int, resume_training : bool = False):  
        self.neural_network.train(stop_after_epochs=stop_after_epochs, max_num_epochs=max_num_epochs, resume_training=resume_training)
        self.posterior = self.neural_network.build_posterior(sample_with='direct') 
    # direct : faster but less precise, used for diagnostics
    # rejection : compromise between direct and mcmc, used for the final version
    # mcmc : the most precise, but extremely slow, used for the final version if unlimited time is available

    def build_posterior_with(self, sample_with : str): # 'direct', 'rejection' or 'mcmc'
        return self.neural_network.build_posterior(sample_with=sample_with)

    # draw parameters from the posterior predicted for some observed sample
    def draw_parameters_from_predicted_posterior(self, observed_sample : Tensor, n_parameters : int) -> Tensor:
        if len(observed_sample.shape) == 2:  # (n_points, point_dim)
            observed_sample = observed_sample.unsqueeze(0)  # -> (1, n_points, point_dim) add a batch dimension
        return self.posterior.sample((n_parameters,), x=observed_sample)

    # used to compare an observed sample with samples produced with parameters drawn from the posterior distribution predicted for the observed sample
    def simulate_data_from_predicted_posterior(self, observed_sample : Tensor, n_samples : int, n_points : int) -> tuple[Tensor, Tensor]:
        parameters = self.draw_parameters_from_predicted_posterior(observed_sample, n_samples)
        return self.simulate_data_with_parameters(parameters, n_points), parameters
    
    def get_random_true_parameter(self, n_points : int) -> tuple[Tensor, Tensor]:
        data, parameters = self.simulate_data(n_samples=1, n_points=n_points)
        return parameters.squeeze(0), data.squeeze(0)
    