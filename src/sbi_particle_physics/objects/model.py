import torch
from torch import Tensor
import numpy as np
from sbi_particle_physics.objects.simulator import Simulator
from sbi_particle_physics.objects.imperfections import Imperfections
from sbi_particle_physics.objects.normalizer import Normalizer
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.utils import BoxUniform
from matplotlib.pylab import RandomState
from sbi_particle_physics.config import ENCODED_POINT_DIM, DEFAULT_TRIAL_EMBEDDING_DIM, DEFAULT_TRIAL_NUM_LAYERS, DEFAULT_AGGREGATED_NUM_HIDDENS, DEFAULT_AGGREGATED_NUM_LAYERS, DEFAULT_AGGREGATED_OUTPUT_DIM, DEFAULT_NSF_HIDDEN_FEATURES, DEFAULT_NSF_NUM_BINS, DEFAULT_NSF_NUM_TRANSFORMS, DEFAULT_TRIAL_NUM_HIDDENS, DEFAULT_SAMPLE_WITH, DEFAULT_ENCODER_ACTIVATION_FUNCTION, DEFAULT_NSF_ACTIVATION_FUNCTION, DEFAULT_WEIGHT_DECAY, DEFAULT_PRIOR_LOW, DEFAULT_PRIOR_HIGH
from pathlib import Path

# conda activate mlhep
# pip install -e .

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

    def __init__(self, device, n_points : int, seed : int = None):
        self.device = device
        self.seed = np.random.randint(0, 10000) if seed is None else seed
        self.rng : RandomState = np.random.mtrand.RandomState(self.seed)
        self.prior = None
        self.simulator : Simulator = None
        self.normalizer : Normalizer = None
        self.neural_network : NPE = None
        self.posterior = None
        self.training_loss : list[float] = []
        self.validation_loss : list[float] = []
        self.best_val_loss : float = float("inf")
        self.best_val_epoch : int = None
        self.best_val_file : Path = None
        self.epoch : int = 0
        self.data_files_paths : list[dict] = [] # list and not set to keep order (deterministic loading if needed)
        self.n_points : int = n_points

        self.trial_num_layers : int = None
        self.trial_num_hiddens : int = None
        self.trial_embedding_dim : int = None
        self.aggregated_num_layers : int = None
        self.aggregated_num_hiddens : int = None
        self.aggregated_output_dim : int = None
        self.encoder_activation_function : str = None
        self.nsf_hidden_features : int = None
        self.nsf_num_transforms : int = None
        self.nsf_num_bins : int = None
        self.nsf_activation_function : str = None
        self.weight_decay : int = None
        self.model_type = "nsf" # I only implement that for now
        self.z_score_x = "none"
        self.prior_type = "uniform"

        self.round : int = 0 # SNPE
        self.proposals : list = None

    def to_tensor(self, x, dtype=torch.float32) -> Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)
    
    def set_prior(self, raw_low : Tensor | None = None, raw_high : Tensor | None = None):
        # low and high are raw since the normalization has not been done yet when this function is called
        rl = self.to_tensor(DEFAULT_PRIOR_LOW) if raw_low is None else raw_low
        rh = self.to_tensor(DEFAULT_PRIOR_HIGH) if raw_high is None else raw_high
        self.prior = BoxUniform(low=rl, high=rh, device=self.device)
        self.proposals = [self.prior] 

    def set_prior_basic(self, raw_low : list[float] | np.ndarray, raw_high : list[float] | np.ndarray):
        low = self.to_tensor(raw_low)
        high = self.to_tensor(raw_high)
        self.set_prior(low, high)

    def set_simulator(self, use_imperfections: bool = False, imperfections_kwargs: dict | None = None, **simulator_kwargs):
        self.simulator = Simulator(device=self.device, rng=self.rng, **simulator_kwargs)
        if use_imperfections:
            self.simulator.set_imperfections(**(imperfections_kwargs or {}))

    def set_normalizer(self, data_mean : Tensor, data_std : Tensor):
        self.normalizer = Normalizer(self.device, data_mean, data_std)

    def set_normalizer_with_data(self, raw_data : Tensor):
        self.normalizer = Normalizer.create_normalizer(self.device, raw_data)

    @staticmethod
    def _get_activation_function(name : str) -> torch.nn.Module:
        name = name.lower()
        if name == "relu":
            return torch.nn.ReLU()
        elif name == "elu":
            return torch.nn.ELU()
        elif name == "gelu":
            return torch.nn.GELU()
        elif name == "silu":
            return torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function name: {name}")

    def build(self, trial_num_layers : int, trial_num_hiddens : int, trial_embedding_dim : int, 
              aggregated_num_layers : int, aggregated_num_hiddens : int, aggregated_output_dim : int,
              nsf_hidden_features : int, nsf_num_transforms : int, nsf_num_bins : int, encoder_activation_function : str, nsf_activation_function : str, weight_decay : int): 
        
        self.trial_num_layers = trial_num_layers
        self.trial_num_hiddens = trial_num_hiddens
        self.trial_embedding_dim = trial_embedding_dim
        self.aggregated_num_layers = aggregated_num_layers
        self.aggregated_num_hiddens = aggregated_num_hiddens
        self.aggregated_output_dim = aggregated_output_dim
        self.nsf_hidden_features = nsf_hidden_features
        self.nsf_num_transforms = nsf_num_transforms
        self.nsf_num_bins = nsf_num_bins
        self.encoder_activation_function = encoder_activation_function
        self.nsf_activation_function = nsf_activation_function
        self.weight_decay = weight_decay

        single_trial_net = FCEmbedding(
            input_dim=ENCODED_POINT_DIM,
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
            model=self.model_type,
            hidden_features=nsf_hidden_features,
            num_transforms=nsf_num_transforms,
            num_bins=nsf_num_bins,
            embedding_net=embedding_net,
            z_score_x=self.z_score_x,  #  no normalization since I already do that
            activation=Model._get_activation_function(nsf_activation_function)
        )

        self.neural_network = NPE(
            prior=self.prior,
            device=self.device,
            density_estimator=density_estimator
        )
        # I can't change the encoder activation function if I use the sbi built-in classes, so I just store it for reference
        # I will need to implement a custom embedding net later if I want to improve the encoder and change its activation function

    def build_default(self):
        self.build(
            trial_num_layers=DEFAULT_TRIAL_NUM_LAYERS,
            trial_num_hiddens=DEFAULT_TRIAL_NUM_HIDDENS,
            trial_embedding_dim=DEFAULT_TRIAL_EMBEDDING_DIM,
            aggregated_num_layers=DEFAULT_AGGREGATED_NUM_LAYERS,
            aggregated_num_hiddens=DEFAULT_AGGREGATED_NUM_HIDDENS,
            aggregated_output_dim=DEFAULT_AGGREGATED_OUTPUT_DIM,
            nsf_hidden_features=DEFAULT_NSF_HIDDEN_FEATURES, 
            nsf_num_transforms=DEFAULT_NSF_NUM_TRANSFORMS,
            nsf_num_bins=DEFAULT_NSF_NUM_BINS,
            encoder_activation_function=DEFAULT_ENCODER_ACTIVATION_FUNCTION,
            nsf_activation_function=DEFAULT_NSF_ACTIVATION_FUNCTION,
            weight_decay=DEFAULT_WEIGHT_DECAY
        )

    def draw_raw_parameters(self, n_parameters : int, from_prior : bool = False) -> Tensor:
        if from_prior:
            return self.prior.sample((n_parameters,))
        return self.proposals[-1].sample((n_parameters,))
    
    def draw_parameters(self, n_parameters : int, from_prior : bool = False) -> Tensor:
        return self.normalizer.normalize_parameters(self.draw_raw_parameters(n_parameters, from_prior=from_prior))
    
    def simulate_raw_data(self, n_samples : int, n_points : int, from_prior : bool = False) -> tuple[Tensor, Tensor]: # can be used before initializing the normalizer
        raw_parameters = self.draw_raw_parameters(n_samples, from_prior=from_prior)
        return self.simulator.simulate_samples(raw_parameters, n_points), raw_parameters
    
    def simulate_data(self, n_samples : int, n_points : int, from_prior : bool = False) -> tuple[Tensor, Tensor]:
        raw_data, raw_parameters = self.simulate_raw_data(n_samples, n_points, from_prior=from_prior)
        return self.normalizer.normalize_data(raw_data), self.normalizer.normalize_parameters(raw_parameters)
    
    def simulate_data_with_parameters(self, parameters : Tensor, n_points) -> Tensor:
        raw_parameters = self.normalizer.denormalize_parameters(parameters)
        raw_data = self.simulator.simulate_samples(raw_parameters, n_points)
        return self.normalizer.normalize_data(raw_data) 
    
    @staticmethod
    def _extend_unique_paths(self, new: list[Path], proposal_round : int = None):
        prop = proposal_round if proposal_round is not None else self.round
        seen = set(e["path"] for e in self.data_files_paths)
        for p in new:
            if p not in seen:
                self.data_files_paths.append(Model.create_data_dict(p, proposal_round=prop))
                seen.add(p)
    
    @staticmethod
    def create_data_dict(path : Path, proposal_round : int = 0) -> dict:
        return {"path": path, "proposal_round": proposal_round}

    @staticmethod 
    def export_paths(self) -> list[dict]:
        return [{"path" : str(e["path"]), "proposal_round": e["proposal_round"]} for e in self.data_files_paths]

    def append_data(self, data : Tensor, parameters : Tensor, files : list[Path] | None = None, proposal_round : int = None):
        proposal = None
        if proposal_round is not None:
            proposal = self.proposals[proposal_round]
        elif self.round > 0:
            proposal = self.proposals[-1]
        self.neural_network.append_simulations(parameters, data, proposal=proposal)
        if files is not None:
            self._extend_unique_paths(files, proposal_round=proposal_round)

    def train(self, stop_after_epochs : int, max_num_epochs : int, resume_training : bool = False):  
        before = len(self.validation_loss) # todo mettre self.epoch ?
        self.neural_network.train(stop_after_epochs=stop_after_epochs, max_num_epochs=max_num_epochs, resume_training=resume_training, dataloader_kwargs={"num_workers": 0}) # dataloader_kwargs pour éviter de la duplication de mémoire
        self.epoch = self.neural_network.epoch
        new = self.epoch - before
        self.training_loss.extend(self.neural_network.summary["training_loss"][-new:])
        self.validation_loss.extend(self.neural_network.summary["validation_loss"][-new:]) # I store the losses because sbi reset them every time the training is resumed
        self.posterior = self.neural_network.build_posterior(sample_with=DEFAULT_SAMPLE_WITH) 
    # direct : faster but less precise, used for diagnostics
    # rejection : compromise between direct and mcmc, used for the final version
    # mcmc : the most precise, but extremely slow, used for the final version if unlimited time is available

    def SNPE_new_round(self, obs_raw_data : Tensor, sample_with : str = DEFAULT_SAMPLE_WITH):
        self.round += 1
        obs_data = self.normalizer.normalize_data(obs_raw_data)
        new_proposal = self.build_posterior_with(sample_with=sample_with) 
        self.proposals.append(new_proposal.set_default_x(obs_data))

    def build_posterior_with(self, sample_with : str): # 'direct', 'rejection' or 'mcmc'
        return self.neural_network.build_posterior(sample_with=sample_with)

    # draw parameters from the posterior predicted for some observed sample
    def draw_parameters_from_predicted_posterior(self, observed_samples : Tensor, n_parameters : int) -> Tensor: # also works for multiple observed_samples simultaneously
        if len(observed_samples.shape) == 2:  # (n_points, point_dim)
            observed_samples = observed_samples.unsqueeze(0)  # -> (1, n_points, point_dim) add a batch dimension
        return self.posterior.sample_batched((n_parameters,), x=observed_samples).transpose(0,1) # sbi gives it in a weird order

    # used to compare an observed sample with samples produced with parameters drawn from the posterior distribution predicted for the observed sample
    def simulate_data_from_predicted_posterior(self, observed_sample : Tensor, n_samples : int, n_points : int) -> tuple[Tensor, Tensor]:
        parameters = self.draw_parameters_from_predicted_posterior(observed_sample, n_samples).squeeze(0)
        return self.simulate_data_with_parameters(parameters, n_points).squeeze(0), parameters.squeeze(0)
    
    def get_random_true_parameter(self, n_points : int, from_prior : bool = False) -> tuple[Tensor, Tensor]:
        data, parameters = self.simulate_data(n_samples=1, n_points=n_points, from_prior=from_prior)
        return parameters.squeeze(0), data.squeeze(0)
    
    def get_true_parameters_simulations_and_sampled_parameters(self, n_true : int, n_points : int, n_sampled_parameters : int, from_prior : bool = False) -> tuple[Tensor, Tensor, Tensor]:
        observed_data, true_parameters = self.simulate_data(n_samples=n_true, n_points=n_points, from_prior=from_prior)
        sampled_parameters = self.draw_parameters_from_predicted_posterior(observed_samples=observed_data, n_parameters=n_sampled_parameters)
        return true_parameters.squeeze(0), observed_data.squeeze(0), sampled_parameters.squeeze(0)