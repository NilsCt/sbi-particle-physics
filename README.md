# SBI Particle Physics: Inference for $B \to K^* \ell^+ \ell^-$ decays

Simulation-Based Inference (SBI) for Wilson coefficients in rare B-meson decays using neural posterior estimation.

## What is SBI and Why Use It?

**Simulation-Based Inference** (SBI) enables Bayesian parameter estimation when the likelihood function is intractable. Traditional methods require computing p(data|parameters) analytically, which is impractical for complex particle physics simulations involving MCMC sampling from differential decay distributions.

SBI bypasses this limitation by learning the posterior p(parameters|data) directly from simulated data using neural networks. This is particularly valuable for rare decay analyses where:
- The forward model (EOS theory calculations + MCMC sampling) is expensive but simulator-based
- The mapping from Wilson coefficients to 4D kinematic distributions is highly nonlinear
- Rapid inference on experimental data is needed without rerunning expensive simulations

**Why Now?** Recent advances in normalizing flows (neural spline flows) and permutation-invariant architectures make SBI practical for high-dimensional unordered data like particle collision events. The `sbi` Python package provides production-ready implementations of these methods.

## Project Objective

Infer the Wilson coefficient $C_9$ from observed $B \to K^* \mu^+ \mu_-$ decay kinematics in the low-recoil region (1 < $q^2$ < 8 GeV²). This coefficient parameterizes new physics contributions to $b\to s\ell \ell$ transitions and is a key target for LHC flavor physics programs.

The pipeline:
1. **Simulate**: Generate decay kinematics from EOS theory for varying $C_9$ values
2. **Train**: Teach a neural network the inverse mapping from kinematics to $C_9$
3. **Infer**: Given observed events, sample from the posterior p($C_9$|observed data)

## Key Classes

### `Model` (model.py)
Central orchestrator for the SBI workflow. Manages the prior distribution, simulator, normalizer, and neural posterior estimator (NPE). Provides methods for training, sampling from the posterior, and posterior predictive checks.

**Key methods:**
- `set_prior()` – Define parameter bounds for $C_9$
- `set_simulator()` – Initialize EOS-based decay simulator
- `set_normalizer()` - Create an object responsible for the normalization and formating of the data and parameters
- `build_default()` – Construct permutation-invariant neural network architecture
- `train()` – Train NPE on simulated ($C_9$, kinematics) pairs

### `Simulator` (simulator.py)
Forward model wrapping the EOS library. Samples 4D decay kinematics $(q^2, \cos \theta_\ell, \cos \theta_K, \phi)$ from the differential decay distribution using Metropolis-Hastings MCMC. Configured with stride, pre-burn-in, and multiple chains for convergence.

### `Normalizer` (normalizer.py)
Transforms data for neural network efficiency. Applies z-score normalization to kinematic variables and encodes the periodic angle $\phi$ as $(\cos \phi, \sin \phi)$ to preserve periodicity. Each event becomes a 5D vector.

### `Backup` (backup.py)
Handles data persistence and model checkpointing. Supports batch loading of large simulated datasets, incremental training with resumption, and model serialization. Critical for managing ~250 data files totaling 125k simulated samples.

### `Diagnostics` (diagnostics.py)
Comprehensive posterior validation suite implementing six calibration tests:
- **SBC**: Simulation-based calibration via rank statistics
- **PPC**: Posterior predictive checks for data reproduction
- **TARP**: Coverage diagnostics for credible intervals
- **LC2ST**: Classifier-based two-sample test for misspecification
- **ECT**: Expected coverage test
- **Misspecification tests**: LogProb and MMD-based approaches

### `Plotter` (plotter.py)
Visualization utilities for loss curves, data distributions, posterior samples, and posterior predictive comparisons.

## Project Structure

```
sbi-particle-physics/
├── Core Modules
│   ├── model.py              # SBI orchestrator (NPE + training)
│   ├── simulator.py          # EOS-based B→K*ll forward model
│   ├── normalizer.py         # Data/parameter preprocessing
│   ├── backup_manager.py     # Checkpointing and batch data loading
│   ├── diagnostics.py        # Posterior calibration tests
│   └── plotter.py            # Visualization tools
│
├── Training Scripts
│   ├── bot_training.py       # Main training loop with checkpoints
│   ├── bot_data_generator.py # Parallel data generation (230+ files)
│   ├── bot_lc2st.py         # LC2ST diagnostic runner
│   └── test_model.py         # Minimal working example
│
├── Data
│   └── data/main/            # Simulated datasets (~125k samples)
│       └── data*.pt          # PyTorch tensors + metadata
│
├── Models
│   └── models/training_*/    # Checkpoints from training sessions
│       └── epoch_*.pkl       # Model state + optimizer
│
└── Notebooks
    ├── data_sanity.ipynb     # Data validation and correlations
    └── model_validator.ipynb # Posterior quality checks
```

## Implemented Strategies

### Neural Architecture
**Permutation-invariant design** for unordered event sets:
1. **Per-event encoder**: FCEmbedding (5D input → 64D per event)
2. **Aggregation**: PermutationInvariantEmbedding (64D × N events → 128D summary)
3. **Posterior**: Neural Spline Flow with 10 transforms and 8 bins

This architecture respects the exchangeability of events while capturing correlations across the sample.

### Training Strategy
- **Sequential amortized inference**: Train once on a wide parameter range, then reuse for any observation
- **Batch processing**: Load simulated data in batches (default: 10 files at a time) for memory efficiency
- **Early stopping**: Monitor validation loss with configurable patience
- **Checkpointing**: Save model state every 5-10 epochs for fault tolerance

### Data Generation
- **MCMC configuration**: Stride=10, pre_N=200, preruns=2 for decorrelated samples
- **Prior**: Uniform $C_9$ ∈ [3, 5]
- **Sample size**: 1000 events per parameter value (standard for LHCb analyses)

### Validation
- **Multi-test approach**: Combine SBC, TARP, and LC2ST to detect different failure modes (miscalibration, overconfidence, misspecification)
- **Posterior predictive checks**: Verify the posterior can reproduce observed data distributions
- **Data-driven diagnostics**: Use held-out simulated data for calibration tests before applying to real observations

## Dependencies

- `sbi` – Simulation-based inference framework (NPE implementation)
- `torch` – Neural network backend with GPU support
- `eos` – Effective operator set calculations for B physics
- `numpy`, `matplotlib`, `tqdm` – Standard scientific Python stack

## Quick Start

**Generate data:**
```python
python bot_data_generator.py  # Creates data/main/data*.pt files
```

**Train model:**
```python
python bot_training.py  # Trains NPE and saves checkpoints to models/training_*/
```

**Run diagnostics:**

**Infer from observation:**

## References

- EOS: https://eos.github.io/
- SBI: https://sbi-dev.github.io/sbi/
