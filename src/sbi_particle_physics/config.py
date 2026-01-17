from pathlib import Path
import numpy as np

# Project
PROJECT_NAME = "sbi_particle_physics"
PROJECT_VERSION = "0.5"

DEFAULT_SEED = 42

# Paths / filenaming
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIRECTORY_PATTERN = "data_{id}" # important to keep "_" but the name before can be changed
DATA_FILE_PATTERN = "data_{index}.pt"
MODEL_DIRECTORY_PATTERN = "model_{id}"
MODEL_FILE_PATTERN = "epoch_{epoch}.pt"

# Data / Parameters    
DATA_LABELS = ["$q^2$", r"$\cos \theta_l$", r"$\cos \theta_d$", r"$\phi$"]
ENCODED_DATA_LABELS = ["$q^2$", "$\\cos \\theta_l$", "$\\cos \\theta_d$", "$\\cos \\phi$", "$\\sin \\phi$"]
ENCODED_POINT_DIM = 5 # q^2, \cos \theta_l, \cos \theta_d, \cos \phi, \sin \phi
PARAMETERS_LABEL = ["$C_9$"]
PARAMETERS_DIM = 1

# EOS
EOS_KINEMATICS = {
            's':             2.0,   's_min':             1,       's_max' :            8.0,
            'cos(theta_l)^LHCb':  0.0,  'cos(theta_l)^LHCb_min': -1.0,      'cos(theta_l)^LHCb_max': +1.0,
            'cos(theta_k)^LHCb':  0.0,  'cos(theta_k)^LHCb_min': -1.0,      'cos(theta_k)^LHCb_max': +1.0,
            'phi^LHCb':           0.3,  'phi^LHCb_min':           -1.0*np.pi,      'phi^LHCb_max':           1.0 * np.pi,
}
EOS_OPTIONS = {
            'l': 'mu',
            'q': 'd',
            'model': 'WET',
            'debug': 'false',
            'logging': 'quiet',
            'log-level': 'off',
}
EOS_DECAY = 'B->K^*ll::d^4Gamma@LowRecoil'
EOS_PARAMETER = "b->smumu::Re{c9}"

# Simulator
DEFAULT_PRIOR_LOW = [3]
DEFAULT_PRIOR_HIGH = [5]
DEFAULT_STRIDE = 100
DEFAULT_PRE_N = 1000
DEFAULT_PRERUNS = 10

# Data
DEFAULT_SAMPLES_PER_FILE = 500
DEFAULT_POINTS_PER_SAMPLE = 10000

# Model
DEFAULT_TRIAL_NUM_LAYERS = 2
DEFAULT_TRIAL_NUM_HIDDENS = 64
DEFAULT_TRIAL_EMBEDDING_DIM = 64
DEFAULT_AGGREGATED_NUM_LAYERS = 2
DEFAULT_AGGREGATED_NUM_HIDDENS = 64
DEFAULT_AGGREGATED_OUTPUT_DIM = 128
DEFAULT_NSF_HIDDEN_FEATURES = 128
DEFAULT_NSF_NUM_TRANSFORMS = 10
DEFAULT_NSF_NUM_BINS = 8
DEFAULT_SAMPLE_WITH = "direct"

# Training
DEFAULT_DATA_FILE_BATCH_SIZE = 1
DEFAULT_MAX_FILES = 200
DEFAULT_STOP_AFTER_EPOCH = 100
DEFAULT_MAX_EPOCHS = 400

# Plots
AXIS_FONTSIZE = 21
LEGEND_FONTSIZE = 15
TICK_FONTSIZE = 15 
