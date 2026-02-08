from pathlib import Path
import numpy as np

# Project
PROJECT_NAME = "sbi_particle_physics"
PROJECT_VERSION = "0.5"

DEFAULT_SEED = 42

# Paths / filenaming
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REAL_DATA = DATA_DIR / "real_data"
IMPERFECTIONS = PROJECT_ROOT / "imperfections"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
DATA_DIRECTORY_PATTERN = "data_{id}" # important to keep "_" but the name before can be changed
DATA_FILE_PATTERN = "data_{index}.pt"
MODEL_DIRECTORY_PATTERN = "model_{id}"
MODEL_FILE_PATTERN = "epoch_{epoch}.pt"
KEEP_LAST_N_BACKUPS = 2

# Data / Parameters 
C9_SM = 4.27 # theorical C_9
C9 = C9_SM - 0.7 # empirical C_9
C9_uncertainty = 0.2 # 0.2-0.3
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
DEFAULT_SAMPLES_PER_FILE = 50
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
DEFAULT_ENCODER_ACTIVATION_FUNCTION = "ReLU"
DEFAULT_NSF_ACTIVATION_FUNCTION = "ReLU"
DEFAULT_WEIGHT_DECAY = 0

# Training
DEFAULT_DATA_FILE_BATCH_SIZE = 1
DEFAULT_MAX_FILES = 400
DEFAULT_STOP_AFTER_EPOCH = 400
DEFAULT_MAX_EPOCHS = 2000

# Plots
AXIS_FONTSIZE = 21
LEGEND_FONTSIZE = 15
TICK_FONTSIZE = 15 

# Imperfections
IMPERFECTIONS_OVERSAMPLE_FACTOR = 1.5
IMPERFECTIONS_MAX_TRIES = 10

MKPI = 0.892
ACCEPTANCE_COEFFS_PATH = IMPERFECTIONS / "2017_nominal_B0_highq2.dat"

RESOLUTION_Q2_SIGMA_CORE = 0.05 # GeV^2
RESOLUTION_Q2_SIGMA_TAIL = 0.20 # GeV^2
RESOLUTION_Q2_TAIL_FRACTION = 0.10
RESOLUTION_Q2_SIGMA_SLOPE = 0.00 # optional: sigma = base*(1 + slope*q2)
RESOLUTION_COSTHETA_SIGMA = 0.02
RESOLUTION_PHI_SIGMA = 0.02
RESOLUTION_Q2_MIN = 0.9
RESOLUTION_Q2_MAX = 19.1

BACKGROUND_CTL_P1 = 0.47729639827133913
BACKGROUND_CTL_P2 = 0.20711973237496167
BACKGROUND_CTK_P1 = 0.08340808277703779
BACKGROUND_CTK_P2 = 0.3354274513791995
BACKGROUND_PHI_P1 = 0.22164874798386383
BACKGROUND_PHI_P2 = 0.06741043820550917
BACKGROUND_TAU_BKG_MB = -5.745 # background mB exponential slope en GeV^-1
BACKGROUND_FSIG_MB_WINDOW = 0.770766 # signal fraction in mB window
BACKGROUND_MB_MIN, BACKGROUND_MB_MAX = 5.170, 5.700 # GeV
BACKGROUND_MB_SIG_MEAN, BACKGROUND_MB_SIG_SIGMA = 5.279, 0.015 # GeV

