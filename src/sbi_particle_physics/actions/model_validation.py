import torch
import numpy as np
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.plotter import Plotter
from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.managers.model_diagnostics import ModelDiagnostics
from sbi_particle_physics.config import MODELS_DIR, DATA_DIR, PLOTS_DIR

subdirectory = PLOTS_DIR / "plots1"

model = Backup.load_model_for_inference_basic(directory=MODELS_DIR / "training_15", device=torch.device("cpu"))

files = Backup.detect_files(DATA_DIR / "data_3")[-5:] # 5 last files
raw_data, raw_parameters, _ = Backup.load_data(files, model.device)
data = model.normalizer.normalize_data(raw_data[:,:model.n_points])
parameters = model.normalizer.normalize_parameters(raw_parameters[:,:model.n_points])

ModelDiagnostics.simulation_based_calibration(model, data[:200], parameters[:200], num_posterior_samples=1000, path=subdirectory / "sbc.png")

ModelDiagnostics.expected_coverage_test(model, data[:200], parameters[:200], num_posterior_samples=1000, path=subdirectory / "ect.png")

ModelDiagnostics.tarp_test(model, data[:200], parameters[:200], num_posterior_samples=1000, path=subdirectory / "tarp.png")

ModelDiagnostics.misspecification_test(model, data[-1002:-2], x_o=data[-1], path=subdirectory / "miss.png")

ModelDiagnostics.misspecification_test_mmd(model, data[-1002:-2], x_o=data[-1], path=subdirectory / "mmmd.png")
# only needs to be between 0.2->0.8 (model is wrong if <0.05)

ModelDiagnostics.many_posteriors(model, parameter_component_index=0, x_min=3, x_max=5, path=subdirectory / "many.png") # component 0 of the parameters (C_9)

ModelDiagnostics.posterior_predictive_checks(model, x_o=data[-1], n_samples=200, n_points=model.n_points, path=subdirectory/ "ppc.png")

n_posterior_samples = 1000
deltas = np.linspace(0.0, 0.3, 15).tolist()
ModelDiagnostics.robustness_to_noise(model, x_o_raw=raw_data[-100:], n_posterior_samples=n_posterior_samples, deltas=deltas, path = subdirectory / "noise")

n_posterior_samples = 1000
ModelDiagnostics.robustness_to_npoints(model, x_o_raw=raw_data[-100:], n_posterior_samples=n_posterior_samples, use_random_subsample=False, number_of_ns=20, path = subdirectory / "npoints")