import torch
import numpy as np
from model import Model
from plotter import Plotter
from backup_manager import BackupManager
from diagnostics import Diagnostics

model = BackupManager.load_model_basic(directory="models/training_2")

files = BackupManager.detect_files("data/main")[-10:] # 10 last files
raw_data, raw_parameters, _ = BackupManager.load_data(files)
data = model.normalizer.normalize_data(raw_data)
parameters = model.normalizer.normalize_parameters(raw_parameters)

Diagnostics.lc2st_test(model, x_o=data[-1].unsqueeze(0), n_samples=10000, n_points=1000)