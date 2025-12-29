import torch
import numpy as np
from model import Model
from backup_manager import BackupManager

n_samples = 500
n_points = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(device)

prior_low_raw = model.to_tensor([3])
prior_high_raw = model.to_tensor([5])
model.set_prior(prior_low_raw, prior_high_raw)

stride = 10
pre_N = 200
preruns = 2
model.set_simulator(stride, pre_N, preruns)
# not necessary to build the nn

start_index =130
print("Starting to generate data")
for i in range(100):
    location = f"data/data{start_index + i}.pt"
    raw_data, raw_parameters = model.simulate_raw_data(n_samples, n_points)
    BackupManager.save_data(location, device, prior_low_raw, prior_high_raw, raw_data, raw_parameters, stride, pre_N, preruns)
    print(f"Saved data : raw_data {raw_data.shape}, raw_parameters {raw_parameters.shape}")
    print(f"File : {location}")