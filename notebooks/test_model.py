from sbi_particle_physics.objects.model import Model
import torch
import numpy as np
from sbi_particle_physics.objects.normalizer import Normalizer

device = "cpu" # this small test works on cpu
n_points = 5
n_samples = 5
model = Model(device, n_points)

model.set_prior_basic([3], [5])
model.set_simulator(stride=2, pre_N=2, preruns=2)

raw_data, raw_parameters = model.simulate_raw_data(n_samples=n_samples, n_points=n_points)
model.set_normalizer_with_data(raw_data=raw_data)
model.build_default()
data = model.normalizer.normalize_data(raw_data)
parameters = model.normalizer.normalize_parameters(raw_parameters)
model.append_data(data, parameters)
model.train(max_num_epochs=2, stop_after_epochs=1)
print("All done")

print("normalized stats", data.mean(dim=(0,1)), data.std(dim=(0,1)))