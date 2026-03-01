from sbi_particle_physics.objects.model import Model
import sbi
from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.config import DATA_DIR

print(sbi.__version__)

device = "cpu" # this small test works on cpu
n_points = 5
n_samples = 5
model = Model(device, n_points)

prior_low_raw = model.to_tensor([3])
prior_high_raw = model.to_tensor([5])
model.set_prior(prior_low_raw, prior_high_raw)
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

Backup.generate_many_data(model, DATA_DIR / "tmp", start_index=0, amount=2, n_samples=n_samples, n_points=n_points, prior_low_raw=prior_low_raw, prior_high_raw=prior_high_raw)

raw_data, raw_parameters, metadata = Backup.load_one_file(DATA_DIR / "tmp" / "data_0.pt", device=device)
print(f"downloaded raw data shape {raw_data.shape}, raw parameters shape {raw_parameters.shape}")

raw_data, raw_parameters, metadata = Backup.load_data([DATA_DIR  / "tmp" / "data_0.pt", DATA_DIR / "tmp" / "data_1.pt"], device=device)
print(f"after BIG DOWNLOAD, downloaded raw data shape {raw_data.shape}, raw parameters shape {raw_parameters.shape}")



