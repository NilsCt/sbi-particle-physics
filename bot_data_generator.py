import torch
from model import Model
from backup_manager import BackupManager


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(device)
prior_low_raw, prior_high_raw = model.to_tensor([3]), model.to_tensor([5])
model.set_prior(prior_low_raw, prior_high_raw)
model.set_simulator(stride=100, pre_N=1000, preruns=10)
# not necessary to build the nn

BackupManager.generate_many_data(model, f"data/main2", start_index=350, amount=10, n_samples=500, n_points=1000, prior_low_raw=prior_low_raw, prior_high_raw=prior_high_raw)
