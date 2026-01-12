import torch
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.backup import Backup
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path) # data/data_2
parser.add_argument("start_index", type=int) # 0
parser.add_argument("amount", type=int) # 10
parser.add_argument("n_samples", type=int) # 500
parser.add_argument("n_points", type=int) # 1000
parser.add_argument("prior_low", type=float) # 3
parser.add_argument("prior_high", type=float) # 5
parser.add_argument("stride", type=int) # 100
parser.add_argument("pre_N", type=int) # 1000
parser.add_argument("preruns", type=int) # 10
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(device)
prior_low_raw, prior_high_raw = model.to_tensor([args.prior_low]), model.to_tensor([args.prior_high])
model.set_prior(prior_low_raw, prior_high_raw)
model.set_simulator(stride=args.stride, pre_N=args.pre_N, preruns=args.preruns)
# not necessary to build the nn

Backup.generate_many_data(model, directory=args.directory, start_index=args.start_index, amount=args.amount, n_samples=args.n_samples, n_points=args.n_points, prior_low_raw=prior_low_raw, prior_high_raw=prior_high_raw)
