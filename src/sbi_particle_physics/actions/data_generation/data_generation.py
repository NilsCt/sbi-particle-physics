import argparse
from pathlib import Path
import torch

from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.config import DATA_DIR, DEFAULT_PRERUNS, DEFAULT_PRE_N, DEFAULT_STRIDE, DEFAULT_PRIOR_LOW, DEFAULT_PRIOR_HIGH, DEFAULT_POINTS_PER_SAMPLE, DEFAULT_SAMPLES_PER_FILE


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--amount", type=int, required=True)

    parser.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLES_PER_FILE)
    parser.add_argument("--n-points", type=int, default=DEFAULT_POINTS_PER_SAMPLE)

    parser.add_argument("--prior-low", type=float, default=DEFAULT_PRIOR_LOW)
    parser.add_argument("--prior-high", type=float, default=DEFAULT_PRIOR_HIGH)

    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--pre-N", type=int, default=DEFAULT_PRE_N)
    parser.add_argument("--preruns", type=int, default=DEFAULT_PRERUNS)

    args = parser.parse_args()
    print("RAW ARGS:", vars(args))

    device = "cpu"  # eos not CUDA-compatible
    model = Model(device, n_points=args.n_points)

    prior_low_raw = model.to_tensor([args.prior_low])
    prior_high_raw = model.to_tensor([args.prior_high])

    model.set_prior(prior_low_raw, prior_high_raw)
    model.set_simulator(stride=args.stride, pre_N=args.pre_N, preruns=args.preruns)
    directory = DATA_DIR / args.directory
    print(f"Generating data in {directory}, "f"start={args.start_index}, amount={args.amount}")

    Backup.generate_many_data(model, directory=directory, start_index=args.start_index, amount=args.amount, n_samples=args.n_samples, n_points=args.n_points, prior_low_raw=prior_low_raw, prior_high_raw=prior_high_raw)


if __name__ == "__main__":
    main()
