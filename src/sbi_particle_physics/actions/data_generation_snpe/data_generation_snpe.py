import argparse
from pathlib import Path
import torch

from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.config import DATA_DIR, DEFAULT_PRERUNS, DEFAULT_PRE_N, DEFAULT_STRIDE, DEFAULT_PRIOR_LOW, DEFAULT_PRIOR_HIGH, DEFAULT_POINTS_PER_SAMPLE, DEFAULT_SAMPLES_PER_FILE, MODELS_DIR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--amount", type=int, required=True)

    parser.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLES_PER_FILE)
    parser.add_argument("--n-points", type=int, default=DEFAULT_POINTS_PER_SAMPLE)

    args = parser.parse_args()
    print("RAW ARGS:", vars(args))

    device = "cpu" # eos not CUDA-compatible
    model = Backup.load_model_for_inference_basic(directory=MODELS_DIR / args.model_dir, device=torch.device(device))

    directory = DATA_DIR / args.data_dir
    print(f"Generating data in {directory}, "f"start={args.start_index}, amount={args.amount}")

    Backup.generate_many_data(model, directory=directory, start_index=args.start_index, amount=args.amount, n_samples=args.n_samples, n_points=args.n_points, prior_low_raw=[-1], prior_high_raw=[-1]) 
    # prior_low_raw and prior_high_raw are just informative, here there value is meaningless

    # THIS ACTION DOES NOT ADD THE FILES TO THE MODEL (just data generation)

if __name__ == "__main__":
    main()
