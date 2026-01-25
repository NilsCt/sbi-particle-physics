import torch
import numpy as np
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.plotter import Plotter
from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.managers.model_diagnostics import ModelDiagnostics
from sbi_particle_physics.config import MODELS_DIR, DATA_DIR, PLOTS_DIR
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--plot-dir", type=str, required=True)
    parser.add_argument("--number-data-files", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = Backup.load_model_for_inference_basic(directory=MODELS_DIR / args.model_dir, device=torch.device(args.device))

    files = Backup.detect_files(DATA_DIR / args.data_dir)[-args.number_data_files:] # x last files
    raw_data, raw_parameters, _ = Backup.load_data(files, model.device)

    ModelDiagnostics.do_them_all(model, PLOTS_DIR / args.plot_dir, raw_data, raw_parameters)

if __name__ == "__main__":
    main()