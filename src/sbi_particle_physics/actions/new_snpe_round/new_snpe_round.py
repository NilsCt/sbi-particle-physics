import torch
import numpy as np
from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.config import MODELS_DIR, REAL_DATA
from sbi_particle_physics.managers.real_data import RealData
import argparse

# ===== BATCH-SAFE MATPLOTLIB CONFIG =====
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})
# =======================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--new-model-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = Backup.load_model_for_training_basic(directory=MODELS_DIR / args.model_dir, device=torch.device(args.device))

    raw_data, _ = RealData.load_n_points(REAL_DATA, model.n_points, device=model.device)
    
    model.SNPE_new_round(raw_data)

    name = Backup._epoch_file_path(MODELS_DIR / args.new_model_dir, model.epoch)
    Backup.save_model(model, name ) # doesn't save in the same directory

if __name__ == "__main__":
    main()