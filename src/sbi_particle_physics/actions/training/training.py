from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.config import DATA_DIR, MODELS_DIR, DEFAULT_DATA_FILE_BATCH_SIZE, DEFAULT_STRIDE, DEFAULT_PRE_N, DEFAULT_PRERUNS, DEFAULT_SEED, DEFAULT_MAX_FILES, DEFAULT_STOP_AFTER_EPOCH, DEFAULT_MAX_EPOCHS, DEFAULT_POINTS_PER_SAMPLE, PLOTS_DIR
import torch
from sbi_particle_physics.managers.model_diagnostics import ModelDiagnostics
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

    parser.add_argument("--training-id", type=str, required=True) # 29
    parser.add_argument("--data-dir", type=str, required=True) # data_3

    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES) # 400
    parser.add_argument("--batchsize", type=int, default=DEFAULT_DATA_FILE_BATCH_SIZE) # 1
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--pre-n", type=int, default=DEFAULT_PRE_N)
    parser.add_argument("--preruns", type=int, default=DEFAULT_PRERUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--stop-after-epochs", type=int, default=DEFAULT_STOP_AFTER_EPOCH)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)

    parser.add_argument("--points-per-sample", type=int, default=DEFAULT_POINTS_PER_SAMPLE)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diagnostics-device", type=str, default="cpu")

    parser.add_argument("--run-diagnostics", action="store_true")
    parser.add_argument("--n-diagnostic-files", type=int, default=5)

    args = parser.parse_args()

    id = args.training_id
    data_dir = DATA_DIR / args.data_dir
    model_dir = MODELS_DIR / f"training_{id}"
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    diagnostics_device = torch.device(args.diagnostics_device)

    model = Backup.load_data_and_build_model(data_dir, device=device, batchsize=args.batchsize, stride=args.stride, pre_N=args.pre_n, preruns=args.preruns, seed=args.seed, max_files=args.max_files, max_points=args.points_per_sample)
    print(f"Training {id} on device {device}")
    Backup.train_model_with_backups(model, stop_after_epochs=args.stop_after_epochs, max_epochs=args.max_epochs, directory=model_dir)

    if args.run_diagnostics:
        best_backup = Backup.get_best_backup_file(model, model_dir)
        model = None # free up memory?
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        best_model = Backup.load_model_for_inference(file=best_backup, device=diagnostics_device)
        files = Backup.detect_files(data_dir)[-args.n_diagnostic_files:] # x last files
        raw_data, raw_parameters, _ = Backup.load_data(files, best_model.device)
        ModelDiagnostics.do_them_all(best_model, PLOTS_DIR / f"training_{id}", raw_data, raw_parameters) # automatically launches the diagnostics on best backup

# trained nn
# * : need resume training
# * 12 n points 10 000 and many backup files (to check progress as a function of epoch)
# * 13 n points 8000
# * 14 n points 6000
# * 15 n points 4000
# * 16 n points 2000
# * 17 n points 1000
# * 18 n points 800
# 19 n points 500
# 20 n points 300
# 21 n points 150
# * 23 n files 350
# * 25 n files 200
# 26 n files 100
# 27 n files 50
# * 28 GeLU NSF # ca se trouve je me suis trompé et c'est SiLU

# training nn
# 29 SiLU NSF
# * 24 n files 300 a 2001 epoch quand relancer
# 30 normal 

# nn that need to be trained
# 30, ... change architecture ? -> improve encoder

# condor_q nrc25 -hold -af ClusterId ProcId HoldReason


if __name__ == "__main__":
    main()
