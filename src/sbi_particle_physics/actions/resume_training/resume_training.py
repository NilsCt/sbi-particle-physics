from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.config import MODELS_DIR, DEFAULT_STOP_AFTER_EPOCH, DEFAULT_MAX_EPOCHS, PLOTS_DIR, DATA_DIR, DEFAULT_DATA_FILE_BATCH_SIZE
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
    parser.add_argument("--training-id", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True) # used for diagnostics
    parser.add_argument("--stop-after-epochs", type=int, default=DEFAULT_STOP_AFTER_EPOCH)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batchsize", type=int, default=DEFAULT_DATA_FILE_BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diagnostics-device", type=str, default="cpu")
    parser.add_argument("--delete-old-backups", action="store_true")
    parser.add_argument("--run-diagnostics", action="store_true")
    parser.add_argument("--n-diagnostic-files", type=int, default=5)
    args = parser.parse_args()

    id = args.training_id
    directory = MODELS_DIR / f"training_{id}"
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Resuming training {id} on device {device}")

    model = Backup.load_model_for_training_basic(directory, device=device, load_back_data=True, batchsize=args.batchsize) # to resume training on same data
    Backup.train_model_with_backups(model, stop_after_epochs=args.stop_after_epochs, max_epochs=args.max_epochs, directory=directory, resume=True, delete_old_backups=args.delete_old_backups)
    # can go past max_epoch to reach at least one backup

    # delete_old_backups = True only if I am sure that the new training will improve the performance

    if args.run_diagnostics:
        data_dir = DATA_DIR / args.data_dir
        diagnostics_device = torch.device(args.diagnostics_device)
        best_backup = Backup.get_best_backup_file(model, directory)
        model = None # free up memory?
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        best_model = Backup.load_model_for_inference(file=best_backup, device=diagnostics_device)
        files = Backup.detect_files(data_dir)[-args.n_diagnostic_files:] # x last files
        raw_data, raw_parameters, _ = Backup.load_data(files, best_model.device)
        ModelDiagnostics.do_them_all(best_model, PLOTS_DIR / f"training_{id}", raw_data, raw_parameters) # automatically launches the diagnostics on the best backup

if __name__ == "__main__":
    main()