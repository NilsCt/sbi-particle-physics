from backup_manager import BackupManager

id = "4"

model, _ = BackupManager.load_data_and_build_model(f"data/main", batchsize=10, stride=10, pre_N=200, preruns=2, seed=42, max_files=200)
BackupManager.train_model_with_backups(model, stop_after_epochs=60, max_epochs=400, directory=f"models/training_{id}")