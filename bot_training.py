from backup_manager import BackupManager

id = "5"

model, _ = BackupManager.load_data_and_build_model(f"data/main2", batchsize=1, stride=100, pre_N=1000, preruns=10, seed=42, max_files=200)
BackupManager.train_model_with_backups(model, stop_after_epochs=60, max_epochs=400, directory=f"models/training_{id}")