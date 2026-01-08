from backup_manager import BackupManager

id = "4"

model = BackupManager.load_model_basic(directory=f"models/training_{id}")
BackupManager.train_model_with_backups(model, stop_after_epochs=100, max_epochs=500, directory=f"models/training_{id}", resume=True, delete_old_backups=False)
# can go past max_epoch to reach at least one backup

# delete_old_backups = True only if I am sure that the new training will improve the performance