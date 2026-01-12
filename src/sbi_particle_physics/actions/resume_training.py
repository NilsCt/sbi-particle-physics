from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.config import MODELS_DIR, DEFAULT_STOP_AFTER_EPOCH, DEFAULT_MAX_EPOCHS

id = "4"

directory = MODELS_DIR / f"training_{id}"
model = Backup.load_model_basic(directory)
Backup.train_model_with_backups(model, stop_after_epochs=DEFAULT_STOP_AFTER_EPOCH, max_epochs=DEFAULT_MAX_EPOCHS, directory=directory, resume=True, delete_old_backups=False)
# can go past max_epoch to reach at least one backup

# delete_old_backups = True only if I am sure that the new training will improve the performance