from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.config import DATA_DIR, MODELS_DIR, DEFAULT_DATA_FILE_BATCH_SIZE, DEFAULT_STRIDE, DEFAULT_PRE_N, DEFAULT_PRERUNS, DEFAULT_SEED, DEFAULT_MAX_FILES, DEFAULT_STOP_AFTER_EPOCH, DEFAULT_MAX_EPOCHS
import torch

id = "11"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # to optimize training speed
model = Backup.load_data_and_build_model(DATA_DIR / f"data_3", device=device, batchsize=DEFAULT_DATA_FILE_BATCH_SIZE, stride=DEFAULT_STRIDE, pre_N=DEFAULT_PRE_N, preruns=DEFAULT_PRERUNS, seed=DEFAULT_SEED, max_files=50)
Backup.train_model_with_backups(model, stop_after_epochs=DEFAULT_STOP_AFTER_EPOCH, max_epochs=DEFAULT_MAX_EPOCHS, directory=MODELS_DIR / f"training_{id}")