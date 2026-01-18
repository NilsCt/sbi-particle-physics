from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.config import DATA_DIR, MODELS_DIR, DEFAULT_DATA_FILE_BATCH_SIZE, DEFAULT_STRIDE, DEFAULT_PRE_N, DEFAULT_PRERUNS, DEFAULT_SEED, DEFAULT_MAX_FILES, DEFAULT_STOP_AFTER_EPOCH, DEFAULT_MAX_EPOCHS
import torch

id = "15"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # to optimize training speed
model = Backup.load_data_and_build_model(DATA_DIR / f"data_3", device=device, batchsize=DEFAULT_DATA_FILE_BATCH_SIZE, stride=DEFAULT_STRIDE, pre_N=DEFAULT_PRE_N, preruns=DEFAULT_PRERUNS, seed=DEFAULT_SEED, max_files=400, max_points=4_000)
Backup.train_model_with_backups(model, stop_after_epochs=DEFAULT_STOP_AFTER_EPOCH, max_epochs=DEFAULT_MAX_EPOCHS, directory=MODELS_DIR / f"training_{id}")

# nn trained
# 12 n points 10 000 and many backup files (to check progress as a function of epoch)
# 13 n points 8000
# 14 n points 6000
# 15 n points 4000

# to train these nn
# 16 n points 2000
# 17 n points 1000
# 18 n points 800
# 19 n points 500
# 20 n points 300
# 21 n points 150

# 22 n files 400
# 23 n files 350
# 24 n files 300
# 25 n files 200
# 26 n files 100
# 27 n files 50
