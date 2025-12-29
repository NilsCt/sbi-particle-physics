import torch
import numpy as np
from model import Model
from ploter import Ploter
from backup_manager import BackupManager

model, data, parameters, raw_data, raw_parameters = BackupManager.load_data_and_build_model("data", stride=10, pre_N=200, preruns=2, seed=42)
n_samples = data.shape[0]
n_points = data.shape[1]

id = "2"
epochs = 200
save_every = 1
model.train(data, parameters, max_num_epochs=save_every-1)
for epoch in range(2*save_every,epochs,save_every): 
    if epoch > 10: save_every = 5
    if epoch > 30: save_every = 10
    model.resume_training(max_num_epochs=epoch-1)
    BackupManager.save_model(model, f"models/training_{id}/epoch_{epoch}.pkl")