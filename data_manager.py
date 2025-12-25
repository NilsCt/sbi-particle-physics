import numpy as np
import torch
import os
import glob
from model import Model

class DataManager:

    @staticmethod
    def save_data(file_path, raw_data, raw_parameters, n_samples, n_points, device, seed, stop_after_epochs, stride, pre_N, preruns, prior_low_raw, prior_high_raw):
        torch.save({
        'raw_data': raw_data,
        'raw_parameters': raw_parameters,
        'metadata': {
            'n_samples': n_samples,
            'n_points': n_points,
            'device': str(device),
            'seed': seed,
            'stride': stride,
            'pre_N': pre_N,
            'preruns': preruns,
            'prior_low_raw': prior_low_raw.cpu().numpy(),
            'prior_high_raw': prior_high_raw.cpu().numpy()
            }
        }, file_path)

    @staticmethod
    def load_data(directory_path):
        data_files = sorted(glob.glob(f"{directory_path}/data*.pt"))

        print(f"#files : {len(data_files)}")
        print(f"files : {[os.path.basename(f) for f in data_files]}\n")
        all_raw_data = []
        all_raw_parameters = []

        for file_path in data_files:
            print(f"Loading {os.path.basename(file_path)}", end=" ")
            checkpoint = torch.load(file_path, weights_only=False)
            file_raw_data = checkpoint['raw_data']
            file_raw_parameters = checkpoint['raw_parameters']
            all_raw_data.append(file_raw_data)
            all_raw_parameters.append(file_raw_parameters)
            print(f"(data: {file_raw_data.shape}, params: {file_raw_parameters.shape})")

        raw_data = torch.cat(all_raw_data, dim=0)
        raw_parameters = torch.cat(all_raw_parameters, dim=0)

        checkpoint = torch.load(data_files[0], weights_only=False)
        metadata = checkpoint['metadata']

        n_samples = raw_data.shape[0]
        print(f"Merged data shape: {raw_data.shape}")
        print(f"Merged parameters shape: {raw_parameters.shape}")
        return raw_data, raw_parameters, n_samples, metadata
    
    @staticmethod
    def load_data_and_init(directory_path):
        raw_data, raw_parameters, n_samples, metadata = DataManager.load_data(directory_path)
        n_points = metadata['n_points']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # should agree with the one used in training
        seed = metadata['seed']
        print("seed:", seed)

        stop_after_epochs = 1
        model = Model(device, seed, stop_after_epochs)

        prior_low_raw = model.to_tensor(metadata['prior_low_raw'])
        prior_high_raw = model.to_tensor(metadata['prior_high_raw'])
        model.set_prior(prior_low_raw, prior_high_raw)

        stride = metadata['stride']
        pre_N = metadata['pre_N']
        preruns = metadata['preruns']
        model.set_simulator(stride, pre_N, preruns)

        model.set_normalizer(raw_data, raw_parameters)
        data = model.normalizer.normalize_data(raw_data)
        parameters = model.normalizer.normalize_parameters(raw_parameters)

        model.build_default() 

        return model, data, parameters, n_samples, n_points, raw_data, raw_parameters