from xml.parsers.expat import model
import numpy as np
import torch
import os
import glob
from model import Model
import pickle
from tqdm.notebook import tqdm

class BackupManager:

    @staticmethod
    def save_data(file, device, prior_low_raw, prior_high_raw, raw_data, raw_parameters, stride, pre_N, preruns):
        torch.save({
        'raw_data': raw_data,
        'raw_parameters': raw_parameters,
        'metadata': {
            'device': str(device),
            'prior_low_raw': prior_low_raw.cpu().numpy(),
            'prior_high_raw': prior_high_raw.cpu().numpy(),
            'stride': stride, # just for information
            'pre_N': pre_N,
            'preruns': preruns
            }
        }, file)

    @staticmethod
    def load_one_file(file):
        print(f"Loading {os.path.basename(file)}")
        checkpoint = torch.load(file, weights_only=False)
        file_raw_data = checkpoint['raw_data']
        file_raw_parameters = checkpoint['raw_parameters']
        metadata = checkpoint['metadata']
        print(f"(data: {file_raw_data.shape}, params: {file_raw_parameters.shape})")
        return file_raw_data, file_raw_parameters, metadata


    @staticmethod
    def load_data(directory, max_files=None):
        data_files = sorted(glob.glob(f"{directory}/data*.pt"))
        if max_files is not None :
            data_files = data_files[:max_files]
        print(f"#files : {len(data_files)}")
        print(f"files : {[os.path.basename(f) for f in data_files]}\n")
        all_raw_data = []
        all_raw_parameters = []

        metadata = None
        for file_path in tqdm(data_files, desc="Loading files", leave=True):
            checkpoint = torch.load(file_path, weights_only=False)
            if metadata is None: metadata = checkpoint['metadata'] # we keed the metadata from the first file
            file_raw_data = checkpoint['raw_data']
            file_raw_parameters = checkpoint['raw_parameters']
            all_raw_data.append(file_raw_data)
            all_raw_parameters.append(file_raw_parameters)

        raw_data = torch.cat(all_raw_data, dim=0)
        raw_parameters = torch.cat(all_raw_parameters, dim=0)

        print(f"Merged data shape: {raw_data.shape}")
        print(f"Merged parameters shape: {raw_parameters.shape}")
        return raw_data, raw_parameters, metadata
    
    @staticmethod
    def load_data_and_build_model(directory, stride, pre_N, preruns, seed=None, max_files=None):
        raw_data, raw_parameters, metadata = BackupManager.load_data(directory, max_files=max_files)

        device = torch.device(metadata['device'])
        model = Model(device, seed)

        prior_low_raw = model.to_tensor(metadata['prior_low_raw'])
        prior_high_raw = model.to_tensor(metadata['prior_high_raw'])
        model.set_prior(prior_low_raw, prior_high_raw)

        model.set_simulator(stride, pre_N, preruns)

        model.set_normalizer(raw_data, raw_parameters)
        data = model.normalizer.normalize_data(raw_data)
        parameters = model.normalizer.normalize_parameters(raw_parameters)

        model.build_default() 

        return model, data, parameters, raw_data, raw_parameters
    
    @staticmethod
    def save_model(model, file):
        save_dict = {
            'device': model.device,
            'rng': model.rng,
            'prior': model.prior, 
            'stride': model.simulator.stride, # the simulator is not pickleable because of eos
            'pre_N': model.simulator.pre_N,
            'preruns': model.simulator.preruns,
            'normalizer': model.normalizer,
            'posterior': model.posterior,
            'neural_network': model.neural_network
        }
        with open(file, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Model saved to {file}")

    @staticmethod
    def load_model(file):
        with open(file, 'rb') as f:
            save_dict = pickle.load(f)
        device = torch.device(save_dict['device']) # todo marche ?
        model = Model(device)
        model.rng = save_dict['rng']
        model.prior = save_dict['prior']
        stride, pre_N, preruns = save_dict['stride'], save_dict['pre_N'], save_dict['preruns']
        model.set_simulator(stride, pre_N, preruns)
        model.normalizer = save_dict['normalizer']
        model.posterior = save_dict['posterior']
        model.neural_network = save_dict['neural_network']
        print(f"Model loaded from {file}")
        return model, device
    
    @staticmethod
    def build_everything_fast(device, n_samples, n_points, prior_low_raw, prior_high_raw, stride, pre_N, preruns, seed=None):
        model = Model(device, seed)
        model.set_prior(model.to_tensor(prior_low_raw), model.to_tensor(prior_high_raw))
        model.set_simulator(stride, pre_N, preruns)
        raw_data, raw_parameters = model.simulate_raw_data(n_samples, n_points)
        model.set_normalizer(raw_data, raw_parameters)
        data = model.normalizer.normalize_data(raw_data)
        parameters = model.normalizer.normalize_parameters(raw_parameters)
        model.build_default()
        return model, data, parameters, raw_data, raw_parameters