import numpy as np
import torch
from torch import Tensor
from model import Model
from normalizer import Normalizer
import pickle
from tqdm.notebook import tqdm
import glob
from pathlib import Path
from plotter import Plotter


class BackupManager:
    """
    Responsible for everything related to saving and loading data or models
    """

    @staticmethod
    def save_data(file : str, device, prior_low_raw : Tensor, prior_high_raw : Tensor, raw_data : Tensor, raw_parameters : Tensor, stride : int, pre_N : int, preruns : int):
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
    def generate_many_data(model : Model, directory : str, start_index : int, amount : int, n_samples : int, n_points : int, prior_low_raw : Tensor, prior_high_raw : Tensor):
        print("Starting to generate data")
        for i in range(start_index, start_index + amount):
            location = f"{directory}/data{i}.pt"
            raw_data, raw_parameters = model.simulate_raw_data(n_samples, n_points)
            BackupManager.save_data(location, model.device, prior_low_raw, prior_high_raw, raw_data, raw_parameters, model.simulator.stride, model.simulator.pre_N, model.simulator.preruns)


    @staticmethod
    def _extract_id(filepath : str) -> int:
        match = filepath.split("data")[-1].split(".pt")[0]
        return int(match)
    
    @staticmethod
    def detect_files(directory : str) -> list[str]:
        data_files = sorted(glob.glob(f"{directory}/data*.pt"), key=BackupManager._extract_id)
        return data_files
    
    
    @staticmethod
    def load_one_file(file : str):
        checkpoint = torch.load(file, weights_only=False)
        file_raw_data = checkpoint['raw_data']
        file_raw_parameters = checkpoint['raw_parameters']
        metadata = checkpoint['metadata']
        return file_raw_data, file_raw_parameters, metadata

    @staticmethod
    def load_data(files : list[str]):
        all_raw_data = []
        all_raw_parameters = []
        metadata = None
        for file in tqdm(files, desc="Loading files", leave=True):
            file_raw_data, file_raw_parameters, file_metadata = BackupManager.load_one_file(file)
            if metadata is None: metadata = file_metadata # we keep the metadata of the first file
            all_raw_data.append(file_raw_data)
            all_raw_parameters.append(file_raw_parameters)

        raw_data = torch.cat(all_raw_data, dim=0)
        raw_parameters = torch.cat(all_raw_parameters, dim=0)
        #print(f"Merged data shape: {raw_data.shape}")
        #print(f"Merged parameters shape: {raw_parameters.shape}")
        return raw_data, raw_parameters, metadata
    
    
    @staticmethod
    def calculate_stats(files : list[str], batchsize : int) -> tuple[float, float]:
        mean = 0
        std = 1
        cursor = 0
        while cursor < len(files):
            selected_files = files[cursor: cursor+batchsize]
            cursor += batchsize
            raw_data, _, _ = BackupManager.load_data(selected_files)
            data_mean, data_std = Normalizer.calculate_stats(raw_data)
            mean += data_mean * len(selected_files)
            std += data_std * len(selected_files)
        return mean / len(files), std / len(files)
    
    @staticmethod
    def load_and_append_data(model : Model, files : list[str], batchsize : int):
        cursor = 0
        while cursor < len(files):
            raw_data, raw_parameters, met = BackupManager.load_data(files[cursor: cursor+batchsize])
            cursor += batchsize
            data = model.normalizer.normalize_data(raw_data)
            parameters = model.normalizer.normalize_parameters(raw_parameters)
            model.append_data(data, parameters)

    @staticmethod
    def load_data_and_build_model(directory : str, batchsize : int, stride : int, pre_N : int, preruns : int, seed : int = None, max_files : int = None) -> tuple[Model, int]:
        # warning: here batchsize corresponds to the number of data files used at a time, not of samples
        # one file contains around 500 samples
        files = BackupManager.detect_files(directory) 
        if max_files is not None: files = files[:max_files]
        if len(files) == 0: raise BaseException("No files found") 
       
        mean, std = BackupManager.calculate_stats(files, batchsize)
        data0, _, metadata = BackupManager.load_one_file(files[0])
        n_points = data0
        device = torch.device(metadata['device'])
        model = Model(device, seed)

        prior_low_raw = model.to_tensor(metadata['prior_low_raw'])
        prior_high_raw = model.to_tensor(metadata['prior_high_raw'])
        model.set_prior(prior_low_raw, prior_high_raw)
        model.set_simulator(stride, pre_N, preruns)
        model.set_normalizer(mean, std)
        model.build_default() 
        BackupManager.load_and_append_data(model, files, batchsize)
        return model, n_points # todo traiter mieux n_points
    

    @staticmethod
    def save_model(model : Model, file : str):
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
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(save_dict, f)
        print(f"Model saved to {file}")

    @staticmethod
    def load_model(file : str) -> Model:
        with Path(file).open("rb") as f:
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
        return model
    
    @staticmethod
    def _extract_epoch(filepath : str) -> int:
        match = filepath.split("epoch_")[-1].split(".pkl")[0]
        return int(match)

    @staticmethod
    def _get_corresponding_file(directory : str, epoch : int | None) -> str:
        files = glob.glob(f"{directory}/epoch_*.pkl")
        if epoch is None:
            return sorted(files, key=BackupManager._extract_epoch)[-1] # last epoch
        else:
            for file in files:
                if BackupManager._extract_epoch(file) == epoch:
                    return file
            raise BaseException(f"No file corresponding to epoch {epoch} in directory {directory}")
        
    @staticmethod
    def load_model_basic(directory : str, epoch : int | None = None) -> Model: # useful method to load more easily a model
        file  = BackupManager._get_corresponding_file(directory, epoch)
        return BackupManager.load_model(file)

    @staticmethod
    def _epochs_step(epochs : int):
        if epochs < 10: return 1 
        elif epochs < 30: return 5 
        else: return 10

    @staticmethod
    def train_model_with_backups(model : Model, stop_after_epochs : int, max_epochs : int, directory : str):
        epoch = 0
        resume = False
        files = []
        print("Start of training")
        while epoch < max_epochs:
            epoch += BackupManager._epochs_step(epoch)
            model.train(max_num_epochs=epoch-1, stop_after_epochs=stop_after_epochs, resume_training=resume) # -1 otherwise epoch and real number of epochs trained doesn't match (because of sbi...)
            if not resume: resume = True
            real_epoch = model.neural_network.epoch
            name = f"{directory}/epoch_{real_epoch}.pkl"
            BackupManager.save_model(model, name)
            files.append(name)
            if real_epoch < epoch: break # early_stopping detected
            # I put the early stopping first because, if the nn converges exactly on a backup (exemple 110) then only one file would remain
            if len(files) > 2: # we only keep the last 2 backups (because it takes a lot of space)
                Path(files[0]).unlink()
                files.remove(files[0])
        Plotter.plot_loss(model.neural_network, f"{directory}/loss.png")

