import numpy as np
import torch
from torch import Tensor
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.objects.normalizer import Normalizer
import pickle
from tqdm.notebook import tqdm
import glob
from pathlib import Path
from sbi_particle_physics.managers.plotter import Plotter
from sbi_particle_physics.config import DATA_FILE_PATTERN, MODEL_FILE_PATTERN


class Backup:
    """
    Responsible for everything related to saving and loading data or models
    """

    @staticmethod
    def _data_file_path(directory: Path, index: int) -> Path:
        filename = DATA_FILE_PATTERN.format(index=index)
        return directory / filename

    @staticmethod
    def save_data(file : Path, device, prior_low_raw : Tensor, prior_high_raw : Tensor, raw_data : Tensor, raw_parameters : Tensor, stride : int, pre_N : int, preruns : int):
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
    def generate_many_data(model : Model, directory : Path, start_index : int, amount : int, n_samples : int, n_points : int, prior_low_raw : Tensor, prior_high_raw : Tensor):
        print("Starting to generate data")
        directory.mkdir(parents=True, exist_ok=True) # create the directory if it doesn't exists
        for i in range(start_index, start_index + amount):
            location = Backup._data_file_path(directory, i)
            raw_data, raw_parameters = model.simulate_raw_data(n_samples, n_points)
            Backup.save_data(location, model.device, prior_low_raw, prior_high_raw, raw_data, raw_parameters, model.simulator.stride, model.simulator.pre_N, model.simulator.preruns)


    @staticmethod
    def _extract_id(filepath: Path) -> int:
        name = filepath.stem # ex: "epoch_12"
        _, id_str = name.split("_")
        return int(id_str)
    
    @staticmethod
    def detect_files(directory : Path) -> list[Path]:
        pattern = DATA_FILE_PATTERN.format(index="*")
        data_files = sorted(directory.glob(pattern), key=Backup._extract_id)
        return data_files
    
    
    @staticmethod
    def load_one_file(file : Path):
        checkpoint = torch.load(file, weights_only=False)
        file_raw_data = checkpoint['raw_data']
        file_raw_parameters = checkpoint['raw_parameters']
        metadata = checkpoint['metadata']
        return file_raw_data, file_raw_parameters, metadata

    @staticmethod
    def load_data(files : list[Path]):
        all_raw_data = []
        all_raw_parameters = []
        metadata = None
        for file in tqdm(files, desc="Loading files", leave=False):
            file_raw_data, file_raw_parameters, file_metadata = Backup.load_one_file(file)
            if metadata is None: metadata = file_metadata # we keep the metadata of the first file
            all_raw_data.append(file_raw_data)
            all_raw_parameters.append(file_raw_parameters)

        raw_data = torch.cat(all_raw_data, dim=0)
        raw_parameters = torch.cat(all_raw_parameters, dim=0)
        #print(f"Merged data shape: {raw_data.shape}")
        #print(f"Merged parameters shape: {raw_parameters.shape}")
        return raw_data, raw_parameters, metadata
    
    
    @staticmethod
    def calculate_stats(files : list[Path], batchsize : int) -> tuple[float, float]:
        mean = 0
        std = 1
        cursor = 0
        while cursor < len(files):
            selected_files = files[cursor: cursor+batchsize]
            cursor += batchsize
            raw_data, _, _ = Backup.load_data(selected_files)
            data_mean, data_std = Normalizer.calculate_stats(raw_data)
            mean += data_mean * len(selected_files)
            std += data_std * len(selected_files)
        return mean / len(files), std / len(files)
    
    @staticmethod
    def load_and_append_data(model : Model, files : list[Path], batchsize : int):
        cursor = 0
        while cursor < len(files):
            raw_data, raw_parameters, met = Backup.load_data(files[cursor: cursor+batchsize])
            #raw_data = raw_data[:,:150]
            #raw_parameters = raw_parameters[:,:150]
            cursor += batchsize
            data = model.normalizer.normalize_data(raw_data)
            parameters = model.normalizer.normalize_parameters(raw_parameters)
            model.append_data(data, parameters)

    @staticmethod
    def load_data_and_build_model(directory : Path, batchsize : int, stride : int, pre_N : int, preruns : int, seed : int = None, max_files : int = None) -> tuple[Model, int]:
        # warning: here batchsize corresponds to the number of data files used at a time, not of samples
        # one file contains around 500 samples
        files = Backup.detect_files(directory) 
        if max_files is not None: files = files[:max_files]
        if len(files) == 0: raise BaseException("No files found") 
       
        print(f"{len(files)} files")
        mean, std = Backup.calculate_stats(files, batchsize)
        data0, _, metadata = Backup.load_one_file(files[0])
        n_points = data0.shape[1]
        device = torch.device(metadata['device'])
        model = Model(device, seed)

        prior_low_raw = model.to_tensor(metadata['prior_low_raw'])
        prior_high_raw = model.to_tensor(metadata['prior_high_raw'])
        model.set_prior(prior_low_raw, prior_high_raw)
        model.set_simulator(stride, pre_N, preruns)
        model.set_normalizer(mean, std)
        model.build_default() 
        Backup.load_and_append_data(model, files, batchsize)
        return model, n_points # todo traiter mieux n_points
    

    @staticmethod
    def save_model(model : Model, file : Path):
        save_dict = {
            'device': model.device,
            'rng': model.rng,
            'prior': model.prior, 
            'stride': model.simulator.stride,
            'pre_N': model.simulator.pre_N,
            'preruns': model.simulator.preruns,
            'data_mean': model.normalizer.data_mean,
            'data_std': model.normalizer.data_std,
            'posterior': model.posterior,
            'neural_network': model.neural_network,
            'training_loss': model.training_loss,
            'validation_loss': model.validation_loss
        }
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(save_dict, f)
        print(f"Model saved to {file}")

    @staticmethod
    def load_model(file : Path) -> Model:
        # its better to not pickle compex objects such as class, but instead their variables
        with file.open("rb") as f:
            save_dict = pickle.load(f)
        device = torch.device(save_dict['device']) # todo marche ?
        model = Model(device)
        model.rng = save_dict['rng']
        model.prior = save_dict['prior']
        stride, pre_N, preruns = save_dict['stride'], save_dict['pre_N'], save_dict['preruns']
        model.set_simulator(stride, pre_N, preruns)
        model.normalizer = Normalizer(save_dict['data_mean'], save_dict['data_std'])
        model.posterior = save_dict['posterior']
        model.neural_network = save_dict['neural_network']
        model.training_loss = save_dict['training_loss']
        model.validation_loss = save_dict['validation_loss']
        print(f"Model loaded from {file}")
        return model
    
    @staticmethod
    def _extract_epoch(filepath: Path) -> int:
        name = filepath.stem  # ex: "epoch_12"
        _, epoch_str = name.split("_")
        return int(epoch_str)

    @staticmethod
    def _get_corresponding_file(directory: Path, epoch: int | None) -> Path:
        pattern = MODEL_FILE_PATTERN.format(epoch="*")
        files = list(directory.glob(pattern))
        if epoch is None:
            return max(files, key=Backup._extract_epoch) # last epoch
        for file in files:
            if Backup._extract_epoch(file) == epoch:
                return file
        raise FileNotFoundError(f"No file corresponding to epoch {epoch} in directory {directory}")
        
    @staticmethod
    def load_model_basic(directory : Path, epoch : int | None = None) -> Model: # useful method to load more easily a model
        file  = Backup._get_corresponding_file(directory, epoch)
        return Backup.load_model(file)

    @staticmethod
    def _epochs_step(epochs : int):
        if epochs < 10: return 1 
        elif epochs < 30: return 5 
        else: return 10    

    @staticmethod
    def _epoch_file_path(directory: Path, epoch: int) -> Path:
        filename = MODEL_FILE_PATTERN.format(epoch=epoch)
        return directory / filename

    @staticmethod
    def train_model_with_backups(model : Model, stop_after_epochs : int, max_epochs : int, directory : Path, resume : bool = False, delete_old_backups : bool = False):
        # resume = True if the neural network has already been partially trained before
        # delete_old_backups = True: the old back up files from previous partial trainings are deleted (replaced by new backups)
        directory.mkdir(parents=True, exist_ok=True) # creates the directory if it doesn't exists
        epoch = model.neural_network.epoch if resume else 0 # model.neural_network.epoch doesn't work it the neural network hasn't been trained yet
        files = []
        if delete_old_backups:
            pattern = MODEL_FILE_PATTERN.format(epoch="*")
            files = sorted(directory.glob(pattern), key=Backup._extract_epoch,)
        print("Start of training")
        while epoch < max_epochs:
            epoch += Backup._epochs_step(epoch)
            model.train(max_num_epochs=epoch-1, stop_after_epochs=stop_after_epochs, resume_training=resume) # -1 otherwise epoch and real number of epochs trained doesn't match (because of sbi...)
            resume = True
            real_epoch = model.neural_network.epoch
            name = Backup._epoch_file_path(directory, real_epoch)
            Backup.save_model(model, name)
            Plotter.plot_loss(model, directory / "loss")
            files.append(name)
            if len(files) > 2: # we only keep the last 2 backups (because it takes a lot of space)
                Path(files[0]).unlink()
                files.remove(files[0])
            if real_epoch < epoch: break # early_stopping detected
            # it's normal that if the nn converges on a backup (exemple 110) then only the last file remains
        
        


