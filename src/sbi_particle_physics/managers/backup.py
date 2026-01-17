import numpy as np
from sklearn import neural_network
import torch
from torch import Tensor
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.objects.normalizer import Normalizer
import pickle
from tqdm.notebook import tqdm
from pathlib import Path
from sbi_particle_physics.managers.plotter import Plotter
from sbi_particle_physics.config import DATA_FILE_PATTERN, MODEL_FILE_PATTERN, ENCODED_POINT_DIM, DEFAULT_POINTS_PER_SAMPLE, PARAMETERS_DIM
import sbi


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
    def load_one_file(file : Path, device : torch.device) -> tuple[Tensor, Tensor, dict]:
        checkpoint = torch.load(file, weights_only=False, map_location=device)
        file_raw_data = checkpoint['raw_data']
        file_raw_parameters = checkpoint['raw_parameters']
        metadata = checkpoint['metadata']
        return file_raw_data, file_raw_parameters, metadata

    @staticmethod
    def load_data(files : list[Path], device : torch.device) -> tuple[Tensor, Tensor, dict]:
        all_raw_data = []
        all_raw_parameters = []
        metadata = None
        for file in tqdm(files, desc="Loading files", leave=False):
            file_raw_data, file_raw_parameters, file_metadata = Backup.load_one_file(file, device)
            if metadata is None: metadata = file_metadata # we keep the metadata of the first file
            all_raw_data.append(file_raw_data)
            all_raw_parameters.append(file_raw_parameters)

        raw_data = torch.cat(all_raw_data, dim=0)
        raw_parameters = torch.cat(all_raw_parameters, dim=0)
        #print(f"Merged data shape: {raw_data.shape}")
        #print(f"Merged parameters shape: {raw_parameters.shape}")
        return raw_data, raw_parameters, metadata
    
    
    @staticmethod
    def calculate_stats(files : list[Path], batchsize : int, device : torch.device) -> tuple[float, float]:
        mean = 0
        std = 1
        cursor = 0
        while cursor < len(files):
            selected_files = files[cursor: cursor+batchsize]
            cursor += batchsize
            raw_data, _, _ = Backup.load_data(selected_files, device)
            data_mean, data_std = Normalizer.calculate_stats(raw_data)
            mean += data_mean * len(selected_files)
            std += data_std * len(selected_files)
        return mean / len(files), std / len(files)
    
    @staticmethod
    def load_and_append_data(model : Model, files : list[Path], batchsize : int, max_points : int = None):
        cursor = 0
        while cursor < len(files):
            f = files[cursor: cursor+batchsize]
            raw_data, raw_parameters, met = Backup.load_data(f, model.device)
            if max_points is not None:
                raw_data = raw_data[:,:max_points]
                raw_parameters = raw_parameters[:,:max_points]
            cursor += batchsize
            data = model.normalizer.normalize_data(raw_data)
            parameters = model.normalizer.normalize_parameters(raw_parameters)
            model.append_data(data, parameters, f)

    @staticmethod
    def load_data_and_build_model(directory : Path, device : torch.device, batchsize : int, stride : int, pre_N : int, preruns : int, seed : int = None, max_files : int = None, max_points : int = None) -> Model:
        # warning: here batchsize corresponds to the number of data files used at a time, not of samples
        # one file contains around 500 samples
        files = Backup.detect_files(directory) 
        if max_files is not None: files = files[:max_files]
        if len(files) == 0: raise BaseException("No files found") 
       
        print(f"{len(files)} files")
        mean, std = Backup.calculate_stats(files, batchsize=batchsize, device=device)
        data0, _, metadata = Backup.load_one_file(files[0], device)
        n_points = data0.shape[1]
        model = Model(device, n_points, seed)

        prior_low_raw = model.to_tensor(metadata['prior_low_raw'])
        prior_high_raw = model.to_tensor(metadata['prior_high_raw'])
        model.set_prior(prior_low_raw, prior_high_raw)
        model.set_simulator(stride, pre_N, preruns)
        model.set_normalizer(mean, std)
        model.build_default() 
        Backup.load_and_append_data(model, files, batchsize, max_points)
        return model
    

    @staticmethod
    def save_model(model : Model, file : Path):
        posterior_cpu = model.posterior
        if posterior_cpu is not None:
            posterior_cpu.to("cpu") # avec sbi ca modifie l'objet en place (comme moi)
        save_dict = {
            'device': model.device, # utils
            'n_points': model.n_points,
            'seed': model.seed,

            'prior_type' : model.prior_type, # prior
            'prior_low': model.prior.low.cpu(),
            'prior_high': model.prior.high.cpu(),

            'stride': model.simulator.stride, # simulator
            'pre_N': model.simulator.pre_N,
            'preruns': model.simulator.preruns,

            'data_mean': model.normalizer.data_mean.cpu(), # normalizer
            'data_std': model.normalizer.data_std.cpu(),
            'parameters_mean': model.normalizer.parameters_mean,
            'parameters_std': model.normalizer.parameters_std,

            'training_loss': model.training_loss, # training
            'validation_loss': model.validation_loss,
            'epoch': model.epoch,

            'trial_num_layers': model.trial_num_layers, # architecture
            'trial_num_hiddens': model.trial_num_hiddens,
            'trial_embedding_dim': model.trial_embedding_dim,
            'aggregated_num_layers': model.aggregated_num_layers,
            'aggregated_num_hiddens': model.aggregated_num_hiddens,
            'aggregated_output_dim': model.aggregated_output_dim,
            'nsf_hidden_features': model.nsf_hidden_features,
            'nsf_num_transforms': model.nsf_num_transforms,
            'nsf_num_bins': model.nsf_num_bins,

            'model_type': model.model_type, # for now constant information
            'z_score_x': model.z_score_x,

            'posterior' : posterior_cpu, # sbi object for inference

            'sbi_version' : sbi.__version__, # versions
            'torch_version' : torch.__version__,

            'neural_net_state_dict': model.neural_network._neural_net.state_dict(), # weights

            'optimizer_state_dict': model.neural_network.optimizer.state_dict(), # optimizer
            # si version finale, il est plus courant de ne pas stocke l'optimizer (qui prend autant de place que le réseau)
            # qui n'est plus utilisé et qui peut poser des problèmes lors du loading

            'data_files_paths': [str(x) for x in model.data_files_paths] # data
        }
        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, file)
        print(f"Model saved to {file}")

    @staticmethod
    def _load_util(file : Path, device : torch.device) -> tuple[Model, dict]:
        # its better to not pickle compex objects such as class, but instead their variables
        save_dict = torch.load(file, map_location=device) # move every tensor in the dict to the specified device

        # old_device = torch.device(save_dict['device'])
        model = Model(device, save_dict['n_points'], save_dict['seed'])

        model.prior_type = save_dict['prior_type']
        model.set_prior(save_dict['prior_low'], save_dict['prior_high'])

        model.set_simulator(save_dict['stride'], save_dict['pre_N'], save_dict['preruns'])

        model.set_normalizer(save_dict['data_mean'], save_dict['data_std'])

        model.data_files_paths = [Path(x) for x in save_dict['data_files_paths']]

        model.training_loss = save_dict['training_loss']
        model.validation_loss = save_dict['validation_loss']
        model.epoch = save_dict['epoch']

        model.posterior = save_dict['posterior']

        print(f"Model loaded from {file}")
        return model, save_dict

    @staticmethod
    def load_model_for_inference(file : Path, device : torch.device) -> Model:
        # when loaded for inference, neural_nework can't be used, can't be trained, new posteriors can't be created
        # only other variables and model.posterior are loaded
        model, _ = Backup._load_util(file, device)
        return model

    @staticmethod
    def load_model_for_training(file : Path, device : torch.device, load_back_data : bool = True, batchsize : int = 1, first_file : Path = None) -> Model:
        # if load_back_data is False, first_file must be specified (to load data from and do a dummy epoch to initialize the nn)
        model, save_dict = Backup._load_util(file, device)

        model.build(
            trial_num_layers=save_dict['trial_num_layers'],
            trial_num_hiddens=save_dict['trial_num_hiddens'],
            trial_embedding_dim=save_dict['trial_embedding_dim'],
            aggregated_num_layers=save_dict['aggregated_num_layers'],
            aggregated_num_hiddens=save_dict['aggregated_num_hiddens'],
            aggregated_output_dim=save_dict['aggregated_output_dim'],
            nsf_hidden_features=save_dict['nsf_hidden_features'],
            nsf_num_transforms=save_dict['nsf_num_transforms'],
            nsf_num_bins=save_dict['nsf_num_bins']
        )

        if load_back_data:
            Backup.load_and_append_data(model, model.data_files_paths, batchsize=batchsize, max_points=model.n_points)
        elif first_file is not None:
            Backup.load_and_append_data(model, [first_file], batchsize=batchsize, max_points=model.n_points)
        model.neural_network.train(max_num_epochs=1) # otherwise _neural_net is not initialized and the weights can't be loaded

        model.neural_network.epoch = save_dict['epoch']
        model.neural_network._neural_net.load_state_dict(save_dict['neural_net_state_dict'])

        model.neural_network.optimizer.load_state_dict(save_dict['optimizer_state_dict'])

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
    def load_model_for_inference_basic(directory : Path, device : torch.device, epoch : int | None = None) -> Model: # useful method to load more easily a model
        file  = Backup._get_corresponding_file(directory, epoch)
        return Backup.load_model_for_inference(file, device)

    @staticmethod
    def load_model_for_training_basic(directory : Path, device : torch.device, epoch : int | None = None, load_back_data : bool = True, batchsize : int = 1) -> Model: # useful method to load more easily a model
        file  = Backup._get_corresponding_file(directory, epoch)
        return Backup.load_model_for_training(file, device, load_back_data=load_back_data, batchsize=batchsize)


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
            real_epoch = model.epoch
            name = Backup._epoch_file_path(directory, real_epoch)
            Backup.save_model(model, name)
            Plotter.plot_loss(model, directory / "loss")
            files.append(name)
            if len(files) > 2: # we only keep the last 2 backups (because it takes a lot of space)
                Path(files[0]).unlink()
                files.remove(files[0])
            if real_epoch < epoch: break # early_stopping detected
            # it's normal that if the nn converges on a backup (exemple 110) then only the last file remains
        
        


