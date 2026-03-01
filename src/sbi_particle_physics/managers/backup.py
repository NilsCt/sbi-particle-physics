import numpy as np
from sklearn import neural_network
import torch
from torch import Tensor
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.objects.normalizer import Normalizer
from tqdm.notebook import tqdm
from pathlib import Path
from sbi_particle_physics.managers.plotter import Plotter
from sbi_particle_physics.config import DATA_FILE_PATTERN, MODEL_FILE_PATTERN, KEEP_LAST_N_BACKUPS, DEFAULT_ENCODER_ACTIVATION_FUNCTION, DEFAULT_NSF_ACTIVATION_FUNCTION, DEFAULT_WEIGHT_DECAY
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
    def save_data(file : Path, raw_data : Tensor, raw_parameters : Tensor, metadata : dict):
        torch.save({
            'raw_data': raw_data,
            'raw_parameters': raw_parameters,
            'metadata': metadata
        }, file)

    @staticmethod
    def generate_many_data(model : Model, directory : Path, start_index : int, amount : int, n_samples : int, n_points : int, prior_low_raw : Tensor, prior_high_raw : Tensor):
        print("Starting to generate data")
        directory.mkdir(parents=True, exist_ok=True) # create the directory if it doesn't exists
        for i in range(start_index, start_index + amount):
            location = Backup._data_file_path(directory, i)
            raw_data, raw_parameters = model.simulate_raw_data(n_samples, n_points)
            metadata = model.simulator.get_metadata(prior_low_raw, prior_high_raw)
            Backup.save_data(location, raw_data, raw_parameters, metadata)


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
        print("Loading a file")
        checkpoint = torch.load(file, weights_only=False, map_location=device)
        file_raw_data = checkpoint['raw_data']
        #file_raw_parameters = checkpoint['raw_parameters']
        file_raw_parameters = checkpoint['raw_parameters']
        if len(file_raw_parameters.shape) >= 3:  # j'ai des fichiers de formats bizarre [N_samples, 1, 1] (ou un des deux 1 est d_parameter)
            file_raw_parameters = file_raw_parameters.squeeze(-1)
        # todo j'ai sauvegardé tout mes fichiers dans un format bizarre ? savoir pourqoui est le regler. est ce que le squeeze fait casser si les fichiers étaient dans le bon format ?
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
                raw_parameters = raw_parameters
            cursor += batchsize
            data = model.normalizer.normalize_data(raw_data)
            parameters = model.normalizer.normalize_parameters(raw_parameters)
            model.append_data(data, parameters, f) 

    @staticmethod
    def load_and_append_data_proposals(model : Model, files : list[dict], batchsize : int, max_points : int = None):
        cursor = 0
        while cursor < len(files):
            f_with_info = [files[cursor]]
            proposal_round = files[cursor].get("proposal_round", 0)
            cursor += 1
            for e in files[cursor: cursor+batchsize-1]:
                if e.get("proposal_round", 0) == proposal_round:
                    f_with_info.append(e)
                    cursor += 1
                else:
                    break
            f = [f["path"] for f in f_with_info]
            raw_data, raw_parameters, met = Backup.load_data(f, model.device)
            if max_points is not None:
                raw_data = raw_data[:,:max_points]
                raw_parameters = raw_parameters
            cursor += batchsize
            data = model.normalizer.normalize_data(raw_data)
            parameters = model.normalizer.normalize_parameters(raw_parameters)
            model.append_data(data, parameters, f, proposal_round=proposal_round) 

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
        if max_points is not None:
            n_points = min(n_points, max_points)
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
        imperfections = None
        if hasattr(model.simulator, "imperfections") and model.simulator.imperfections is not None:
            imp = model.simulator.imperfections
            imperfections = {
                "use_acceptance": imp.use_acceptance, # Flags
                "use_resolution": imp.use_resolution,
                "use_background": imp.use_background,

                "mkpi" : imp.mkpi,
                "q2_min": imp.q2_min,
                "q2_max" : imp.q2_max,
                "mb_min" : imp.mb_min,
                "mb_max" : imp.mb_max,
                "acceptance_coeffs_path" : imp.acceptance_coeffs_path,
                "acceptance_orders" : imp.acceptance_orders,
                "acceptance_ranges_dict" : imp.acceptance_ranges_dict,
                "acceptance_coeffs" : imp.acceptance_coeffs,
                "resolution_q2_sigma_core" : imp.resolution_q2_sigma_core,
                "resolution_q2_sigma_tail" : imp.resolution_q2_sigma_tail,
                "resolution_q2_tail_fraction" : imp.resolution_q2_tail_fraction,
                "resolution_q2_sigma_slope" : imp.resolution_q2_sigma_slope,
                "resolution_cos_theta_sigma" : imp.resolution_cos_theta_sigma,
                "resolution_phi_sigma" : imp.resolution_phi_sigma,
                "background_ctl_p1" : imp.background_ctl_p1,
                "background_ctl_p2" : imp.background_ctl_p2,
                "background_ctk_p1" : imp.background_ctk_p1,
                "background_ctk_p2" : imp.background_ctk_p2,
                "background_phi_p1" : imp.background_phi_p1,
                "background_phi_p2" : imp.background_phi_p2,
                "background_tau_bkg_mb" : imp.background_tau_bkg_mb,
                "background_mb_min" : imp.background_mb_min,
                "background_mb_max" : imp.background_mb_max,
                "background_fsig_mb_window" : imp.background_fsig_mb_window
        }
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
            'q2_min': model.simulator.q2_min,
            'q2_max': model.simulator.q2_max,
            'mb_min': model.simulator.mb_min,
            'mb_max': model.simulator.mb_max,
            'lepton': model.simulator.lepton,
            'quark': model.simulator.quark,
            'model': model.simulator.model,
            'decay': model.simulator.decay,
            'imperfections': imperfections,

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
            'encoder_activation_function': model.encoder_activation_function,
            'nsf_activation_function': model.nsf_activation_function,
            'weight_decay': model.weight_decay,

            'model_type': model.model_type, # for now constant information
            'z_score_x': model.z_score_x,

            'posterior' : posterior_cpu, # sbi object for inference

            'sbi_version' : sbi.__version__, # versions
            'torch_version' : torch.__version__,

            'neural_net_state_dict': model.neural_network._neural_net.state_dict(), # weights

            'optimizer_state_dict': model.neural_network.optimizer.state_dict(), # optimizer
            # si version finale, il est plus courant de ne pas stocke l'optimizer (qui prend autant de place que le réseau)
            # qui n'est plus utilisé et qui peut poser des problèmes lors du loading

            'data_files_paths': model.export_paths(), # data

            'round': model.round, # SNPE
            'proposals_ignoring_prior': model.proposals[1:]
        }
        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, file)
        print(f"Model saved to {file}")

    @staticmethod
    def _load_util(file : Path, device : torch.device) -> tuple[Model, dict]:
        # its better to not pickle compex objects such as class, but instead their variables
        save_dict = torch.load(file, map_location=device, weights_only=False) # move every tensor in the dict to the specified device

        # old_device = torch.device(save_dict['device'])
        model = Model(device, save_dict['n_points'], save_dict['seed'])

        model.prior_type = save_dict['prior_type']
        model.set_prior(save_dict['prior_low'], save_dict['prior_high'])
        model.round = save_dict.get('round', 0)
        proposals = save_dict.get('proposals_ignoring_prior', None)
        if proposals is not None:
            for e in proposals: model.proposals.append(e)

        imperfections_cfg = save_dict.get("imperfections", None)
        q2_min = save_dict.get("q2_min", None)
        q2_max = save_dict.get("q2_max", None)
        mb_min = save_dict.get("mb_min", None)
        mb_max = save_dict.get("mb_max", None)
        lepton = save_dict.get("lepton", None)
        quark = save_dict.get("quark", None)
        model_name = save_dict.get("model", None)
        decay = save_dict.get("decay", None)
        if imperfections_cfg is None:
            model.set_simulator(save_dict['stride'], save_dict['pre_N'], save_dict['preruns'], use_imperfections=False, q2_min=q2_min, q2_max=q2_max, mb_min=mb_min, mb_max=mb_max, lepton=lepton, quark=quark, model=model_name, decay=decay)
        else:
            model.set_simulator(save_dict['stride'], save_dict['pre_N'], save_dict['preruns'], use_imperfections=True, q2_min=q2_min, q2_max=q2_max, mb_min=mb_min, mb_max=mb_max, lepton=lepton, quark=quark, model=model_name, decay=decay, **imperfections_cfg)
            # todo charger correctement les paramètres d'imperfections car pour l'instant ca ne va pas marcher

        model.set_normalizer(save_dict['data_mean'], save_dict['data_std'])

        data_files = save_dict["data_files_paths"]
        # Case 1 : list of str (for the path)
        if isinstance(data_files, list) and all(isinstance(x, str) for x in data_files):
            model.data_files_paths = [Model.create_data_dict(Path(x)) for x in data_files]
        # Cas 2 : list of dict {"path": str, "proposal_round": int}
        elif isinstance(data_files, list) and all(isinstance(x, dict) for x in data_files):
            model.data_files_paths = [Model.create_data_dict(Path(x["path"]), proposal_round=x.get("proposal_round", 0)) for x in data_files]
        else:
            raise ValueError("Unsupported format for data_files_paths")

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
            nsf_num_bins=save_dict['nsf_num_bins'],
            encoder_activation_function=save_dict.get('encoder_activation_function', DEFAULT_ENCODER_ACTIVATION_FUNCTION),
            nsf_activation_function=save_dict.get('nsf_activation_function', DEFAULT_NSF_ACTIVATION_FUNCTION),
            weight_decay=save_dict.get('weight_decay', DEFAULT_WEIGHT_DECAY)
        )

        if load_back_data:
            Backup.load_and_append_data_proposals(model, model.data_files_paths, batchsize=batchsize, max_points=model.n_points)
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
    def get_all_backup_files(directory: Path) -> list[Path]:
        pattern = MODEL_FILE_PATTERN.format(epoch="*")
        files = sorted(directory.glob(pattern), key=Backup._extract_epoch)
        return files
    
    @staticmethod
    def get_best_backup_file(model : Model, directory : Path) -> Path:
        best_epoch = model.best_val_epoch
        return Backup._get_corresponding_file(directory, best_epoch)
    
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

            # I keep the last 2 backups plus the best one
            current_val_loss = model.validation_loss[-1]
            if current_val_loss < model.best_val_loss:
                model.best_val_loss = current_val_loss
                model.best_val_epoch = real_epoch
                model.best_val_file = name
            files_sorted = sorted(files, key=Backup._extract_epoch) 
            files_to_keep = []
            files_to_keep.extend(files_sorted[-KEEP_LAST_N_BACKUPS:])
            if model.best_val_file is not None and model.best_val_file not in files_to_keep:
                files_to_keep.append(model.best_val_file)
            for f in files_sorted: 
                if f not in files_to_keep and f.exists(): f.unlink()
            files = sorted(files_to_keep, key=Backup._extract_epoch)

            if real_epoch < epoch: break # early stopping detected