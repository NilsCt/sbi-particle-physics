import numpy as np
import torch
from torch import Tensor
from pathlib import Path
import uproot
import awkward as ak
from sbi_particle_physics.config import REAL_DATA_FILE_PATTERN, TREE_NAME, BRANCHES, MKPI, MKPI_DELTA, PLOTS_DIR, GREEN_COLOR, AXIS_FONTSIZE, TICK_FONTSIZE, LEGEND_FONTSIZE, PARAMETERS_LABEL, C9, DEFAULT_PRIOR_LOW, DEFAULT_PRIOR_HIGH
import re
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sbi_particle_physics.objects.model import Model

class RealData:
    """
    Responsible to load and format real LHCb data from root files
    """

    @staticmethod
    def _data_file_path(directory: Path, bin: int, job1: int, job2: int) -> Path:
        filename = REAL_DATA_FILE_PATTERN.format(bin=bin, job1=job1, job2=job2)
        return directory / filename

    @staticmethod
    def _extract_bin_job(filename: str | Path) -> tuple[int, int, int]:
        filename = Path(filename).name
        pattern = r"dataset_bin_(\d+)_job_(\d+)_(\d+)\.root"
        match = re.fullmatch(pattern, filename)
        if match is None:
            raise ValueError(f"Filename does not match expected pattern: {filename}")
        bin_, job1, job2 = map(int, match.groups())
        return bin_, job1, job2
    
    @staticmethod
    def _bin_job_score(filepath: Path) -> int:
        bin, job1, job2 = RealData._extract_bin_job(filepath)
        return bin*1e9 + job1*1e6 + job2*1e3
    
    @staticmethod
    def detect_files(directory : Path) -> list[Path]:
        pattern = REAL_DATA_FILE_PATTERN.format(bin="*", job1="*", job2="*")
        data_files = sorted(directory.glob(pattern), key=RealData._bin_job_score)
        return data_files
    
    @staticmethod
    def _filter_data(real_data : Tensor, mkpi : Tensor) -> tuple[Tensor, Tensor]:
        mask = (mkpi >= MKPI - MKPI_DELTA) & (mkpi <= MKPI + MKPI_DELTA)
        return real_data[mask], mkpi[mask]

    @staticmethod
    def load_one_file(file : Path, device : torch.device) -> tuple[Tensor, Tensor]:
        """
        Load real LHCb data from a root file
        """
        file = uproot.open(file)
        tree = file[TREE_NAME]
        #print(f"tree branches: {tree.keys()}")
        raw_data = tree.arrays(BRANCHES, library="ak")
        raw_X = np.stack([ak.to_numpy(raw_data[b]) for b in BRANCHES], axis=1)
        raw_X[:, -1] = raw_X[:, -1] / 1000.0 # convert mB from MeV to GeV
        raw_X = torch.tensor(raw_X, dtype=torch.float32, device=device)
        mkpi = ak.to_numpy(tree.arrays("mkpi", library="ak")["mkpi"])
        mkpi = torch.tensor(mkpi, dtype=torch.float32, device=device)
        #raw_X, mkpi = RealData._filter_data(raw_X, mkpi)
        #print(f"mkpi shape {mkpi.shape}, mkpi {mkpi[:5]}")
        #plt.hist(mkpi, bins=50)
        #plt.axvline(0.892+0.04, color="red", linestyle="--", label="±40 MeV")
        #plt.axvline(0.892-0.04, color="red", linestyle="--")
        #plt.axvline(0.892+0.05, color="black", linestyle="--", label="±50 MeV")
        #plt.axvline(0.892-0.05, color="black", linestyle="--")
        #plt.axvline(0.892+0.060, color="green", linestyle="--", label="±60 MeV")
        #plt.axvline(0.892-0.060, color="green", linestyle="--")
        #plt.legend()
        #plt.xlabel("mkpi (GeV)")
        return raw_X, mkpi
    
    @staticmethod
    def load_files(files : list[Path], device : torch.device) -> tuple[Tensor, Tensor]:
        all_raw_data = []
        all_mkpi = []
        for file in tqdm(files, desc="Loading LHCb files", leave=False):
            file_raw_data, file_mkpi = RealData.load_one_file(file, device)
            all_raw_data.append(file_raw_data)
            all_mkpi.append(file_mkpi)
        return torch.cat(all_raw_data, dim=0), torch.cat(all_mkpi, dim=0)
    
    @staticmethod
    def load_whole_directory(directory : Path, device: torch.device) -> tuple[Tensor, Tensor]:
        files = RealData.detect_files(directory)
        return RealData.load_files(files=files, device=device)
    
    @staticmethod
    def load_n_points(directory: Path, n_points: int, device: torch.device, ignore_first: int = 0) -> tuple[Tensor, Tensor]:
        """
        Load a tensor of n_points from LHCb data files in directory.
        The first ignore_first points are ignored globally while reading files sequentially.
        """
        files = RealData.detect_files(directory)
        chunks: list[Tensor] = []
        mkpi_chunks: list[Tensor] = []
        collected = 0
        skipped = 0
        for file in tqdm(files, desc="Loading LHCb data (partial)", leave=False):
            if collected >= n_points:
                break
            X, mkpi = RealData.load_one_file(file, device)
            n_entries = X.shape[0]
            if skipped < ignore_first: # skip events if needed
                skip_here = min(ignore_first - skipped, n_entries)
                entry_start = skip_here
                skipped += skip_here
            else:
                entry_start = 0
            if entry_start >= n_entries:
                continue
            need = n_points - collected
            entry_stop = min(entry_start + need, n_entries)
            X_slice = X[entry_start:entry_stop]
            mkpi_slice = mkpi[entry_start:entry_stop]
            chunks.append(X_slice)
            mkpi_chunks.append(mkpi_slice)
            collected += X_slice.shape[0]

        if len(chunks) == 0:
            return (torch.empty((0, len(BRANCHES)), dtype=torch.float32, device=device),
                torch.empty((0,), dtype=torch.float32, device=device))
        out = torch.cat(chunks, dim=0)
        mkpi_out = torch.cat(mkpi_chunks, dim=0)
        if out.shape[0] < n_points:
            print(f"[RealData.load_n_points] Warning: requested {n_points} events but only found {out.shape[0]} after skipping {ignore_first}.")
        return out[:n_points], mkpi_out[:n_points]
    
    @staticmethod
    def plot_real_data_posterior(model : Model, real_data : Tensor, n_samples : int = 1000, path : Path = None):
        sampled_parameters = model.draw_parameters_from_predicted_posterior(real_data, n_parameters=n_samples).squeeze(0)
        fig, ax = plt.subplots(figsize=(5.5,4), constrained_layout=True)
        ax.set_xlim(DEFAULT_PRIOR_LOW[0], DEFAULT_PRIOR_HIGH[0])
        ax.hist(sampled_parameters[:,0], bins=40, density=True, alpha=0.8, color=GREEN_COLOR, label="posterior")
        ax.axvline(C9, color="red", linestyle="--", linewidth=2, label="True value")
        ax.set_xlabel(PARAMETERS_LABEL[0], fontsize=AXIS_FONTSIZE+8, labelpad=0) # , fontweight='bold'
        ax.set_ylabel("Density", fontsize=AXIS_FONTSIZE, labelpad=0)  #, fontweight='bold'
        ax.tick_params(labelsize=TICK_FONTSIZE, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.55, handlelength=1.3, handleheight=0.6, handletextpad=0.4, borderpad=0.3, labelspacing=0.2, loc="upper left")
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor('white')
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
        plt.close()