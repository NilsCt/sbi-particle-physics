import numpy as np
import torch
from torch import Tensor
from pathlib import Path
import uproot
import awkward as ak
from sbi_particle_physics.config import REAL_DATA, REAL_DATA_FILE_PATTERN, TREE_NAME, BRANCHES
import re
from tqdm.notebook import tqdm

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
    def load_one_file(file : Path, device : torch.device) -> Tensor:
        """
        Load real LHCb data from a root file
        """
        file = uproot.open(file)
        tree = file[TREE_NAME]
        raw_data = tree.arrays(BRANCHES, library="ak")
        raw_X = np.stack([ak.to_numpy(raw_data[b]) for b in BRANCHES], axis=1)
        return torch.tensor(raw_X, dtype=torch.float32, device=device)
    
    @staticmethod
    def load_files(files : list[Path], device : torch.device) -> Tensor:
        all_raw_data = []
        for file in tqdm(files, desc="Loading LHCb files", leave=False):
            file_raw_data = RealData.load_one_file(file, device)
            all_raw_data.append(file_raw_data)
        return torch.cat(all_raw_data, dim=0)
    
    @staticmethod
    def load_whole_directory(directory : Path, device: torch.device) -> Tensor:
        files = RealData.detect_files(directory)
        return RealData.load_files(files=files, device=device)
    
    @staticmethod
    def load_n_points(directory: Path, n_points: int, device: torch.device, ignore_first: int = 0) -> Tensor:
        """
        Load a tensor of n_points from LHCb data files in directory.
        The first ignore_first points are ignored globally while reading files sequentially.
        """
        files = RealData.detect_files(directory)
        chunks: list[Tensor] = []
        collected = 0
        skipped = 0 
        for file in tqdm(files, desc="Loading LHCb data (partial)", leave=False):
            if collected >= n_points: break
            f = uproot.open(file)
            tree = f[TREE_NAME]
            n_entries = tree.num_entries
            if skipped < ignore_first: # Decide how many points can we skip in this file
                skip_here = min(ignore_first - skipped, n_entries)
                entry_start = skip_here
                skipped += skip_here
            else:
                entry_start = 0
            if entry_start >= n_entries:
                continue  # nothing to read in this file
            need = n_points - collected
            entry_stop = min(entry_start + need, n_entries)
            
            raw = tree.arrays(BRANCHES, library="ak", entry_start=entry_start, entry_stop=entry_stop)
            X = np.stack([ak.to_numpy(raw[b]) for b in BRANCHES], axis=1)
            X = torch.tensor(X, dtype=torch.float32, device=device)
            chunks.append(X)
            collected += X.shape[0]

        if len(chunks) == 0:
            return torch.empty((0, len(BRANCHES)), dtype=torch.float32, device=device)
        out = torch.cat(chunks, dim=0)
        if out.shape[0] < n_points:
            print(f"[RealData.load_n_points] Warning: requested {n_points} events but only found {out.shape[0]} after skipping {ignore_first}.")
        return out[:n_points]

