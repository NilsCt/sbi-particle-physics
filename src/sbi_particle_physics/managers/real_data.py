import numpy as np
import torch
from torch import Tensor
from pathlib import Path
import uproot
import awkward as ak
from sbi_particle_physics.config import REAL_DATA_FILE_PATTERN, TREE_NAME, BRANCHES, MKPI, MKPI_DELTA, PLOTS_DIR, GREEN_COLOR, AXIS_FONTSIZE, TICK_FONTSIZE, LEGEND_FONTSIZE, PARAMETERS_LABEL, C9, DEFAULT_PRIOR_LOW, DEFAULT_PRIOR_HIGH, RED_COLOR, BLUE_COLOR, GREEN_COLOR
import re
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.predictions import Predictions
from typing import Any
import math

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
        mean, uncertainty = Predictions.calculate_estimator(sampled_parameters)
        print(f"Estimated mean: {mean}, Uncertainty: {uncertainty}")
        fig, ax = plt.subplots(figsize=(5.5,4), constrained_layout=True)
        ax.set_xlim(DEFAULT_PRIOR_LOW[0], DEFAULT_PRIOR_HIGH[0])
        ax.hist(sampled_parameters[:,0], bins=40, density=True, alpha=0.8, color=GREEN_COLOR, label="posterior")
        #ax.axvline(C9, color="red", linestyle="--", linewidth=2, label="True value") todo
        ax.set_xlabel(PARAMETERS_LABEL[0], fontsize=AXIS_FONTSIZE+8, labelpad=0) # , fontweight='bold'
        ax.set_ylabel("Density", fontsize=AXIS_FONTSIZE, labelpad=0)  #, fontweight='bold'
        ax.tick_params(labelsize=TICK_FONTSIZE-2, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.55, handlelength=1.3, handleheight=0.6, handletextpad=0.4, borderpad=0.3, labelspacing=0.2, loc="upper left")
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor('white')
        if path is None:
            plt.show()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path.with_suffix(".pdf"))
        plt.close()

    @staticmethod
    def _save_or_show(path: Path | None) -> None:
        if path is None:
            plt.show()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path.with_suffix(".pdf"))
        plt.close()

    @staticmethod
    def _plot_subset_estimates(means: Tensor, sigmas: Tensor, final_mean: float, final_unc: float, path: Path | None = None, true_value: float | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
        ax.hist(means.numpy(), bins=30, density=True, alpha=0.8, color="green", label="Estimators")
        ax.axvline(final_mean, color="red", linestyle="-", linewidth=2.5, label="Final estimate")
        ax.axvspan(final_mean - final_unc, final_mean + final_unc, alpha=0.50, color="red", label=r"Final $\pm 1\sigma$", edgecolor="none", linewidth=0)
        #if true_value is not None: todo
        #    ax.axvline(true_value, color=RED_COLOR, linestyle="--", linewidth=2, label="True value")
        ax.set_xlabel("$C_9$", fontsize=40)
        ax.set_ylabel("Density", fontsize=AXIS_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE-4, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE+1, frameon=True, framealpha=0.55, handlelength=1.3, handleheight=0.6, handletextpad=0.4, borderpad=0.3, labelspacing=0.2)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        RealData._save_or_show(path)

    @staticmethod
    def _plot_subset_errorbars(means: Tensor, sigmas: Tensor, final_mean: float, final_unc: float, path: Path | None = None, true_value: float | None = None) -> None:
        fig, ax = plt.subplots(figsize=(6.0, 4), constrained_layout=True)
        x = torch.arange(len(means)).numpy()

        ax.errorbar(
            x,
            means.numpy(),
            yerr=sigmas.numpy(),
            fmt="o",
            alpha=0.45,
            markersize=5,
            linewidth=1.5,
            capsize=4.0,
            label="Estimators",
            color="blue",
        )
        ax.axhline(final_mean, color="black", linestyle="-", linewidth=1.5, label="Final estimate")
        ax.axhspan(final_mean - final_unc, final_mean + final_unc, alpha=0.50, color="black", label=r"Final $\pm 1\sigma$", edgecolor="none", linewidth=0)

        #if true_value is not None:
        #    ax.axhline(true_value, color="red", linestyle="--", linewidth=2.5, label="True value")

        ax.set_xlabel("Index", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel("$C_9$", fontsize=40)
        ax.tick_params(labelsize=TICK_FONTSIZE - 4, width=1.2)
        ax.grid(True, alpha=0.4, linewidth=0.8)

        leg = ax.legend(
            fontsize=LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.55,
            handlelength=1.3,
            handleheight=0.6,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.2,
            loc="best",
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")

        RealData._save_or_show(path)

    @staticmethod
    def _plot_pulls(pulls: Tensor, path: Path | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
        ax.hist(pulls.numpy(), bins=30, density=True, alpha=0.8, color=GREEN_COLOR, label="pulls")
        #ax.axvline(0.0, color=RED_COLOR, linestyle="-", linewidth=2, label="0")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label=r"$\pm 1 \sigma$")
        ax.axvline(-1.0, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel(r"$(\hat{\theta}_i - \hat{\theta}_{\mathrm{final}})/\sigma_i$", fontsize=AXIS_FONTSIZE, labelpad=0)
        ax.set_ylabel("Density", fontsize=AXIS_FONTSIZE, labelpad=0)
        ax.tick_params(labelsize=TICK_FONTSIZE-3, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE+2, frameon=True, framealpha=0.55, handlelength=1.3, handleheight=0.6, handletextpad=0.4, borderpad=0.3, labelspacing=0.2, loc="upper left")
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        RealData._save_or_show(path)

    @staticmethod
    def _plot_subset_uncertainties(sigmas: Tensor, path: Path | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
        ax.hist(sigmas.numpy(), bins=30, density=True, alpha=0.8, color=GREEN_COLOR, label="subset uncertainties")
        ax.axvline(torch.median(sigmas).item(), color="black", linestyle="-", linewidth=2, label="median")
        ax.set_xlabel(r"Subset estimated uncertainty", fontsize=AXIS_FONTSIZE, labelpad=0)
        ax.set_ylabel("Density", fontsize=AXIS_FONTSIZE, labelpad=0)
        ax.tick_params(labelsize=TICK_FONTSIZE, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.55, handlelength=1.3, handleheight=0.6, handletextpad=0.4, borderpad=0.3, labelspacing=0.2, loc="upper left")
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        RealData._save_or_show(path)

    @staticmethod
    def _plot_estimate_vs_uncertainty(means: Tensor, sigmas: Tensor, final_mean: float, path: Path | None = None, true_value: float | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
        ax.scatter(sigmas.numpy(), means.numpy(), alpha=0.75, s=18, label="subsets")
        ax.axhline(final_mean, color="black", linestyle="-", linewidth=2, label="final estimate")
        if true_value is not None:
            ax.axhline(true_value, color="red", linestyle="--", linewidth=2, label="True value")
        ax.set_xlabel(r"Subset estimated uncertainty", fontsize=AXIS_FONTSIZE, labelpad=0)
        ax.set_ylabel(PARAMETERS_LABEL[0], fontsize=AXIS_FONTSIZE + 5, labelpad=0)
        ax.tick_params(labelsize=TICK_FONTSIZE, width=1.2)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.55, handlelength=1.3, handleheight=0.6, handletextpad=0.4, borderpad=0.3, labelspacing=0.2, loc="best")
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        RealData._save_or_show(path)

    @staticmethod
    def _plot_cumulative_estimator(means: Tensor, sigmas: Tensor, path: Path | None = None, true_value: float | None = None) -> None:
        weights = 1.0 / (sigmas ** 2)
        cumulative_w = torch.cumsum(weights, dim=0)
        cumulative_wm = torch.cumsum(weights * means, dim=0)
        cumulative_mean = cumulative_wm / cumulative_w
        cumulative_unc = torch.sqrt(1.0 / cumulative_w)

        x = torch.arange(1, len(means) + 1).numpy()

        fig, ax = plt.subplots(figsize=(6.0, 4), constrained_layout=True)
        ax.plot(x, cumulative_mean.numpy(), linewidth=2.5, label="Cumulative estimate", color="blue")
        ax.fill_between(
            x,
            (cumulative_mean - cumulative_unc).numpy(),
            (cumulative_mean + cumulative_unc).numpy(),
            alpha=0.20,
            label=r"Cumulative $\pm 1\sigma$",
            color="red"
        )
        if true_value is not None:
            ax.axhline(true_value, color=RED_COLOR, linestyle="--", linewidth=2, label="True value")
        ax.set_xlabel("Number of subsets included", fontsize=AXIS_FONTSIZE-3, labelpad=0)
        ax.set_ylabel(PARAMETERS_LABEL[0], fontsize=AXIS_FONTSIZE + 5, labelpad=0)
        ax.tick_params(labelsize=TICK_FONTSIZE-4, width=1.2)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.55, handlelength=1.3, handleheight=0.6, handletextpad=0.4, borderpad=0.3, labelspacing=0.2, loc="best")
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        RealData._save_or_show(path)

    @staticmethod
    def calculate_best_estimator(model: Model, path_real_data: Path, n_parameters: int = 1000, n_subsamples: int = 200, sample_with_replacement: bool = False, path: Path | None = None) -> tuple[float, float]:
        real_raw_data, _ = RealData.load_whole_directory(path_real_data, device=torch.device("cpu"))
        real_raw_data = real_raw_data.to(torch.device("cpu"))
        real_data = model.normalizer.normalize_data(real_raw_data)

        n_total = real_data.shape[0]
        n_points = model.n_points

        if n_total == 0:
            raise ValueError("real_data is empty.")
        if n_total < n_points and not sample_with_replacement:
            raise ValueError(f"Not enough data points ({n_total}) for subsets of size {n_points} without replacement.")

        subset_means = []
        subset_uncertainties = []

        for _ in range(n_subsamples):
            if sample_with_replacement:
                idx = torch.randint(0, n_total, (n_points,))
            else:
                idx = torch.randperm(n_total)[:n_points]

            subset = real_data[idx]
            sampled_parameters = model.draw_parameters_from_predicted_posterior(
                subset,
                n_parameters=n_parameters,
            )

            if sampled_parameters.ndim >= 3 and sampled_parameters.shape[0] == 1:
                sampled_parameters = sampled_parameters.squeeze(0)

            mean_i, unc_i = Predictions.calculate_estimator(sampled_parameters)
            mean_i = mean_i.detach().cpu().float()
            unc_i = unc_i.detach().cpu().float()

            if torch.isfinite(mean_i) and torch.isfinite(unc_i) and unc_i > 0:
                subset_means.append(mean_i)
                subset_uncertainties.append(unc_i)

        if len(subset_means) == 0:
            raise RuntimeError("No valid subset estimate could be computed.")

        means = torch.stack(subset_means)
        sigmas = torch.stack(subset_uncertainties)
        n_valid = len(means)
        weights = 1.0 / (sigmas ** 2)
        final_mean = torch.sum(weights * means) / torch.sum(weights)
        final_unc_stat = torch.sqrt(1.0 / torch.sum(weights))

        if n_valid > 1:
            subset_std = torch.std(means, unbiased=True)
            chi2 = torch.sum(((means - final_mean) / sigmas) ** 2)
            chi2_red = chi2 / (n_valid - 1)
            birge_ratio = torch.sqrt(torch.clamp(chi2_red, min=1.0))
            final_unc = final_unc_stat * birge_ratio
            pulls = (means - final_mean) / sigmas
            pull_mean = torch.mean(pulls)
            pull_std = torch.std(pulls, unbiased=True)
            frac_within_1sigma = torch.mean((torch.abs(pulls) <= 1.0).float())
            frac_within_2sigma = torch.mean((torch.abs(pulls) <= 2.0).float())
            q16 = torch.quantile(means, 0.16)
            q84 = torch.quantile(means, 0.84)
            robust_half_68 = 0.5 * (q84 - q16)
        else:
            subset_std = torch.tensor(0.0)
            chi2_red = torch.tensor(float("nan"))
            birge_ratio = torch.tensor(1.0)
            final_unc = final_unc_stat
            pulls = torch.zeros_like(means)
            pull_mean = torch.tensor(0.0)
            pull_std = torch.tensor(float("nan"))
            frac_within_1sigma = torch.tensor(float("nan"))
            frac_within_2sigma = torch.tensor(float("nan"))
            robust_half_68 = torch.tensor(0.0)

        print("\n===== Best estimator diagnostics =====")
        print(f"Total real data points            : {n_total}")
        print(f"Points per subset                 : {n_points}")
        print(f"Requested random subsets          : {n_subsamples}")
        print(f"Valid subsets                     : {n_valid}")
        print(f"Fraction valid subsets            : {n_valid / n_subsamples:.3f}")
        print()
        print(f"Final estimate                    : {final_mean.item():.6g} ± {final_unc.item():.6g}")
        print(f"Stat-only uncertainty             : {final_unc_stat.item():.6g}")
        print()
        print(f"Mean subset uncertainty           : {torch.mean(sigmas).item():.6g}")
        print(f"Median subset uncertainty         : {torch.median(sigmas).item():.6g}")
        print(f"Std of subset estimates           : {subset_std.item():.6g}")
        print(f"Robust half central 68% width     : {robust_half_68.item():.6g}")
        print()
        print(f"Reduced chi2                      : {chi2_red.item():.6g}")
        print(f"Birge ratio                       : {birge_ratio.item():.6g}")
        print(f"Pull mean                         : {pull_mean.item():.6g}")
        print(f"Pull std                          : {pull_std.item():.6g}")
        print(f"Frac within 1 sigma               : {frac_within_1sigma.item():.3f}")
        print(f"Frac within 2 sigma               : {frac_within_2sigma.item():.3f}")
        print("=====================================\n")

        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            RealData._plot_subset_estimates(means, sigmas, final_mean.item(), final_unc.item(), path / "best_estimator_subset_estimates", C9)
            RealData._plot_subset_errorbars(means, sigmas, final_mean.item(), final_unc.item(), path / "best_estimator_subset_errorbars", C9)
            RealData._plot_pulls(pulls, path / "best_estimator_pulls")
            RealData._plot_subset_uncertainties(sigmas, path / "best_estimator_subset_uncertainties")
            RealData._plot_estimate_vs_uncertainty(means, sigmas, final_mean.item(), path / "best_estimator_estimate_vs_uncertainty", C9)
            RealData._plot_cumulative_estimator(means, sigmas, path / "best_estimator_cumulative_estimate", C9)
        else:
            RealData._plot_subset_estimates(means, sigmas, final_mean.item(), final_unc.item(), None, C9)
            RealData._plot_subset_errorbars(means, sigmas, final_mean.item(), final_unc.item(), None, C9)
            RealData._plot_pulls(pulls, None)
            RealData._plot_subset_uncertainties(sigmas, None)
            RealData._plot_estimate_vs_uncertainty(means, sigmas, final_mean.item(), None, C9)
            RealData._plot_cumulative_estimator(means, sigmas, None, C9)

        return final_mean.item(), final_unc.item()