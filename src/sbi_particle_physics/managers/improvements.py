import matplotlib.pyplot as plt
import torch
from torch import Tensor
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.plotter import Plotter
import numpy as np
from pathlib import Path
from sbi_particle_physics.config import AXIS_FONTSIZE, TICK_FONTSIZE, LEGEND_FONTSIZE
from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.managers.predictions import Predictions

class Improvements:
    """
    Diagnostics needing multiple version of the neural network to deduce what needs to be improved

    What can limit the performance?
        - not enough data -> increasing the number of files
        - not enough points in a sample -> increasing the number of points
        - the nn is not complex enough to express the posterior -> increasing the number of layers and neurons (encoder or nsf)

    Elements that can also improve the predictions:
        - the activation function: switching from ReLU to SiLU / GeLU?
        - SNPE
        - regularisation
        - removing some observables?

    How to know what is limiting the performance?

    - Plot a graph of the average posterior width as a function of the number of points in a sample.
    If the width decreases with 1/sqrt(n_points) then it is limiting and n_points can be increased to make the predictions better
    - Plot a graph of the average posterior width as a function of the number of files.
    If the witdh decreases, then it is limiting and the number of files can be increased to make the predictions better.
    - Same method for the architecture of the neural network: increase the number of layers and neurons and check if the predictions are better.

    What would the ideal neural network be able to do?
    If we have access to N points of real life data, then the ideal scenario would be to train a neural network such that the limiting
    factor is the number of points in a sample (information of the sample itself) and with a number of points equal to N.
    So the architecture of the neural network and the amount of data needs to be improved until they are not limiting.
    Note that it might be possible that even by increasing the number of layers, of neurons and of data files, the number of points might still limit the predictions for an unknown reason.
    Then it would be better to use the maximum number of points that achieve the maximum performance.
    Then maybe the real life data can be split into multiple sets, and the values of the parameters can be infered with these sets and then compared.
    This would allow to cross-check and maybe also to decrease the uncertainty.
    """

    @staticmethod
    def plot_width_by_npoints(model_dirs: list[Path], device: torch.device, raw_observed_data: Tensor, n_posterior_samples: int = 1000):
        """
        Plots a graph of the average width of the posteriors as a function of the number 
        of points in each samples given to the neural network during trainig.
        If the curve follows 1/sqrt(n_points) then the posteriors witdh are limited by the information in a sample
        To improve the performance, the number of points per sample should be increased
        """
        n_points_list = []
        avg_widths = []
        for model_dir in model_dirs:
            model: Model = Backup.load_model_for_inference_basic(directory=model_dir, device=device)
            n_points = model.n_points
            observed_data = model.normalizer.normalize_data(raw_observed_data)
            with torch.no_grad():
                posterior_samples = model.draw_parameters_from_predicted_posterior(observed_data[:,:n_points], n_parameters=n_posterior_samples)
            avg_width = Predictions.average_uncertainty(posterior_samples)
            n_points_list.append(n_points)
            avg_widths.append(avg_width)
        n_points_arr = np.array(n_points_list, dtype=float)
        avg_widths_arr = np.array(avg_widths, dtype=float)
        order = np.argsort(n_points_arr)
        n_points_arr = n_points_arr[order]
        avg_widths_arr = avg_widths_arr[order]
        N_ref = n_points_arr[0]
        width_ref = avg_widths_arr[0]
        trend_1_over_sqrtN = width_ref * np.sqrt(N_ref / n_points_arr)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(n_points_arr, avg_widths_arr, marker="o", linestyle="-", label="Neural network")
        ax.plot(n_points_arr, trend_1_over_sqrtN, linestyle="--", color="black", label=r"$\propto 1/\sqrt{n_{\mathrm{points}}}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n_{\mathrm{points}}$", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel("Average posterior width (68%)", fontsize=AXIS_FONTSIZE - 4)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_width_by_npoints_pro(model_dirs: list[Path], device: torch.device, raw_observed_data : Tensor, n_posterior_samples: int = 1000):
        """
        Plots a graph of the following:
        For each neural network (that can be trained with different n_points per sample)
        I calculate the average posterior width for different number of points of the observed_sample (can't exceed n_points during training)
        This plot turned out to not be really useful.
        The average width do decrease, but not with 1/sqrt(n_points_observed) as the nn can extrapolate
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        for model_dir in model_dirs:
            avg_widths = []
            model: Model = Backup.load_model_for_inference_basic(directory=model_dir, device=device)
            n_points = model.n_points
            observed_data = model.normalizer.normalize_data(raw_observed_data)[:,:n_points]
            na = np.linspace(130, n_points, 8, dtype=int)
            for n in na:
                with torch.no_grad():
                    x_padded = torch.full(observed_data.shape, float('nan'), device=model.device)
                    x_padded[:,:n] = observed_data[:,:n]
                    posterior_samples = model.draw_parameters_from_predicted_posterior(x_padded, n_parameters=n_posterior_samples)
                avg_width = Predictions.average_uncertainty(posterior_samples)
                avg_widths.append(avg_width)
            n_points_arr = np.array(na, dtype=float)
            avg_widths_arr = np.array(avg_widths, dtype=float)
            ax.plot(n_points_arr, avg_widths_arr, marker="o", linestyle="-", label=f"nn trained with {n_points}")
            N_ref = n_points_arr[0]
            width_ref = avg_widths_arr[0]
            trend_1_over_sqrtN = width_ref * np.sqrt(N_ref / n_points_arr)
            ax.plot(n_points_arr, trend_1_over_sqrtN, linestyle="--", color="black", alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n_{\mathrm{points}}$", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel("Average posterior width (68%)", fontsize=AXIS_FONTSIZE - 4)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=LEGEND_FONTSIZE-4)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_width_by_npoints_quantify(model_dirs: list[Path], device: torch.device, raw_observed_data: Tensor, n_posterior_samples: int = 1000):
        """
        Quantifies whether the posterior uncertainties are statistically limited
        by fitting the relation: sigma^2(N) = a / N + b
        where:
            - sigma(N) is the average posterior width (68%)
            - a / N is the statistical contribution
            - b is the intrinsic (irreducible) uncertainty floor
        If b ≈ 0, uncertainties are dominated by statistics and increasing the
        number of points per sample will continue to reduce posterior widths.
        If b > 0, the uncertainties saturate and increasing N yields diminishing returns.
        """
        n_points_list = []
        avg_widths = []
        for model_dir in model_dirs:
            model: Model = Backup.load_model_for_inference_basic(directory=model_dir, device=device)
            n_points = model.n_points
            observed_data = model.normalizer.normalize_data(raw_observed_data)
            with torch.no_grad():
                posterior_samples = model.draw_parameters_from_predicted_posterior(observed_data[:, :n_points],n_parameters=n_posterior_samples,)
            avg_width = Predictions.average_uncertainty(posterior_samples)
            n_points_list.append(n_points)
            avg_widths.append(avg_width)
        n_points_arr = np.array(n_points_list, dtype=float)
        avg_widths_arr = np.array(avg_widths, dtype=float)
        order = np.argsort(n_points_arr)
        n_points_arr = n_points_arr[order]
        avg_widths_arr = avg_widths_arr[order]
        # --- Prepare fit variables ---
        inv_N = 1.0 / n_points_arr
        sigma2 = avg_widths_arr ** 2
        # --- Linear fit: sigma^2 = a * (1/N) + b ---
        coeffs = np.polyfit(inv_N, sigma2, deg=1)
        a, b = coeffs
        inv_N_fit = np.linspace(inv_N.min(), inv_N.max(), 200)
        sigma2_fit = a * inv_N_fit + b
        # --- Characteristic N* where a/N = b ---
        N_star = a / b if b > 0 else np.inf
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(inv_N,sigma2,marker="o",linestyle="",label=r"Measured $\sigma^2$",)
        ax.plot(inv_N_fit,sigma2_fit,linestyle="--",color="black",label=rf"Fit: $\sigma^2 = ({a:.3e})/N + ({b:.3e})$",)
        ax.set_xlabel(r"$1 / n_{\mathrm{points}}$", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel(r"$(\mathrm{Average\ posterior\ width})^2$", fontsize=AXIS_FONTSIZE - 4)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        plt.show()
        print("===== Statistical scaling analysis =====")
        print(f"a (statistical term)       = {a:.4e}")
        print(f"b (intrinsic floor)        = {b:.4e}")
        if np.isfinite(N_star):
            print(f"Characteristic N* (a/N=b) = {N_star:.1f}")
        else:
            print("Characteristic N*         = infinity (purely statistical regime)")

