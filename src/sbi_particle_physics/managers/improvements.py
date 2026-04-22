import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from sbi_particle_physics.objects import model
import torch
from torch import Tensor
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.plotter import Plotter
import numpy as np
from pathlib import Path
from sbi_particle_physics.config import AXIS_FONTSIZE, PLOT_COLORS, TICK_FONTSIZE, LEGEND_FONTSIZE, PLOTS_DIR, RED_COLOR, GREEN_COLOR, BLUE_COLOR
from sbi_particle_physics.managers.backup import Backup
from sbi_particle_physics.managers.predictions import Predictions
from sbi.diagnostics import run_sbc
from sbi.diagnostics import run_tarp, check_tarp

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
    def _plot_width_by(x_values: np.ndarray | list, width: np.ndarray | list, x_label: str, curve_label : str, also_x_log : bool = True, no_line : bool = False) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray]:
        x_values = np.array(x_values, dtype=float)
        width = np.array(width, dtype=float)
        order = np.argsort(x_values)
        x_values = x_values[order]
        width = width[order]
        fig, ax = plt.subplots(figsize=(5.5,4), constrained_layout=True)
        linestyle = "" if no_line else "-"
        ax.plot(x_values, width, marker="o", linestyle=linestyle, label=curve_label, linewidth=2.2, color=BLUE_COLOR)
        if also_x_log: ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label, fontsize=AXIS_FONTSIZE+6, labelpad=0) # , fontweight='bold'
        ax.set_ylabel("$\\langle \\sigma \\rangle$", fontsize=AXIS_FONTSIZE+4, labelpad=0) # , fontweight='bold'
        #ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.tick_params(labelsize=TICK_FONTSIZE-2, width=1.2)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE+2, frameon=True, framealpha=0.55, borderpad=0.4, labelspacing=0.3)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor('white')
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5)))
        ax.tick_params(axis="y", which="major", width=1.2, labelsize=TICK_FONTSIZE-2, length=6)
        ax.tick_params(axis="y", which="minor", width=1.2, labelsize=TICK_FONTSIZE-6, length=6)
        return fig, ax, x_values, width
    
    @staticmethod
    def _plot_width_by_quantify(x_values: np.ndarray | list, width: np.ndarray | list, x_label: str, curve_label: str, ignore_n_first_points : int = 0) -> tuple[plt.Figure, plt.Axes]:
        x_values = np.array(x_values, dtype=float)
        width = np.array(width, dtype=float)
        order = np.argsort(x_values)
        x_values = x_values[order][ignore_n_first_points:]
        width = width[order][ignore_n_first_points:]

        inv_N = 1.0 / x_values
        sigma2 = width ** 2
        coeffs = np.polyfit(inv_N, sigma2, deg=1)
        a, b = coeffs
        inv_N_fit = np.linspace(inv_N.min(), inv_N.max(), 200)
        sigma2_fit = a * inv_N_fit + b
        N_star = a / b if b > 0 else np.inf

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(inv_N,sigma2,marker="o",linestyle="",label=curve_label)
        ax.plot(inv_N_fit,sigma2_fit,linestyle="--",color="black",label=rf"Fit: $\sigma^2 = a/N + b$",)
        ax.set_xlabel(x_label, fontsize=AXIS_FONTSIZE)
        ax.set_ylabel(r"$(\langle \sigma \rangle)^2$", fontsize=AXIS_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        fig.tight_layout()
        print(f"a = {a:.4e}")
        print(f"b = {b:.4e}")
        print(f"N* = a/b = {N_star:.1f}")
        return fig, ax

    @staticmethod
    def plot_width_by_npoints(model_dirs: list[Path], device: torch.device, raw_observed_data: Tensor, n_posterior_samples: int = 1000):
        """
        Plots a graph of the average width of the posteriors as a function of the number 
        of points in each samples given to the neural network during trainig.
        If the curve follows 1/sqrt(n_points) then the posteriors witdh are limited by the information in a sample
        To improve the performance, the number of points per sample should be increased

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
            model: Model = Backup.load_model_for_inference_basic(directory=model_dir, device=device, use_best=True)
            n_points = model.n_points
            observed_data = model.normalizer.normalize_data(raw_observed_data)
            posterior_samples = model.draw_parameters_from_predicted_posterior(observed_data[:,:n_points], n_parameters=n_posterior_samples)
            avg_width = Predictions.average_uncertainty(posterior_samples)
            n_points_list.append(n_points)
            avg_widths.append(avg_width)
        fig, ax, n_points_arr, avg_widths_arr = Improvements._plot_width_by(n_points_list, avg_widths, r"$N_e$", "Neural networks", no_line=True) # $n_{\mathrm{points}}$
        N_ref = n_points_arr[0]
        width_ref = avg_widths_arr[0]
        trend_1_over_sqrtN = width_ref * np.sqrt(N_ref / n_points_arr)
        ax.plot(n_points_arr, trend_1_over_sqrtN, linestyle="--", color="red", label=r"$\propto 1/\sqrt{N_e}$", linewidth=2.5, alpha=0.9)
        plt.legend(fontsize=LEGEND_FONTSIZE+1, frameon=True, framealpha=0.55, borderpad=0.4, labelspacing=0.3)
        plt.savefig(PLOTS_DIR / "viva" / "uncertainty_npoints.pdf")
        fig.show()
        fig, ax = Improvements._plot_width_by_quantify(n_points_list, avg_widths, r"$1/n_{\mathrm{points}}$", "Neural network")
        fig.show()

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
            model: Model = Backup.load_model_for_inference_basic(directory=model_dir, device=device, use_best=True)
            n_points = model.n_points
            observed_data = model.normalizer.normalize_data(raw_observed_data)[:,:n_points]
            na = np.linspace(n_points//1.5, n_points, 8, dtype=int) # don't go below n_points/2 or the nn won't be able to infere a posterior
            for n in na:
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
    def plot_width_by_nfiles(model_dirs: list[Path], device: torch.device, raw_observed_data: Tensor, n_posterior_samples: int = 1000):
        """
        Plots a graph of the average width of the posteriors as a function of the number 
        of files used during training.
        If the curve decreases with increasing number of files then the posteriors witdh are limited by the amount of data
        To improve the performance, the number of files should be increased
        """
        n_files_list = []
        avg_widths = []
        for model_dir in model_dirs:
            model: Model = Backup.load_model_for_inference_basic(directory=model_dir, device=device, use_best=True)
            n_files = len(model.data_files_paths)
            observed_data = model.normalizer.normalize_data(raw_observed_data)
            posterior_samples = model.draw_parameters_from_predicted_posterior(observed_data, n_parameters=n_posterior_samples)
            avg_width = Predictions.average_uncertainty(posterior_samples)
            n_files_list.append(n_files)
            avg_widths.append(avg_width)
        fig, ax, n_files_arr, avg_widths_arr = Improvements._plot_width_by(n_files_list, avg_widths, r"$N_s$", "Neural networks", no_line=True) # n_{\mathrm{files}}
        plt.savefig(PLOTS_DIR / "viva" / "uncertainty_nsamples.pdf")
        fig.show()
        fig, ax = Improvements._plot_width_by_quantify(n_files_list, avg_widths, r"$1/N_s$", "Neural network")
        fig.show()

    @staticmethod
    def plot_width_by_epochs(files : list[Path], device: torch.device, raw_observed_data: Tensor, n_posterior_samples: int = 1000, ignore_n_first_points: int = 0):
        """
        Plots a graph of the average width of the posteriors as a function of the number 
        of training epochs.
        If the curve decreases with increasing number of epochs then the posteriors witdh are limited by the training time
        To improve the performance, the number of epochs should be increased
        """
        n_epochs_list = []
        avg_widths = []

        for file in files:
            model: Model = Backup.load_model_for_inference(file=file, device=device)
            n_epochs = model.epoch
            observed_data = model.normalizer.normalize_data(raw_observed_data[:,:model.n_points])
            posterior_samples = model.draw_parameters_from_predicted_posterior(observed_data, n_parameters=n_posterior_samples)
            avg_width = Predictions.average_uncertainty(posterior_samples)
            n_epochs_list.append(n_epochs)
            avg_widths.append(avg_width)
        fig, ax, n_epochs_arr, avg_widths_arr = Improvements._plot_width_by(n_epochs_list, avg_widths, r"$N_{\mathrm{epochs}}$", "Neural networks", also_x_log=False, no_line = True)
        plt.savefig(PLOTS_DIR / "viva" / "uncertainty_epochs.pdf")
        fig.show()
        fig, ax = Improvements._plot_width_by_quantify(n_epochs_list, avg_widths, r"$1/n_{\mathrm{epochs}}$", "Neural network", ignore_n_first_points=ignore_n_first_points)
        fig.show()


    @staticmethod
    def _bar_plot(model_names : list[str], values : list[float], ylabel : str):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(model_names, values, color=BLUE_COLOR, alpha=1)
        ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE+1)
        ax.tick_params(axis="x", rotation=30, labelsize=TICK_FONTSIZE-3)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE-7)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        plt.savefig(PLOTS_DIR / "viva" / f"compare_{ylabel.replace(' ', '_').replace('/', '_').replace('\\', '_')}.pdf")
        plt.show()

    def _normalize(x): # Normalisation min-max for charplots
        if np.allclose(x.max(), x.min()):
            return np.ones_like(x)
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def _radar_plot(model_names: list[str], avg_widths: list[float], avg_info_gains: list[float], avg_log_contractions: list[float], avg_entropies: list[float], robust_cv: list[float]):
        """
        Radar chart for multi-metric model comparison.
        All axes are normalized so that higher is better and to make all metrics comparable.
        """
        metrics = {
            "Sharpness\n(- width)": np.array(avg_widths),
            "Information\ngain": np.array(avg_info_gains),
            "Log\ncontraction": np.array(avg_log_contractions),
            "Simplicity\n(- entropy)": np.array(avg_entropies),
            "Robustness\n(- CV)" : np.array(robust_cv)
            
        }
        radar_data = {
            "Sharpness\n(- width)": 1.0 - Improvements._normalize(metrics["Sharpness\n(- width)"]),
            "Information\ngain": Improvements._normalize(metrics["Information\ngain"]),
            "Log\ncontraction": Improvements._normalize(metrics["Log\ncontraction"]),
            "Simplicity\n(- entropy)": 1.0 - Improvements._normalize(metrics["Simplicity\n(- entropy)"]),
            "Robustness\n(- CV)": 1.0 - Improvements._normalize(metrics["Robustness\n(- CV)"])
        }
        labels = list(radar_data.keys())
        n_axes = len(labels)
        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles += angles[:1]  # close loop

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        for i, name in enumerate(model_names):
            values = [radar_data[label][i] for label in labels]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.15)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=TICK_FONTSIZE)
        ax.set_ylim(0, 1)
        ax.set_rlabel_position(0)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        plt.show()

        
    @staticmethod
    def compare_models(model_dirs: list[Path], model_names: list[str], device: torch.device, raw_observed_data: Tensor, n_posterior_samples: int = 1000,):
        """
        Compare several neural networks trained with different non-quantifiable choices
        (activation, encoder, regularisation, architecture, etc.) at fixed conditions.

        Compute and display the metrics:
            - Average posterior width (68%)
            - Log-contraction with respect to the prior
            - Information gain (entropy reduction)
            - Posterior entropy
            - Distribution of uncertainties per parameter

        Good model: narrow posteriors, high contraction, high information gain, low entropy
        Robust model: similar uncertainties across parameters, narrow box plot, no catastrophic outliers
        """
        avg_widths = []
        avg_log_contractions = []
        avg_info_gains = []
        avg_entropies = []
        all_widths = []
        robust_cv = []
        robust_worst_ratio = []
        robust_quantile_ratio = []
        for i, model_dir in enumerate(model_dirs):
            model = Backup.load_model_for_inference_basic(directory=model_dir, device=device, use_best=True)
            name = model_names[i]
            observed_data = model.normalizer.normalize_data(raw_observed_data)
            posterior_samples = model.draw_parameters_from_predicted_posterior(observed_data, n_parameters=n_posterior_samples)
            prior_samples = model.prior.sample((n_posterior_samples,)).to(device) # for comparisons

            avg_width = Predictions.average_uncertainty(posterior_samples)
            log_contr = Predictions.log_contraction(prior_samples, posterior_samples)
            info_gain = Predictions.information_gain(prior_samples, posterior_samples)
            entropy_post = Predictions._entropy_from_samples(posterior_samples, xmin=posterior_samples.min(), xmax=posterior_samples.max())
            avg_widths.append(avg_width)
            avg_log_contractions.append(log_contr.mean().item())
            avg_info_gains.append(info_gain.mean().item())
            avg_entropies.append(entropy_post.mean().item())
            widths = Predictions._uncertainty(posterior_samples) # per-parameter widths
            all_widths.append(widths.cpu().numpy())
            mean_w = widths.mean().item()
            std_w = widths.std(unbiased=False).item()
            robust_cv.append(std_w / mean_w)
            robust_worst_ratio.append((widths.max() / widths.min()).item())
            q16, q84 = torch.quantile(widths, torch.tensor([0.16, 0.84], device=widths.device))
            robust_quantile_ratio.append((q84 / q16).item())    
            print(f"Model: {name}")
            print(f"  Avg width (68%)        = {avg_width:.4e}")
            print(f"  Avg log-contraction    = {log_contr.mean().item():.4f}")
            print(f"  Avg information gain   = {info_gain.mean().item():.4f}")
            print(f"  Avg posterior entropy  = {entropy_post.mean().item():.4f}")
            print("")

        Improvements._bar_plot(model_names, avg_widths, r"$\langle \sigma \rangle$") # lower is better
        Improvements._bar_plot(model_names, avg_info_gains, r"Information gain") # higher is better
        Improvements._bar_plot(model_names, avg_log_contractions, r"Log contraction") # relative to prior, higher is better
        Improvements._bar_plot(model_names, avg_entropies, r"Posterior entropy") # lower is better
        Improvements._bar_plot(model_names, robust_cv, r"CV of posterior widths") # robust should be close to 0
        Improvements._bar_plot(model_names, robust_quantile_ratio, r"$q_{84} / q_{16}$ of widths") # robust should be close to 1

        fig, ax = plt.subplots(figsize=(8, 4)) # Robustness: per-parameter widths
        ax.boxplot(all_widths, labels=model_names, showfliers=False)
        ax.set_ylabel(r"Posterior width (68%)", fontsize=AXIS_FONTSIZE)
        ax.set_title("Robustness across parameters", fontsize=AXIS_FONTSIZE)
        ax.tick_params(axis="x", rotation=30, labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        plt.show()

        Improvements._radar_plot(model_names=model_names, avg_widths=avg_widths, avg_info_gains=avg_info_gains, avg_log_contractions=avg_log_contractions, avg_entropies=avg_entropies, robust_cv=robust_cv)

    @staticmethod
    def plot_drift_by_noise(model_dirs: list[Path], model_names: list[str], device: torch.device, raw_observed_data: Tensor, noise_levels: list[float], n_posterior_samples: int = 1000):
        """
        Compare the robustness of multiple neural networks to additive noise perturbations
        applied to the normalized observed data.
        For each model, Gaussian noise of amplitude δ is added to the normalized observation,
        and the posterior mean θ̂(δ) inferred from the perturbed input is compared to the
        reference posterior mean θ̂(0) inferred from the unperturbed observation.
        The robustness metric is the drift:
            Drift(δ) = || θ̂(δ) - θ̂(0) ||
        which measures the sensitivity of the inferred parameters to small changes in the input.

        Interpretation:
            - Robust model:
                • Drift(δ) ≈ 0 for small δ
                • Drift increases smoothly and monotonically as δ increases
                • No abrupt jumps or non-monotonic behavior
            - Fragile model:
                • Large drift already for small δ
                • Non-smooth or non-monotonic drift
                • Abrupt changes indicating instability of the inference
        """
        if raw_observed_data.ndim == 2:
            x_o = raw_observed_data.unsqueeze(0)
        else:
            x_o = raw_observed_data
        fig, (ax_drift, ax_width) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        for i, model_dir in enumerate(model_dirs):
            model = Backup.load_model_for_inference_basic(directory=model_dir, device=device, use_best=True)
            model_name = model_names[i]
            x_base = model.normalizer.normalize_data(x_o[:,:model.n_points]) # this time, noise is added after normalization
            #posterior_ref = model.draw_parameters_from_predicted_posterior(x_base, n_parameters=n_posterior_samples)
            #mean_ref, _ = Predictions.calculate_estimator(posterior_ref)
            mean_ref = None
            drifts = []
            widths = []
            for delta in noise_levels:
                #if delta == 0.0:
                #    drifts.append(0.0)
                #    widths.append(Predictions.average_uncertainty(posterior_ref))
                #    continue
                noise = delta * torch.randn_like(x_base)
                x_noisy = x_base + noise
                posterior = model.draw_parameters_from_predicted_posterior(x_noisy, n_parameters=n_posterior_samples)
                avg_width = Predictions.average_uncertainty(posterior)
                widths.append(avg_width)
                mean_delta, _ = Predictions.calculate_estimator(posterior)
                print("mean_delta shape", mean_delta.shape)
                if delta == 0: mean_ref = mean_delta
                drift = torch.norm(mean_delta - mean_ref, dim=-1).mean().item()
                drifts.append(drift)
            ax_drift.plot(noise_levels, drifts, marker="o", linewidth=2, label=model_name, color=PLOT_COLORS[i % len(PLOT_COLORS)])
            ax_width.plot(noise_levels, widths, marker="o", linewidth=2, label=model_name, color=PLOT_COLORS[i % len(PLOT_COLORS)])

        ax_drift.set_xlabel(r"Noise amplitude $\delta$", fontsize=AXIS_FONTSIZE+1)
        ax_drift.set_ylabel(r"$\|\hat{\theta}(\delta)-\hat{\theta}(0)\|$", fontsize=AXIS_FONTSIZE+1)
        ax_drift.tick_params(labelsize=TICK_FONTSIZE)
        ax_drift.grid(alpha=0.3)
        ax_drift.legend(fontsize=LEGEND_FONTSIZE+2, loc="lower right")

        ax_width.set_xlabel(r"Noise amplitude $\delta$", fontsize=AXIS_FONTSIZE+1)
        ax_width.set_ylabel(r"$\langle \sigma \rangle$", fontsize=AXIS_FONTSIZE+1)
        ax_width.tick_params(labelsize=TICK_FONTSIZE)
        ax_width.grid(alpha=0.3)

        fig.tight_layout()
        plt.savefig(PLOTS_DIR / "viva" / "drift_by_noise.pdf")
        plt.show()

    @staticmethod
    def plot_robust_npoints(model_dirs: list[Path], model_names: list[str], device: torch.device, raw_observed_data: Tensor, n_points_list: list[int] | None = None, n_posterior_samples: int = 1000, default_number_ns : int = 20):
        """
        Compare the robustness of multiple neural networks to a reduction in the number
        of observed points by measuring the drift of the posterior mean.
        For each model, the posterior mean inferred using only n observed points
        (with remaining points padded with NaNs) is compared to the reference posterior
        mean inferred using the maximum number of points.
        Drift(n) = || θ̂(n) - θ̂(N_max) ||

        Interpretation:
            - Robust model:
                • Drift(n) ≈ 0 for n close to N_max
                • Drift increases smoothly as n decreases
                • No abrupt jumps or threshold effects
            - Fragile model:
                • Large drift for moderately reduced n
                • Sharp transitions or non-monotonic behavior
        """
        if raw_observed_data.ndim == 2:
            x_o = raw_observed_data.unsqueeze(0)
        else:
            x_o = raw_observed_data
        B, N_max, D = x_o.shape
        fig, (ax_drift, ax_width) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        for i, model_dir in enumerate(model_dirs):
            model = Backup.load_model_for_inference_basic(directory=model_dir, device=device, use_best=True)
            model_name = model_names[i]
            x_ref = model.normalizer.normalize_data(x_o)
            posterior_ref = model.draw_parameters_from_predicted_posterior(x_ref, n_parameters=n_posterior_samples)
            mean_ref, _ = Predictions.calculate_estimator(posterior_ref)
            drifts = []
            widths = []
            effective_ns = n_points_list
            if effective_ns is None:
                effective_ns = np.linspace(N_max//1.5, N_max, default_number_ns, dtype=int)
            for n in effective_ns:
                n = max(min(int(n), N_max), 1)
                x_pad = torch.full_like(x_o, float("nan"), device=device) # Pad missing points with NaNs
                x_pad[:, :n, :] = x_o[:, :n, :]
                x_n = model.normalizer.normalize_data(x_pad)
                posterior = model.draw_parameters_from_predicted_posterior(x_n, n_parameters=n_posterior_samples)
                avg_width = Predictions.average_uncertainty(posterior)
                widths.append(avg_width)        
                mean_n, _ = Predictions.calculate_estimator(posterior)
                drift = torch.norm(mean_n - mean_ref, dim=-1).mean().item()
                drifts.append(drift)
            ax_drift.plot(effective_ns, drifts, marker="o", linewidth=2, label=model_name, color=PLOT_COLORS[i % len(PLOT_COLORS)])
            ax_width.plot(effective_ns, widths, marker="o", linewidth=2, label=model_name, color=PLOT_COLORS[i % len(PLOT_COLORS)])

        ax_drift.set_xlabel(r"Observed $N_e$", fontsize=AXIS_FONTSIZE+1) #n_{\mathrm{points}}
        ax_drift.set_ylabel(r"$\|\hat{\theta}(n)-\hat{\theta}(N_{\max})\|$", fontsize=AXIS_FONTSIZE+1)
        ax_drift.tick_params(labelsize=TICK_FONTSIZE)
        ax_drift.grid(alpha=0.3)
        ax_drift.legend(fontsize=LEGEND_FONTSIZE+2)
        ax_width.set_xlabel(r"Observed $N_e$", fontsize=AXIS_FONTSIZE+1) # n_{\mathrm{points}}
        ax_width.set_ylabel(r"$\langle \sigma \rangle$", fontsize=AXIS_FONTSIZE+1)
        ax_width.tick_params(labelsize=TICK_FONTSIZE)
        ax_width.grid(alpha=0.3)
        #ax_width.set_yscale("log")
        fig.tight_layout()
        plt.savefig(PLOTS_DIR / "viva" / "drift_by_npoints.pdf")
        plt.show()

    @staticmethod
    def plot_drift_by_noise_poster(model_dirs: list[Path], device: torch.device, raw_observed_data: Tensor, noise_levels: list[float], labels: list[str], n_posterior_samples: int = 1000):
        if raw_observed_data.ndim == 2:
            x_o = raw_observed_data.unsqueeze(0)
        else:
            x_o = raw_observed_data
        fig, ax_drift = plt.subplots(figsize=(5.5,4), constrained_layout=True)
        i = 0
        for model_dir in model_dirs:
            model = Backup.load_model_for_inference_basic(directory=model_dir, device=device, use_best=True)
            model_name = model_dir.name
            x_base = model.normalizer.normalize_data(x_o) # this time, noise is added after normalization
            posterior_ref = model.draw_parameters_from_predicted_posterior(x_base, n_parameters=n_posterior_samples)
            mean_ref, _ = Predictions.calculate_estimator(posterior_ref)
            drifts = []
            widths = []
            for delta in noise_levels:
                if delta == 0.0:
                    drifts.append(0.0)
                    widths.append(Predictions.average_uncertainty(posterior_ref))
                    continue
                noise = delta * torch.randn_like(x_base)
                x_noisy = x_base + noise
                posterior = model.draw_parameters_from_predicted_posterior(x_noisy, n_parameters=n_posterior_samples)
                avg_width = Predictions.average_uncertainty(posterior)
                widths.append(avg_width)
                mean_delta, _ = Predictions.calculate_estimator(posterior)
                drift = torch.norm(mean_delta - mean_ref, dim=-1).mean().item()
                drifts.append(drift)
            ax_drift.plot(noise_levels, drifts, marker="o", linewidth=2.2, label=labels[i])
            i += 1

        ax_drift.set_xlabel(r"Noise amplitude", fontsize=AXIS_FONTSIZE-2, labelpad=0) # , fontweight='bold'
        ax_drift.set_ylabel(r"Estimator shift", fontsize=AXIS_FONTSIZE-2, labelpad=0) # , fontweight='bold'
        ax_drift.tick_params(labelsize=TICK_FONTSIZE+4, width=1.2)
        ax_drift.locator_params(nbins=4)
        ax_drift.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax_drift.legend(fontsize=LEGEND_FONTSIZE+2, frameon=True, framealpha=0.55, borderpad=0.4, labelspacing=0.3)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor('white')
        plt.savefig(PLOTS_DIR / "poster" / "image_robustness.svg")
        plt.show()

    @staticmethod
    def compare_activate_functions():
        x = torch.linspace(-4, 4, 1000)
        relu = torch.relu(x)
        silu = x * torch.sigmoid(x)
        gelu = torch.nn.functional.gelu(x)
        plt.figure(figsize=(5, 4))
        plt.plot(x.cpu(), relu.cpu(), label="ReLU", linewidth=2.5, color="red")
        plt.plot(x.cpu(), silu.cpu(), label="SiLU", linewidth=2.5, color="blue")
        plt.plot(x.cpu(), gelu.cpu(), label="GeLU", linewidth=2.5, color="black")
        plt.xlabel(r"$t$", fontsize=AXIS_FONTSIZE+2)
        plt.ylabel(r"$f(t)$", fontsize=AXIS_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE-5)
        plt.yticks(fontsize=TICK_FONTSIZE-5)
        plt.legend(fontsize=LEGEND_FONTSIZE+2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "viva" / "activation_functions.pdf")
        plt.show()

    @staticmethod
    def plot_drift_by_noise_fixed(
        model_dirs: list[Path],
        model_names: list[str],
        device: torch.device,
        raw_observed_data: Tensor,
        noise_levels: list[float],
        n_posterior_samples: int = 1000,
    ):
        """
        Robustness diagnostic under additive Gaussian noise on the normalized observation.

        For each model:
        - build a reference estimator on the unperturbed input x_base
        - for each noise amplitude delta, add Gaussian noise after normalization
        - infer posterior samples and compute:
                (1) mean absolute shift:
                    < |theta_hat(delta) - theta_hat(0)| >
                (2) mean posterior uncertainty:
                    < sigma(delta) >
                (3) mean standardized shift:
                    < |theta_hat(delta) - theta_hat(0)| / sigma(0) >

        Notes
        -----
        - raw_observed_data contains a batch of N observed samples.
        - The returned posterior summaries are therefore vectors of shape [N].
        - The drift is averaged over the N observed samples.
        - The uncertainty is also averaged over the N observed samples.
        - The standardized shift compares the typical noise-induced displacement
        to the clean-reference posterior uncertainty.
        """

        n_reference_repeats = 1
        n_noisy_repeats = 1
        n_estimator_repeats = 1
        eps = 1e-12

        def stable_posterior_summary(model, x_input: Tensor, n_samples: int, n_repeats: int):
            means = []
            widths = []

            for _ in range(n_repeats):
                posterior = model.draw_parameters_from_predicted_posterior(
                    x_input, n_parameters=n_samples
                )
                mean, _ = Predictions.calculate_estimator(posterior)  # [N]
                width = Predictions._uncertainty(posterior)           # [N]
                means.append(mean)
                widths.append(width)

            means = torch.stack(means, dim=0)    # [R, N]
            widths = torch.stack(widths, dim=0)  # [R, N]

            mean_stable = means.mean(dim=0)      # [N]
            width_stable = widths.mean(dim=0)    # [N]
            return mean_stable, width_stable

        if raw_observed_data.ndim == 2:
            x_o = raw_observed_data.unsqueeze(0)
        else:
            x_o = raw_observed_data

        fig_shift, ax_shift = plt.subplots(1, 1, figsize=(5.8, 4.3))
        fig_width, ax_width = plt.subplots(1, 1, figsize=(5.8, 4.3))
        fig_std, ax_stdshift = plt.subplots(1, 1, figsize=(5.8, 4.3))

        plot_colors = ["blue", "red", "purple", "green", "black"]
        plot_markers = ["o", "s", "^", "D", "v", "P"]

        for i, model_dir in enumerate(model_dirs):
            model = Backup.load_model_for_inference_basic(
                directory=model_dir,
                device=device,
                use_best=True,
            )
            model_name = model_names[i]
            color = plot_colors[i % len(plot_colors)]
            marker = plot_markers[i % len(plot_markers)]

            x_base = model.normalizer.normalize_data(x_o[:, :model.n_points])

            mean_ref, width_ref = stable_posterior_summary(
                model=model,
                x_input=x_base,
                n_samples=n_posterior_samples,
                n_repeats=n_reference_repeats,
            )

            baseline_abs_shifts = []
            baseline_std_shifts = []
            baseline_widths = []

            for _ in range(n_noisy_repeats):
                mean_0, width_0 = stable_posterior_summary(
                    model=model,
                    x_input=x_base,
                    n_samples=n_posterior_samples,
                    n_repeats=n_estimator_repeats,
                )

                abs_shift_0 = (mean_0 - mean_ref).abs()
                std_shift_0 = abs_shift_0 / (width_ref + eps)

                baseline_abs_shifts.append(abs_shift_0.mean().item())
                baseline_std_shifts.append(std_shift_0.mean().item())
                baseline_widths.append(width_0.mean().item())

            mean_abs_shifts = []
            mean_widths = []
            mean_std_shifts = []

            for delta in noise_levels:
                abs_shift_repeats = []
                width_repeats = []
                std_shift_repeats = []

                if delta == 0.0:
                    abs_shift_repeats = baseline_abs_shifts
                    width_repeats = baseline_widths
                    std_shift_repeats = baseline_std_shifts
                else:
                    for _ in range(n_noisy_repeats):
                        noise = delta * torch.randn_like(x_base)
                        x_noisy = x_base + noise

                        mean_delta, width_delta = stable_posterior_summary(
                            model=model,
                            x_input=x_noisy,
                            n_samples=n_posterior_samples,
                            n_repeats=n_estimator_repeats,
                        )

                        abs_shift = (mean_delta - mean_ref).abs()
                        std_shift = abs_shift / (width_ref + eps)

                        abs_shift_repeats.append(abs_shift.mean().item())
                        width_repeats.append(width_delta.mean().item())
                        std_shift_repeats.append(std_shift.mean().item())

                mean_abs_shifts.append(float(np.mean(abs_shift_repeats)))
                mean_widths.append(float(np.mean(width_repeats)))
                mean_std_shifts.append(float(np.mean(std_shift_repeats)))

            ax_shift.plot(
                noise_levels,
                mean_abs_shifts,
                marker=marker,
                linestyle="-",
                linewidth=2.3,
                markersize=6,
                label=model_name,
                color=color,
            )

            ax_width.plot(
                noise_levels,
                mean_widths,
                marker=marker,
                linestyle="-",
                linewidth=2.3,
                markersize=6,
                label=model_name,
                color=color,
            )

            ax_stdshift.plot(
                noise_levels,
                mean_std_shifts,
                marker=marker,
                linestyle="-",
                linewidth=2.3,
                markersize=6,
                label=model_name,
                color=color,
            )

        ax_shift.set_xlabel(r"Noise amplitude $\delta$", fontsize=AXIS_FONTSIZE)
        ax_shift.set_ylabel(
            r"$\left\langle |\hat{\theta}(\delta)-\hat{\theta}(0)| \right\rangle$",
            fontsize=AXIS_FONTSIZE,
        )
        ax_shift.tick_params(labelsize=TICK_FONTSIZE - 6, width=1.2)
        ax_shift.grid(True, alpha=0.35, linewidth=0.8)
        leg = ax_shift.legend(
            fontsize=LEGEND_FONTSIZE + 1,
            frameon=True,
            framealpha=0.55,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
            loc="best",
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        fig_shift.tight_layout()
        plt.figure(fig_shift.number)
        plt.savefig(PLOTS_DIR / "viva" / "drift_by_noise.pdf")

        ax_width.set_xlabel(r"Noise amplitude $\delta$", fontsize=AXIS_FONTSIZE)
        ax_width.set_ylabel(
            r"$\langle \sigma(\delta) \rangle$",
            fontsize=AXIS_FONTSIZE,
        )
        ax_width.tick_params(labelsize=TICK_FONTSIZE - 6, width=1.2)
        ax_width.grid(True, alpha=0.35, linewidth=0.8)
        leg = ax_width.legend(
            fontsize=LEGEND_FONTSIZE + 3,
            frameon=True,
            framealpha=0.55,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
            loc="best",
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        fig_width.tight_layout()
        plt.figure(fig_width.number)
        plt.savefig(PLOTS_DIR / "viva" / "width_by_noise.pdf")

        ax_stdshift.set_xlabel(r"Noise amplitude $\delta$", fontsize=AXIS_FONTSIZE)
        ax_stdshift.set_ylabel(
            r"$\left\langle \frac{|\hat{\theta}(\delta)-\hat{\theta}(0)|}{\sigma(0)} \right\rangle$",
            fontsize=AXIS_FONTSIZE,
        )
        ax_stdshift.tick_params(labelsize=TICK_FONTSIZE - 2, width=1.2)
        ax_stdshift.grid(True, alpha=0.35, linewidth=0.8)
        ax_stdshift.axhline(1.0, linestyle=":", linewidth=1.5, color="black", alpha=0.8)
        ax_stdshift.axhline(0.5, linestyle=":", linewidth=1.0, color="black", alpha=0.5)
        leg = ax_stdshift.legend(
            fontsize=LEGEND_FONTSIZE + 3,
            frameon=True,
            framealpha=0.55,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
            loc="best",
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        fig_std.tight_layout()
        plt.figure(fig_std.number)
        plt.savefig(PLOTS_DIR / "viva" / "standardized_drift_by_noise.pdf")

        plt.show()


    @staticmethod
    def plot_robust_npoints_fixed(
        model_dirs: list[Path],
        model_names: list[str],
        device: torch.device,
        raw_observed_data: Tensor,
        n_points_list: list[int] | None = None,
        n_posterior_samples: int = 1000,
        default_number_ns: int = 10,
    ):
        """
        Robustness diagnostic under a reduction of the number of observed points.

        For each model:
        - build a clean reference estimator using all available observed points N_max
        - for each reduced number of observed points n, pad the remaining points with NaNs
        - infer posterior samples and compute:
                (1) mean absolute shift:
                    < |theta_hat(n) - theta_hat(N_max)| >
                (2) mean posterior uncertainty:
                    < sigma(n) >
                (3) mean standardized shift:
                    < |theta_hat(n) - theta_hat(N_max)| / sigma(N_max) >
        """

        from matplotlib.ticker import LogLocator

        n_reference_repeats = 4
        n_config_repeats = 2
        eps = 1e-12

        def stable_posterior_summary(model, x_input: Tensor, n_samples: int, n_repeats: int):
            means = []
            widths = []

            for _ in range(n_repeats):
                posterior = model.draw_parameters_from_predicted_posterior(
                    x_input, n_parameters=n_samples
                )
                mean, _ = Predictions.calculate_estimator(posterior)  # [B]
                width = Predictions._uncertainty(posterior)           # [B]
                means.append(mean)
                widths.append(width)

            means = torch.stack(means, dim=0)
            widths = torch.stack(widths, dim=0)

            mean_stable = means.mean(dim=0)
            width_stable = widths.mean(dim=0)
            return mean_stable, width_stable

        if raw_observed_data.ndim == 2:
            x_o = raw_observed_data.unsqueeze(0)
        else:
            x_o = raw_observed_data

        B, N_max, D = x_o.shape

        if n_points_list is None:
            effective_ns = np.linspace(max(1, N_max * 4 // 5), N_max, default_number_ns, dtype=int)
            effective_ns = np.unique(effective_ns).tolist()
        else:
            effective_ns = sorted(set(max(min(int(n), N_max), 1) for n in n_points_list))

        fig_shift, ax_shift = plt.subplots(1, 1, figsize=(5.8, 4.3))
        fig_width, ax_width = plt.subplots(1, 1, figsize=(5.8, 4.3))
        fig_std, ax_stdshift = plt.subplots(1, 1, figsize=(5.8, 4.3))

        plot_colors = ["blue", "red", "purple", "green", "black"]
        plot_markers = ["o", "s", "^", "D", "v", "P"]

        for i, model_dir in enumerate(model_dirs):
            model = Backup.load_model_for_inference_basic(
                directory=model_dir,
                device=device,
                use_best=True,
            )
            model_name = model_names[i]
            color = plot_colors[i % len(plot_colors)]
            marker = plot_markers[i % len(plot_markers)]

            x_ref_raw = x_o[:, :N_max, :]
            x_ref = model.normalizer.normalize_data(x_ref_raw)

            mean_ref, width_ref = stable_posterior_summary(
                model=model,
                x_input=x_ref,
                n_samples=n_posterior_samples,
                n_repeats=n_reference_repeats,
            )

            baseline_abs_shifts = []
            baseline_std_shifts = []
            baseline_widths = []

            for _ in range(n_reference_repeats):
                mean_full, width_full = stable_posterior_summary(
                    model=model,
                    x_input=x_ref,
                    n_samples=n_posterior_samples,
                    n_repeats=n_config_repeats,
                )

                abs_shift_0 = (mean_full - mean_ref).abs()
                std_shift_0 = abs_shift_0 / (width_ref + eps)

                baseline_abs_shifts.append(abs_shift_0.mean().item())
                baseline_std_shifts.append(std_shift_0.mean().item())
                baseline_widths.append(width_full.mean().item())

            mean_abs_shifts = []
            mean_widths = []
            mean_std_shifts = []

            for n in effective_ns:
                abs_shift_repeats = []
                width_repeats = []
                std_shift_repeats = []

                if n == N_max:
                    abs_shift_repeats = baseline_abs_shifts
                    width_repeats = baseline_widths
                    std_shift_repeats = baseline_std_shifts
                else:
                    x_pad = torch.full_like(x_o, float("nan"), device=x_o.device)
                    x_pad[:, :n, :] = x_o[:, :n, :]
                    x_n = model.normalizer.normalize_data(x_pad)

                    for _ in range(n_config_repeats):
                        mean_n, width_n = stable_posterior_summary(
                            model=model,
                            x_input=x_n,
                            n_samples=n_posterior_samples,
                            n_repeats=1,
                        )

                        abs_shift = (mean_n - mean_ref).abs()
                        std_shift = abs_shift / (width_ref + eps)

                        abs_shift_repeats.append(abs_shift.mean().item())
                        width_repeats.append(width_n.mean().item())
                        std_shift_repeats.append(std_shift.mean().item())

                mean_abs_shifts.append(float(np.mean(abs_shift_repeats)))
                mean_widths.append(float(np.mean(width_repeats)))
                mean_std_shifts.append(float(np.mean(std_shift_repeats)))

            ax_shift.plot(
                effective_ns,
                mean_abs_shifts,
                marker=marker,
                linestyle="-",
                linewidth=2.3,
                markersize=6,
                label=model_name,
                color=color,
            )

            ax_width.plot(
                effective_ns,
                mean_widths,
                marker=marker,
                linestyle="-",
                linewidth=2.3,
                markersize=6,
                label=model_name,
                color=color,
            )

            ax_stdshift.plot(
                effective_ns,
                mean_std_shifts,
                marker=marker,
                linestyle="-",
                linewidth=2.3,
                markersize=6,
                label=model_name,
                color=color,
            )

        ax_shift.invert_xaxis()
        ax_width.invert_xaxis()
        ax_stdshift.invert_xaxis()

        ax_shift.set_xlabel(r"Observed $N_e$", fontsize=AXIS_FONTSIZE)
        ax_shift.set_ylabel(
            r"$\left\langle |\hat{\theta}(N_e)-\hat{\theta}(N_{\max})| \right\rangle$",
            fontsize=AXIS_FONTSIZE,
        )
        ax_shift.tick_params(labelsize=TICK_FONTSIZE - 6, width=1.2)
        ax_shift.grid(True, alpha=0.35, linewidth=0.8)
        leg = ax_shift.legend(
            fontsize=LEGEND_FONTSIZE + 3,
            frameon=True,
            framealpha=0.55,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
            loc="best",
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        fig_shift.tight_layout()
        plt.figure(fig_shift.number)
        plt.savefig(PLOTS_DIR / "viva" / "robust_npoints_shift.pdf")

        ax_width.set_xlabel(r"Observed $N_e$", fontsize=AXIS_FONTSIZE)
        ax_width.set_ylabel(
            r"$\langle \sigma(N_e) \rangle$",
            fontsize=AXIS_FONTSIZE,
        )
        ax_width.set_yscale("log")

        # Major/minor ticks on log y-axis, with same size/thickness
        ax_width.yaxis.set_major_locator(LogLocator(base=10))
        #ax_width.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax_width.tick_params(axis="y", which="major", labelsize=TICK_FONTSIZE - 4, width=1.2, length=6)
        ax_width.tick_params(axis="y", which="minor", width=1.2, length=6, labelsize=TICK_FONTSIZE - 4)

        # x-axis ticks same as noise_fixed
        ax_width.tick_params(axis="x", which="major", labelsize=TICK_FONTSIZE - 2, width=1.2)

        ax_width.grid(True, alpha=0.35, linewidth=0.8, which="both")
        leg = ax_width.legend(
            fontsize=LEGEND_FONTSIZE + 3,
            frameon=True,
            framealpha=0.55,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
            loc="best",
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        fig_width.tight_layout()
        plt.figure(fig_width.number)
        plt.savefig(PLOTS_DIR / "viva" / "robust_npoints_width.pdf")

        ax_stdshift.set_xlabel(r"Observed $N_e$", fontsize=AXIS_FONTSIZE)
        ax_stdshift.set_ylabel(
            r"$\left\langle \frac{|\hat{\theta}(N_e)-\hat{\theta}(N_{\max})|}{\sigma(N_{\max})} \right\rangle$",
            fontsize=AXIS_FONTSIZE,
        )
        ax_stdshift.tick_params(labelsize=TICK_FONTSIZE - 2, width=1.2)
        ax_stdshift.grid(True, alpha=0.35, linewidth=0.8)
        ax_stdshift.axhline(1.0, linestyle=":", linewidth=1.5, color="black", alpha=0.8)
        ax_stdshift.axhline(0.5, linestyle=":", linewidth=1.0, color="black", alpha=0.5)
        leg = ax_stdshift.legend(
            fontsize=LEGEND_FONTSIZE + 3,
            frameon=True,
            framealpha=0.55,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
            loc="best",
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        fig_std.tight_layout()
        plt.figure(fig_std.number)
        plt.savefig(PLOTS_DIR / "viva" / "robust_npoints_standardized_shift.pdf")

        plt.show()

    @staticmethod
    def expected_coverage_test_many(
        model_dirs: list[Path],
        model_names: list[str],
        device: torch.device,
        x: Tensor,
        theta: Tensor,
        num_posterior_samples: int,
        path: Path = None,
        num_levels: int = 100,
    ):
        """
        Expected Coverage Test (ECT) for several neural networks on the same plot.

        For each model:
        - load the trained model from disk
        - run SBC on the same simulated pairs (theta_i, x_i)
        - compute the empirical coverage curve from the SBC ranks
        - overlay all curves on the same plot

        Notes
        -----
        - All models are evaluated on the same (x, theta), which is the correct way to compare them.
        - The plotted curve is:
            empirical coverage vs nominal level
        - The diagonal corresponds to perfect calibration.
        """

        def empirical_coverage_from_ranks(
            ranks: Tensor,
            num_posterior_samples: int,
            num_levels: int = 100,
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Convert SBC ranks into an expected coverage curve.

            Parameters
            ----------
            ranks:
                Tensor of shape [N] or [N, P]
                N = number of SBC samples
                P = number of parameters
            num_posterior_samples:
                Number of posterior samples used in SBC
            num_levels:
                Number of nominal coverage levels between 0 and 1

            Returns
            -------
            nominal_levels:
                Array of nominal levels in [0, 1]
            empirical:
                Empirical coverage averaged over parameters if P > 1
            """
            ranks_np = ranks.detach().cpu().numpy() if isinstance(ranks, torch.Tensor) else np.asarray(ranks)

            if ranks_np.ndim == 1:
                ranks_np = ranks_np[:, None]  # [N, 1]

            nominal_levels = np.linspace(0.0, 1.0, num_levels)
            max_rank = num_posterior_samples + 1

            empirical_per_param = []
            for j in range(ranks_np.shape[1]):
                r = ranks_np[:, j]
                empirical_j = []
                for alpha in nominal_levels:
                    threshold = int(np.floor(alpha * max_rank))
                    empirical_j.append(np.mean(r <= threshold))
                empirical_per_param.append(empirical_j)

            empirical_per_param = np.asarray(empirical_per_param)  # [P, num_levels]
            empirical = empirical_per_param.mean(axis=0)           # average over parameters

            return nominal_levels, empirical

        if len(model_dirs) != len(model_names):
            raise ValueError("model_dirs and model_names must have the same length.")

        fig, ax = plt.subplots(figsize=(5.8, 4.3))

        plot_colors = ["blue", "red", "purple", "brown", "black"]
        for i, model_dir in enumerate(model_dirs):
            model = Backup.load_model_for_inference_basic(
                directory=model_dir,
                device=device,
                use_best=True,
            )
            model_name = model_names[i]
            color = plot_colors[i % len(plot_colors)]

            x_norm = model.normalizer.normalize_data(x[:, :model.n_points])
            theta_norm = model.normalizer.normalize_parameters(theta)

            ranks, dap_samples = run_sbc(
                theta_norm,
                x_norm  ,
                model.posterior,
                reduce_fns=lambda theta, x: -model.posterior.log_prob(theta, x),
                num_posterior_samples=num_posterior_samples,
                use_batched_sampling=False,
                num_workers=4,
            )

            nominal_levels, empirical_coverage = empirical_coverage_from_ranks(
                ranks=ranks,
                num_posterior_samples=num_posterior_samples,
                num_levels=num_levels,
            )

            ax.plot(
                nominal_levels,
                empirical_coverage,
                linewidth=2.3,
                label=model_name,
                color=color,
            )

        # "Acceptable calibration" band
        eps = 0.1  # largeur de tolérance (à ajuster)

        x_band = np.linspace(0.0, 1.0, 200)
        y_lower = np.clip(x_band - eps, 0.0, 1.0)
        y_upper = np.clip(x_band + eps, 0.0, 1.0)

        ax.fill_between(
            x_band,
            y_lower,
            y_upper,
            color="green",
            alpha=0.4,
            label="Acceptable calibration",
        )

        ax.set_xlabel("Nominal coverage level", fontsize=AXIS_FONTSIZE-1)
        ax.set_ylabel("Empirical coverage", fontsize=AXIS_FONTSIZE-1)
        ax.tick_params(labelsize=TICK_FONTSIZE - 6, width=1.2)
        ax.grid(True, alpha=0.35, linewidth=0.8)

        leg = ax.legend(
            fontsize=LEGEND_FONTSIZE +1,
            frameon=True,
            framealpha=0.55,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")

        fig.tight_layout()

        if path is None:
            fig.show()
        else:
            fig.savefig(path)


    @staticmethod
    def simulation_based_calibration_many(
        model_dirs: list[Path],
        model_names: list[str],
        device: torch.device,
        x: Tensor,
        theta: Tensor,
        num_posterior_samples: int,
        path: Path = None,
        num_levels: int = 100,
    ):
        """
        Simulation-Based Calibration (SBC) for several neural networks on the same plot.

        For each model:
        - load the trained model from disk
        - normalize x and theta with the model normalizer
        - run SBC on the same simulated pairs (theta_i, x_i)
        - compute the empirical rank CDF
        - overlay all CDF curves on the same plot

        Notes
        -----
        - All models are evaluated on the same (x, theta), which is the correct way to compare them.
        - The green band indicates an acceptable calibration region around the ideal diagonal.
        - Systematic deviations indicate posterior bias, overconfidence, or underconfidence.
        """

        def empirical_rank_cdf(
            ranks: Tensor,
            num_posterior_samples: int,
            num_levels: int = 100,
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Convert SBC ranks into an empirical rank CDF.

            Parameters
            ----------
            ranks:
                Tensor of shape [N] or [N, P]
                N = number of SBC samples
                P = number of parameters
            num_posterior_samples:
                Number of posterior samples used in SBC
            num_levels:
                Number of x-points used to draw the CDF

            Returns
            -------
            nominal_levels:
                Array in [0, 1]
            empirical_cdf:
                Empirical CDF averaged over parameters if P > 1
            """
            ranks_np = ranks.detach().cpu().numpy() if isinstance(ranks, torch.Tensor) else np.asarray(ranks)

            if ranks_np.ndim == 1:
                ranks_np = ranks_np[:, None]

            nominal_levels = np.linspace(0.0, 1.0, num_levels)
            max_rank = num_posterior_samples + 1

            cdf_per_param = []
            for j in range(ranks_np.shape[1]):
                r = ranks_np[:, j]
                cdf_j = []
                for u in nominal_levels:
                    threshold = int(np.floor(u * max_rank))
                    cdf_j.append(np.mean(r <= threshold))
                cdf_per_param.append(cdf_j)

            cdf_per_param = np.asarray(cdf_per_param)
            empirical_cdf = cdf_per_param.mean(axis=0)
            return nominal_levels, empirical_cdf

        if len(model_dirs) != len(model_names):
            raise ValueError("model_dirs and model_names must have the same length.")

        fig, ax = plt.subplots(figsize=(5.8, 4.3))

        # Acceptable calibration band
        eps = 0.1
        x_band = np.linspace(0.0, 1.0, 200)
        y_lower = np.clip(x_band - eps, 0.0, 1.0)
        y_upper = np.clip(x_band + eps, 0.0, 1.0)
        ax.fill_between(
            x_band,
            y_lower,
            y_upper,
            color="green",
            alpha=0.4,
            label="Acceptable calibration",
            zorder=0,
        )

        plot_colors = ["blue", "red", "purple", "brown", "black"]

        for i, model_dir in enumerate(model_dirs):
            model = Backup.load_model_for_inference_basic(
                directory=model_dir,
                device=device,
                use_best=True,
            )
            model_name = model_names[i]
            color = plot_colors[i % len(plot_colors)]

            x_norm = model.normalizer.normalize_data(x[:, :model.n_points])
            theta_norm = model.normalizer.normalize_parameters(theta)

            ranks, dap_samples = run_sbc(
                theta_norm,
                x_norm,
                model.posterior,
                num_posterior_samples=num_posterior_samples,
                use_batched_sampling=False,
                num_workers=4,
            )

            nominal_levels, empirical_cdf = empirical_rank_cdf(
                ranks=ranks,
                num_posterior_samples=num_posterior_samples,
                num_levels=num_levels,
            )

            ax.plot(
                nominal_levels,
                empirical_cdf,
                linewidth=2.3,
                label=model_name,
                color=color,
                zorder=2,
            )

        ax.set_xlabel("Rank quantile", fontsize=AXIS_FONTSIZE-1)
        ax.set_ylabel("Cumulative fraction", fontsize=AXIS_FONTSIZE-1)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE - 6, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.35, linewidth=0.8)

        leg = ax.legend(
            fontsize=LEGEND_FONTSIZE + 1,
            frameon=True,
            framealpha=0.55,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")

        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            fig.savefig(path)


    @staticmethod
    def tarp_test_many(
        model_dirs: list[Path],
        model_names: list[str],
        device: torch.device,
        x: Tensor,
        theta: Tensor,
        num_posterior_samples: int,
        path: Path = None,
    ):
        """
        TARP test for several neural networks on the same plot.

        For each model:
        - load the trained model from disk
        - normalize x and theta with the model normalizer
        - run TARP on the same simulated pairs (theta_i, x_i)
        - overlay all TARP curves on the same plot

        Notes
        -----
        - All models are evaluated on the same (x, theta), which is the right way to compare them.
        - The green band indicates an acceptable calibration region around the ideal diagonal.
        - ATC should be close to 0.
        - KS p-value should ideally be > 0.05.
        """

        if len(model_dirs) != len(model_names):
            raise ValueError("model_dirs and model_names must have the same length.")

        fig, ax = plt.subplots(figsize=(5.8, 4.3))

        # Acceptable calibration band
        eps = 0.1
        x_band = np.linspace(0.0, 1.0, 200)
        y_lower = np.clip(x_band - eps, 0.0, 1.0)
        y_upper = np.clip(x_band + eps, 0.0, 1.0)
        ax.fill_between(
            x_band,
            y_lower,
            y_upper,
            color="green",
            alpha=0.4,
            label="Acceptable calibration",
            zorder=0,
        )

        plot_colors = ["blue", "red", "purple", "brown", "black"]

        for i, model_dir in enumerate(model_dirs):
            model = Backup.load_model_for_inference_basic(
                directory=model_dir,
                device=device,
                use_best=True,
            )
            model_name = model_names[i]
            color = plot_colors[i % len(plot_colors)]

            x_norm = model.normalizer.normalize_data(x[:, :model.n_points])
            theta_norm = model.normalizer.normalize_parameters(theta)

            ecp, alpha = run_tarp(
                theta_norm,
                x_norm,
                model.posterior,
                references=None,
                num_posterior_samples=num_posterior_samples,
                use_batched_sampling=False,
                num_workers=4,
            )

            atc, ks_pval = check_tarp(ecp, alpha)
            print(f"{model_name}  |  ATC: {atc:.4g}  |  KS p-value: {ks_pval:.4g}")

            ax.plot(
                alpha,
                ecp,
                linewidth=2.3,
                color=color,
                label=f"{model_name}",
                zorder=2,
            )

        ax.set_xlabel("Credible level", fontsize=AXIS_FONTSIZE-1)
        ax.set_ylabel("Empirical percentile", fontsize=AXIS_FONTSIZE-1)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE - 6, width=1.2)
        ax.grid(True, alpha=0.35, linewidth=0.8)

        leg = ax.legend(
            fontsize=LEGEND_FONTSIZE + 1,
            frameon=True,
            framealpha=0.55,
            handlelength=1.5,
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")

        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            fig.savefig(path)