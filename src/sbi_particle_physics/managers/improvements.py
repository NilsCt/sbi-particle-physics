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
    def _plot_width_by(x_values: np.ndarray | list, width: np.ndarray | list, x_label: str, curve_label : str) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray]:
        x_values = np.array(x_values, dtype=float)
        width = np.array(width, dtype=float)
        order = np.argsort(x_values)
        x_values = x_values[order]
        width = width[order]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_values, width, marker="o", linestyle="-", label=curve_label)
        ax.set_xlabel(x_label, fontsize=AXIS_FONTSIZE)
        ax.set_ylabel(r"$\langle \sigma \rangle$", fontsize=AXIS_FONTSIZE - 4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
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
            model: Model = Backup.load_model_for_inference_basic(directory=model_dir, device=device)
            n_points = model.n_points
            observed_data = model.normalizer.normalize_data(raw_observed_data)
            posterior_samples = model.draw_parameters_from_predicted_posterior(observed_data[:,:n_points], n_parameters=n_posterior_samples)
            avg_width = Predictions.average_uncertainty(posterior_samples)
            n_points_list.append(n_points)
            avg_widths.append(avg_width)
        fig, ax, n_points_arr, avg_widths_arr = Improvements._plot_width_by(n_points_list, avg_widths, r"$n_{\mathrm{points}}$", "Neural network")
        N_ref = n_points_arr[0]
        width_ref = avg_widths_arr[0]
        trend_1_over_sqrtN = width_ref * np.sqrt(N_ref / n_points_arr)
        ax.plot(n_points_arr, trend_1_over_sqrtN, linestyle="--", color="black", label=r"$\propto 1/\sqrt{n_{\mathrm{points}}}$")
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
            model: Model = Backup.load_model_for_inference_basic(directory=model_dir, device=device)
            n_points = model.n_points
            observed_data = model.normalizer.normalize_data(raw_observed_data)[:,:n_points]
            na = np.linspace(n_points//2, n_points, 8, dtype=int) # don't go below n_points/2 or the nn won't be able to infere a posterior
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
            model: Model = Backup.load_model_for_inference_basic(directory=model_dir, device=device)
            n_files = len(model.data_files_paths)
            observed_data = model.normalizer.normalize_data(raw_observed_data)
            posterior_samples = model.draw_parameters_from_predicted_posterior(observed_data, n_parameters=n_posterior_samples)
            avg_width = Predictions.average_uncertainty(posterior_samples)
            n_files_list.append(n_files)
            avg_widths.append(avg_width)
        fig, ax, n_files_arr, avg_widths_arr = Improvements._plot_width_by(n_files_list, avg_widths, r"$n_{\mathrm{files}}$", "Neural network")
        fig.show()
        fig, ax = Improvements._plot_width_by_quantify(n_files_list, avg_widths, r"$1/n_{\mathrm{files}}$", "Neural network")
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
            observed_data = model.normalizer.normalize_data(raw_observed_data)
            posterior_samples = model.draw_parameters_from_predicted_posterior(observed_data, n_parameters=n_posterior_samples)
            avg_width = Predictions.average_uncertainty(posterior_samples)
            n_epochs_list.append(n_epochs)
            avg_widths.append(avg_width)
        fig, ax, n_epochs_arr, avg_widths_arr = Improvements._plot_width_by(n_epochs_list, avg_widths, r"$n_{\mathrm{epochs}}$", "Neural network")
        fig.show()
        fig, ax = Improvements._plot_width_by_quantify(n_epochs_list, avg_widths, r"$1/n_{\mathrm{epochs}}$", "Neural network", ignore_n_first_points=ignore_n_first_points)
        fig.show()


    @staticmethod
    def _bar_plot(model_names : list[str], values : list[float], ylabel : str):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(model_names, values)
        ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE)
        ax.tick_params(axis="x", rotation=30, labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
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
    def compare_models(model_dirs: list[Path], device: torch.device, raw_observed_data: Tensor, n_posterior_samples: int = 1000,):
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
        model_names = []
        avg_widths = []
        avg_log_contractions = []
        avg_info_gains = []
        avg_entropies = []
        all_widths = []
        robust_cv = []
        robust_worst_ratio = []
        robust_quantile_ratio = []
        for model_dir in model_dirs:
            model = Backup.load_model_for_inference_basic(directory=model_dir, device=device)
            name = model_dir.name
            model_names.append(name)
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
        # todo fix radar plot

    @staticmethod
    def plot_drift_by_noise(model_dirs: list[Path], device: torch.device, raw_observed_data: Tensor, noise_levels: list[float], n_posterior_samples: int = 1000):
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
        for model_dir in model_dirs:
            model = Backup.load_model_for_inference_basic(directory=model_dir, device=device)
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
            ax_drift.plot(noise_levels, drifts, marker="o", linewidth=2, label=model_name)
            ax_width.plot(noise_levels, widths, marker="o", linewidth=2, label=model_name)

        ax_drift.set_xlabel(r"Noise amplitude $\delta$", fontsize=AXIS_FONTSIZE)
        ax_drift.set_ylabel(r"$\|\hat{\theta}(\delta)-\hat{\theta}(0)\|$", fontsize=AXIS_FONTSIZE)
        ax_drift.tick_params(labelsize=TICK_FONTSIZE)
        ax_drift.grid(alpha=0.3)
        ax_drift.legend(fontsize=LEGEND_FONTSIZE)

        ax_width.set_xlabel(r"Noise amplitude $\delta$", fontsize=AXIS_FONTSIZE)
        ax_width.set_ylabel(r"$\langle \sigma(\delta) \rangle$", fontsize=AXIS_FONTSIZE)
        ax_width.tick_params(labelsize=TICK_FONTSIZE)
        ax_width.grid(alpha=0.3)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_robust_npoints(model_dirs: list[Path], device: torch.device, raw_observed_data: Tensor, n_points_list: list[int] | None = None, n_posterior_samples: int = 1000, default_number_ns : int = 20):
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
        for model_dir in model_dirs:
            model = Backup.load_model_for_inference_basic(directory=model_dir, device=device)
            model_name = model_dir.name
            x_ref = model.normalizer.normalize_data(x_o)
            posterior_ref = model.draw_parameters_from_predicted_posterior(x_ref, n_parameters=n_posterior_samples)
            mean_ref, _ = Predictions.calculate_estimator(posterior_ref)
            drifts = []
            widths = []
            effective_ns = n_points_list
            if effective_ns is None:
                effective_ns = np.linspace(N_max//2, N_max, default_number_ns, dtype=int)
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
            ax_drift.plot(effective_ns, drifts, marker="o", linewidth=2, label=model_name)
            ax_width.plot(effective_ns, widths, marker="o", linewidth=2, label=model_name)

        ax_drift.set_xlabel(r"Observed $n_{\mathrm{points}}$", fontsize=AXIS_FONTSIZE)
        ax_drift.set_ylabel(r"$\|\hat{\theta}(n)-\hat{\theta}(N_{\max})\|$", fontsize=AXIS_FONTSIZE)
        ax_drift.tick_params(labelsize=TICK_FONTSIZE)
        ax_drift.grid(alpha=0.3)
        ax_drift.legend(fontsize=LEGEND_FONTSIZE)
        ax_width.set_xlabel(r"Observed $n_{\mathrm{points}}$", fontsize=AXIS_FONTSIZE)
        ax_width.set_ylabel(r"$\langle \sigma(n) \rangle$", fontsize=AXIS_FONTSIZE)
        ax_width.tick_params(labelsize=TICK_FONTSIZE)
        ax_width.grid(alpha=0.3)
        ax_width.set_yscale("log")
        fig.tight_layout()
        plt.show()

    # todo ajouter courbe de référence 1/sqrt(N)
