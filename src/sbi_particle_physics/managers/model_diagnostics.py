import matplotlib
import matplotlib.pyplot as plt
from pyparsing import line
import torch
from torch import Tensor
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.plotter import Plotter
import numpy as np
from sbi.diagnostics import run_sbc
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics import run_tarp, check_tarp
from sbi.analysis.plot import plot_tarp
from sbi.diagnostics.misspecification import calc_misspecification_logprob
from sbi.inference.trainers.marginal import MarginalTrainer
from sbi.diagnostics.misspecification import calc_misspecification_mmd
from sbi.diagnostics.lc2st import LC2ST
from sbi.analysis.plot import pp_plot_lc2st
from sbi.analysis import pairplot
from sbi_particle_physics.config import BLUE_COLOR, LEGEND_FONTSIZE, TICK_FONTSIZE, PLOTS_DIR, AXIS_FONTSIZE, GREEN_COLOR, RED_COLOR, REAL_DATA
from sbi_particle_physics.managers.predictions import Predictions
from pathlib import Path
from sbi_particle_physics.managers.real_data import RealData

class ModelDiagnostics:
    """
    Test, quantify and visualize the performance of a model
    Use conventional diagnostics such as SBC, PPC, Expected coverage, TARP, Missspecification test, LC2ST
    """

    @staticmethod
    def simulation_based_calibration(model: Model, x: Tensor, theta: Tensor, num_posterior_samples: int, path: Path = None):
        """
        Simulation-Based Calibration (SBC)

        Draws simulated parameter-data pairs (theta_i, x_i), infers the posterior
        p(theta | x_i), and computes the rank of the true parameter theta_i among
        posterior samples.

        If the posterior is well calibrated, the rank histogram should be close to
        uniform for each parameter. Systematic deviations indicate posterior bias,
        overconfidence, or underconfidence.
        """
        ranks, dap_samples = run_sbc(
            theta,
            x,
            model.posterior,
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=False,
            num_workers=4,
        )

        fig, ax = sbc_rank_plot(
            ranks,
            num_posterior_samples,
            num_bins=20,
            figsize=(5.5, 4),
            plot_type="cdf",
        )
        for line in ax.get_lines():
            line.set_color(RED_COLOR)
            line.set_linewidth(2.2)
        for coll in ax.collections:
            coll.set_facecolor(GREEN_COLOR)
            coll.set_edgecolor(GREEN_COLOR)
            coll.set_alpha(0.35)
        ax.set_xlabel("Rank", fontsize=AXIS_FONTSIZE, labelpad=0)
        ax.set_ylabel("Count", fontsize=AXIS_FONTSIZE, labelpad=0)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE - 4, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(
            fontsize=LEGEND_FONTSIZE - 3,
            frameon=True,
            framealpha=0.55,
            handlelength=1.3,
            handleheight=0.6,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.2,
        )
        if leg is not None:
            leg.get_frame().set_linewidth(0.7)
            leg.get_frame().set_facecolor("white")
        plt.tight_layout()
        if path is None:
            fig.show()
        else:
            fig.savefig(path)

    #@staticmethod
    #def _summary_stats(x):
    # x shape: (n_points, D)
    #    return torch.stack([
    #        x.mean(dim=0),
    #        x.std(dim=0),
    #    ], dim=0)  # shape (2, D)

    @staticmethod
    def _summary_stats(x : Tensor) -> Tensor:
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        centered = x - mean
        skew = (centered**3).mean(dim=0) / (std**3 + 1e-8)
        kurt = (centered**4).mean(dim=0) / (std**4 + 1e-8)
        q10 = torch.quantile(x, 0.10, dim=0)
        q25 = torch.quantile(x, 0.25, dim=0)
        q50 = torch.quantile(x, 0.50, dim=0)
        q75 = torch.quantile(x, 0.75, dim=0)
        q90 = torch.quantile(x, 0.90, dim=0)
        xmin = x.min(dim=0).values
        xmax = x.max(dim=0).values
        cov = torch.cov(x.T) # correlations
        std_outer = std[:, None] * std[None, :]
        corr = cov / (std_outer + 1e-8)
        corr_features = corr[torch.triu(torch.ones_like(corr), diagonal=1) == 1]
        return torch.cat([ mean, std, skew, kurt, q10, q25, q50, q75, q90, xmin, xmax, corr_features ])

    @staticmethod
    def posterior_predictive_checks(model : Model, x_o : Tensor, n_samples : int, n_points : int, path : Path = None):
        """
        Posterior Predictive Checks (PPC)
        Generates a parameter θ, simulates data x_i ~ p(x | θ),
        infers posteriors p(θ | x_i), samples θ'_j ~ p(θ | x_i),
        and simulates posterior predictive data x'_j ~ p(x | θ'_j).

        Compares observed data x_i with posterior predictive data x'_j
        to assess whether the inferred posteriors can reproduce the observed data.
        """
        x_pp, theta_pp = model.simulate_data_from_predicted_posterior(x_o, n_samples, n_points)
        stats_pp = []
        for i in range(x_pp.shape[0]):
            stats_pp.append(ModelDiagnostics._summary_stats(x_pp[i]))
        stats_pp = torch.stack(stats_pp)
        stats_obs = ModelDiagnostics._summary_stats(x_o)
        S, D = stats_obs.shape
        fig, axes = plt.subplots(S, D, figsize=(3 * D, 3 * S), squeeze=False)
        for s in range(S):
            for d in range(D):
                ax = axes[s, d]
                ax.violinplot(stats_pp[:, s, d].cpu().numpy(), showmeans=False, showmedians=True)
                ax.scatter(1, stats_obs[s, d].item(), color="red", zorder=3)
                ax.set_xticks([])
                ax.set_title(rf"$s_{s}(x_{d})$")
        fig.tight_layout()
        if path is None:
            fig.show()
        else:
            fig.savefig(path)


    @staticmethod
    def expected_coverage_test(model : Model, x : Tensor, theta : Tensor, num_posterior_samples : int, path : Path = None):
        """
        Expected Coverage Test (ECT)
        Generates parameters θ_i, simulates data x_i ~ p(x | θ_i),
        infers posteriors p(θ | x_i), and checks whether θ_i lies
        within posterior credible intervals at nominal coverage levels.

        The empirical coverage is compared to the nominal coverage
        to detect posterior over- or under-confidence.
        """
        #x, theta = model.simulate_data(n_samples, n_points)
        ranks, dap_samples = run_sbc(
            theta,
            x,
            model.posterior,
            reduce_fns=lambda theta, x: -model.posterior.log_prob(theta, x),
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=False,  # `True` can give speed-ups, but can cause memory issues.
            num_workers=4,
        )
        fig, ax = sbc_rank_plot(
            ranks,
            num_posterior_samples,
            plot_type="cdf",
            num_bins=20,
            figsize=(5.5, 4),
            parameter_labels=["Model"]
        )
        for line in ax.get_lines():
            line.set_color(RED_COLOR)   # ta couleur choisie
            line.set_linewidth(3) 
        for coll in ax.collections:
            coll.set_facecolor(GREEN_COLOR)   # ta couleur
            coll.set_alpha(0.4)  
        ax.set_xlabel("Nominal level", fontsize=AXIS_FONTSIZE-2, labelpad=0) # , fontweight='bold'
        ax.set_ylabel("Empirical coverage", fontsize=AXIS_FONTSIZE-3, labelpad=0) # , fontweight='bold'
        ax.tick_params(labelsize=TICK_FONTSIZE-6, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(
            fontsize=LEGEND_FONTSIZE-3,
            frameon=True,
            framealpha=0.55,
            handlelength=1.3,
            handleheight=0.6,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.2
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor('white')
        plt.tight_layout
        plt.savefig(PLOTS_DIR / "poster" / "image_calibration.svg")
        if path is None:
            fig.show()
        else:
            fig.savefig(path)

    @staticmethod
    def tarp_test(model, x, theta, num_posterior_samples: int, path=None):
        ecp, alpha = run_tarp(
            theta,
            x,
            model.posterior,
            references=None,
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=False,
            num_workers=4,
        )
        """
        TARP Test
        Generates parameters θ_i, simulates data x_i ~ p(x | θ_i),
        infers posteriors p(θ | x_i), and computes Expected Credible Percentiles (ECP)
        of the true parameters under the inferred posteriors.

        The ECP distribution should be uniform.
        Deviations indicate bias or miscalibration of the posterior.
        """
        # the tarp method returns the ECP values for a given set of alpha coverage levels.
        atc, ks_pval = check_tarp(ecp, alpha)
        print("ATC:", atc, "(should be close to 0)")
        print("KS p-value:", ks_pval, "(should be > 0.05)")

        fig, ax = plot_tarp(ecp, alpha)
        ax.set_xlabel("Ideal coverage", fontsize=AXIS_FONTSIZE+2)
        ax.set_ylabel("Expected coverage", fontsize=AXIS_FONTSIZE+2)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE-2, width=1.2)
        ax.grid(True, alpha=0.35, linewidth=0.8)
        lines = ax.get_lines()
        if len(lines) >= 1:
            lines[0].set_color(RED_COLOR)
            lines[0].set_linewidth(2.5)
            lines[0].set_label("TARP")
        if len(lines) >= 2:
            lines[1].set_color(BLUE_COLOR)
            lines[1].set_linestyle("--")
            lines[1].set_linewidth(1.6)
            lines[1].set_label("Ideal")
        leg = ax.legend(fontsize=LEGEND_FONTSIZE+7, frameon=True, framealpha=0.55, handlelength=1.4, handletextpad=0.4, borderpad=0.3, labelspacing=0.25)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            fig.savefig(path)


    @staticmethod
    def _flatten_stats(stats_2d: torch.Tensor) -> torch.Tensor:
        # stats_2d: (2, D) -> (2D,)
        return stats_2d.reshape(-1)

    @staticmethod
    def misspecification_test_mmd(x_train: Tensor, x_o: Tensor, path: Path = None, model: Model | None = None):
        """
        Misspecification test based on Maximum Mean Discrepancy (MMD).

        Compares the observed dataset x_o to the reference distribution defined by
        x_train using an MMD two-sample test. The comparison can be performed
        directly in data space ("x_space") or in the learned neural embedding
        ("embedding") when a model is provided.

        A large observed MMD relative to the baseline distribution leads to a small
        p-value and suggests model misspecification, i.e. that the observed data are
        unlikely to have been generated by the same distribution as the training
        simulations under the chosen representation.
        """
        mode = "x_space" if model is None else "embedding"
        inference = None if model is None else model.neural_network
        p_val, (mmds_baseline, mmd) = calc_misspecification_mmd(
            inference=inference,
            x_obs=x_o.unsqueeze(0),
            x=x_train,
            mode=mode,
            # n_shuffle=50,
            # max_samples=100,
        )
        print("MMD p-value:", p_val)  # should typically be > 0.05 to avoid evidence for misspecification

        plt.figure(figsize=(5.5, 4), dpi=80)
        plt.hist(mmds_baseline.detach().cpu().numpy(), bins=50, alpha=0.4, color="blue", label="Simulator fluctuations")
        plt.axvline(mmd.item(), color="red", linewidth=2.5, label=r"Observed MMD")
        ax = plt.gca()
        ax.set_xlabel("MMD", fontsize=AXIS_FONTSIZE -3, labelpad=0)
        ax.set_ylabel("Count", fontsize=AXIS_FONTSIZE -3, labelpad=0)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE - 7, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE-1, frameon=True, framealpha=0.55, handlelength=1.3, handleheight=0.6, handletextpad=0.4, borderpad=0.3, labelspacing=0.2)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor("white")
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

    @staticmethod
    def many_posteriors(model : Model, true_parameters : Tensor, observed_samples : Tensor, parameter_component_index : int, x_min : int, x_max : int, n_cols: int = 6, n_rows: int = 5, bins: int = 40, figsize_per_plot=(3.0, 2.4), path : Path = None):
        """
        Plot many 1D posteriors in a grid to verify the accuracy of the predictions
        """
        n_plots = n_cols * n_rows
        n_points = model.n_points
        n_samples = 1000
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows), squeeze=False,)
        many_samples = model.draw_parameters_from_predicted_posterior(observed_samples, n_samples)
        
        for i in range(n_plots):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            true_parameter = true_parameters[i]
            samples = many_samples[i]
            ax.hist(samples[:,parameter_component_index], bins=bins, density=True, alpha=0.6, color="green")
            ax.set_xlim(x_min, x_max)
            ax.axvline(true_parameter[parameter_component_index], color="red", linestyle="--", linewidth=2.5)
            ax.tick_params(labelsize=TICK_FONTSIZE-10)
            ax.set_xlabel(f"$C_9$", fontsize=AXIS_FONTSIZE-4, labelpad=0)
            ax.grid(True, alpha=0.3)
        # Hide unused axes
        for j in range(n_plots, n_plots):
            axes[j // n_cols, j % n_cols].axis("off")
        # Global legend (once)
        handles = [
            plt.Line2D([], [], color="green", alpha=0.6, linewidth=8, label="posterior"),
            plt.Line2D([], [], color="red", linestyle="--", linewidth=2, label="True value"),
        ]
        #fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=LEGEND_FONTSIZE, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if path is None:
            fig.show()
        else:
            fig.savefig(path)



    @staticmethod
    def robustness_to_noise(model: Model, x_o_raw: Tensor, n_posterior_samples: int = 1000, deltas: list[float] | None = None, path : Path = None):
        """
        Diagnose robustness to small perturbations of the observed data.

        Adds Gaussian noise of increasing amplitude to x_o and measures:
            - average posterior width
            - information gain
            - log contraction
            - drift of the posterior mean
        A robust model should show smooth, monotonic degradation and low derivatives near delta=0.
        """
        if deltas is None: deltas = [0.0, 0.25, 0.5, 1.0, 2.0]
        device = model.device
        x_ref = model.normalizer.normalize_data(x_o_raw)
        posterior_ref = model.draw_parameters_from_predicted_posterior(x_ref, n_parameters=n_posterior_samples)
        mean_ref, _ = Predictions.calculate_estimator(posterior_ref)
        prior_samples = model.draw_parameters(n_parameters=n_posterior_samples, from_prior=True).squeeze(-1)
        avg_widths = []
        info_gains = []
        log_contrs = []
        estimator_drifts = []

        for delta in deltas:
            if delta == 0.0:
                x_delta = x_o_raw.clone()
            else:
                noise = delta * torch.randn_like(x_o_raw, device=device)
                x_delta = x_o_raw + noise
            # IMPORTANT Normalize AFTER noise addition
            x_delta = model.normalizer.normalize_data(x_delta)

            posterior = model.draw_parameters_from_predicted_posterior(x_delta, n_parameters=n_posterior_samples)
            avg_widths.append(Predictions.average_uncertainty(posterior))
            info_gains.append(Predictions.information_gain(prior_samples, posterior).mean().item())
            log_contrs.append(Predictions.log_contraction(prior_samples, posterior).mean().item())
            mean_delta, _ = Predictions.calculate_estimator(posterior)
            drift = torch.norm(mean_delta - mean_ref).item()
            estimator_drifts.append(drift)

        def _plot(y, ylabel, path: Path | None):
            plt.figure(figsize=(5.5, 4))
            plt.plot(deltas, y, marker="o", color=RED_COLOR, linewidth=2.2, markersize=5)
            ax = plt.gca()
            ax.set_xlabel(r"Noise amplitude $\delta$", fontsize=AXIS_FONTSIZE, labelpad=0)
            ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE, labelpad=0)
            ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE - 2, width=1.2)
            ax.locator_params(nbins=4)
            ax.grid(True, alpha=0.4, linewidth=0.8)
            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path)

        if path is not None: path.mkdir(parents=True, exist_ok=True)
        _plot(avg_widths, r"$\langle \sigma \rangle$", None if path is None else path / "width.pdf")
        _plot(info_gains, r"Information gain", None if path is None else path / "info.pdf")
        _plot(log_contrs, r"Log contraction", None if path is None else path / "contraction.pdf")
        _plot(estimator_drifts, r"$\|\hat{\theta}(\delta)-\hat{\theta}(0)\|$", None if path is None else path / "drift.pdf")

        print("=== Robustness to noise summary ===")
        for i, d in enumerate(deltas):
            print(
                f"δ={d:4.2f} | "
                f"width={avg_widths[i]:.3e} | "
                f"IG={info_gains[i]:.3f} | "
                f"logC={log_contrs[i]:.3f} | "
                f"drift={estimator_drifts[i]:.3e}"
            )


    @staticmethod
    def robustness_to_npoints(model: Model, x_o_raw: Tensor, n_posterior_samples: int = 1000, n_list: list[int] | None = None, use_random_subsample: bool = False, number_of_ns: int = 10, path : Path = None):
        """
        Diagnose robustness to fewer observed points by padding missing points with NaNs.

        For each n in n_list:
            - keep only n points (either first n, or random subsample)
            - pad the remaining points with NaN
            - infer posterior and compute metrics
        Measures:
            - average posterior width (68%)
            - information gain
            - log contraction
            - drift of posterior mean relative to n = n_max
        A robust model should degrade smoothly when n decreases:
            width ↑, info gain ↓, log contraction ↓, drift grows smoothly.
        """
        device = model.device
        if x_o_raw.ndim == 2: x_o_raw = x_o_raw.unsqueeze(0)
        B, N_max, D = x_o_raw.shape
        if n_list is None: # pas mettre en dessous de N/2
            n_list = np.linspace(int(N_max/2), N_max, number_of_ns).astype(int).tolist()
            n_list = sorted(list(set([max(1, n) for n in n_list])), reverse=True)
        x_ref = model.normalizer.normalize_data(x_o_raw) # Reference (maximum points)
        posterior_ref = model.draw_parameters_from_predicted_posterior(x_ref, n_parameters=n_posterior_samples)
        mean_ref, _ = Predictions.calculate_estimator(posterior_ref)
        prior_samples = model.draw_parameters(n_parameters=n_posterior_samples, from_prior=True).squeeze(-1)
        avg_widths = []
        info_gains = []
        log_contrs = []
        estimator_drifts = []

        for n in n_list:
            n = max(min(n, N_max), 1)
            x_pad = torch.full((B, N_max, D), float("nan"), device=device)
            if use_random_subsample:
                idx = torch.randperm(N_max, device=device)[:n]
                idx_sorted, _ = torch.sort(idx)
                x_pad[:, :n, :] = x_o_raw[:, idx_sorted, :]
            else:
                x_pad[:, :n, :] = x_o_raw[:, :n, :]
            x_n = model.normalizer.normalize_data(x_pad)
            posterior = model.draw_parameters_from_predicted_posterior(x_n, n_parameters=n_posterior_samples)
            avg_widths.append(Predictions.average_uncertainty(posterior))
            info_gains.append(Predictions.information_gain(prior_samples, posterior).mean().item())
            log_contrs.append(Predictions.log_contraction(prior_samples, posterior).mean().item())
            mean_n, _ = Predictions.calculate_estimator(posterior)
            drift = torch.norm(mean_n - mean_ref).item()
            estimator_drifts.append(drift)

        def _plot(y, ylabel, path: Path | None):
            plt.figure(figsize=(5.5, 4))
            plt.plot(n_list, y, marker="o", color=RED_COLOR, linewidth=2.2, markersize=5)
            ax = plt.gca()
            ax.set_xlabel(r"$N_e$", fontsize=AXIS_FONTSIZE, labelpad=0) # n_\mathrm{points}
            ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE, labelpad=0)
            ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE - 2, width=1.2)
            ax.locator_params(nbins=4)
            ax.grid(True, alpha=0.4, linewidth=0.8)
            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path)

        if path is not None: path.mkdir(parents=True, exist_ok=True)
        _plot(avg_widths, r"$\langle \sigma \rangle$", None if path is None else path / "width.pdf")
        _plot(info_gains, r"Information gain", None if path is None else path / "info.pdf")
        _plot(log_contrs, r"Log contraction", None if path is None else path / "contraction.pdf")
        _plot(estimator_drifts, r"$\|\hat{\theta}(n)-\hat{\theta}(N)\|$", None if path is None else path / "drift.pdf")

        print("=== Robustness to n_points summary ===")
        for i, n in enumerate(n_list):
            print(
                f"n={n:4d} | "
                f"width={avg_widths[i]:.3e} | "
                f"IG={info_gains[i]:.3f} | "
                f"logC={log_contrs[i]:.3f} | "
                f"drift={estimator_drifts[i]:.3e}"
            )


    @staticmethod
    def do_them_all(model : Model, subdirectory : Path, raw_data : Tensor, raw_parameters : Tensor, real_raw_data : Tensor, num_posterior_samples : int = 1000):
        # sensitive to changes of the config
        subdirectory.mkdir(parents=True, exist_ok=True)
        
        data = model.normalizer.normalize_data(raw_data[:,:model.n_points])
        parameters = model.normalizer.normalize_parameters(raw_parameters[:,:model.n_points])
        real_data = model.normalizer.normalize_data(real_raw_data[:model.n_points])

        #ModelDiagnostics.expected_coverage_test(model, data[:200], parameters[:200], num_posterior_samples=num_posterior_samples, path=subdirectory / "ect.pdf")

        #ModelDiagnostics.tarp_test(model, data[:200], parameters[:200], num_posterior_samples=num_posterior_samples, path=subdirectory / "tarp.pdf")

                    #ModelDiagnostics.misspecification_test_mmd(data[-50:-2], x_o=real_data, path=subdirectory / "miss_mmd.pdf")

        #ModelDiagnostics.misspecification_test_mmd(data[-1000:-2], x_o=real_data, path=subdirectory / "miss_mmd_embedding.pdf", model=model)

        #RealData.plot_real_data_posterior(model, real_data, path=subdirectory / "real_posterior.pdf", n_samples=num_posterior_samples)

        #ModelDiagnostics.many_posteriors(model, true_parameters=parameters[-50:], observed_samples=data[-50:], parameter_component_index=0, x_min=3, x_max=5, path=subdirectory / "many.pdf", n_cols=4, n_rows=4) # component 0 of the parameters (C_9)

        #ModelDiagnostics.simulation_based_calibration(model, data[:200], parameters[:200], num_posterior_samples=num_posterior_samples, path=subdirectory / "sbc.pdf")
        
        #ModelDiagnostics.robustness_to_npoints(model, x_o_raw=raw_data[-100:,:model.n_points], n_posterior_samples=num_posterior_samples, use_random_subsample=False, number_of_ns=20, path = subdirectory / "npoints")

        deltas = np.linspace(0.0, 0.3, 15).tolist()

        #ModelDiagnostics.robustness_to_noise(model, x_o_raw=raw_data[-100:,:model.n_points], n_posterior_samples=num_posterior_samples, deltas=deltas, path = subdirectory / "noise")
        
        RealData.calculate_best_estimator(model=model, path_real_data=REAL_DATA, n_parameters=1000, n_subsamples=200, sample_with_replacement=False, path=subdirectory)
        
        ##ModelDiagnostics.posterior_predictive_checks(model, x_o=data[-1], n_samples=200, n_points=model.n_points, path=subdirectory/ "ppc.pdf")