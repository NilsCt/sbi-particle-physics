import matplotlib.pyplot as plt
import torch
from torch import Tensor
from model import Model
from plotter import Plotter
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

class Diagnostics:
    """
    Test, quantify and visualize the performance of a model
    Use conventional diagnostics such as SBC, PPC, Expected coverage, TARP, Missspecification test, LC2ST
    """

    """
    Simulation-Based Calibration (SBC)
    Generates parameters θ_i from the prior, simulates data x_i ~ p(x | θ_i),
    infers posteriors p(θ | x_i), and evaluates the rank of θ_i within each posterior.

    The distribution of ranks should be uniform.
    Uniform rank histograms, KS tests, and χ² tests are used to detect
    systematic bias and posterior over- or under-confidence.
    """
    @staticmethod
    def simulation_based_calibration(model : Model, x : Tensor, theta : Tensor, num_posterior_samples : int):
        # x, theta = model.simulate_data(n_samples, n_points)
        ranks, dap_samples = run_sbc(
            theta,
            x,
            model.posterior,
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=False,  # `True` can give speed-ups, but can cause memory issues.
            num_workers=4,
        )
        fig, ax = sbc_rank_plot(
            ranks,
            num_posterior_samples,
            num_bins=20,
            figsize=(5, 3),
        )
        plt.show()

    @staticmethod
    def _summary_stats(x):
    # x shape: (n_points, D)
        return torch.stack([
            x.mean(dim=0),
            x.std(dim=0),
        ], dim=0)  # shape (2, D)

    """
    Posterior Predictive Checks (PPC)
    Generates a parameter θ, simulates data x_i ~ p(x | θ),
    infers posteriors p(θ | x_i), samples θ'_j ~ p(θ | x_i),
    and simulates posterior predictive data x'_j ~ p(x | θ'_j).

    Compares observed data x_i with posterior predictive data x'_j
    to assess whether the inferred posteriors can reproduce the observed data.
    """
    @staticmethod
    def posterior_predictive_checks(model : Model, x_o : Tensor, n_samples : int, n_points : int):
        x_pp, theta_pp = model.simulate_data_from_predicted_posterior(x_o, n_samples, n_points)
        stats_pp = []
        for i in range(x_pp.shape[0]):
            stats_pp.append(Diagnostics._summary_stats(x_pp[i]))
        stats_pp = torch.stack(stats_pp)
        stats_obs = Diagnostics._summary_stats(x_o)

        S, D = stats_obs.shape

        # fig, axes = plt.subplots(
        #     S, D,
        #     figsize=(3 * D, 3 * S),
        #     squeeze=False,
        # )

        # for s in range(S):
        #     for d in range(D):
        #         ax = axes[s, d]

        #         ax.boxplot(
        #             stats_pp[:, s, d].cpu().numpy(),
        #             vert=True,
        #             widths=0.6,
        #             showfliers=False,
        #         )

        #         # Observed statistic
        #         ax.scatter(
        #             1,
        #             stats_obs[s, d].item(),
        #             color="red",
        #             zorder=3,
        #             label="Observed" if (s == 0 and d == 0) else None,
        #         )

        #         ax.set_xticks([])
        #         ax.set_title(rf"$s_{s}(x_{d})$")

        # axes[0, 0].legend()
        # plt.tight_layout()
        # plt.show()

        fig, axes = plt.subplots(
            S, D,
            figsize=(3 * D, 3 * S),
            squeeze=False,
        )

        for s in range(S):
            for d in range(D):
                ax = axes[s, d]

                ax.violinplot(
                    stats_pp[:, s, d].cpu().numpy(),
                    showmeans=False,
                    showmedians=True,
                )

                ax.scatter(
                    1,
                    stats_obs[s, d].item(),
                    color="red",
                    zorder=3,
                )

                ax.set_xticks([])
                ax.set_title(rf"$s_{s}(x_{d})$")

        plt.tight_layout()
        plt.show()


    """
    Expected Coverage Test (ECT)
    Generates parameters θ_i, simulates data x_i ~ p(x | θ_i),
    infers posteriors p(θ | x_i), and checks whether θ_i lies
    within posterior credible intervals at nominal coverage levels.

    The empirical coverage is compared to the nominal coverage
    to detect posterior over- or under-confidence.
    """
    @staticmethod
    def expected_coverage_test(model : Model, x : Tensor, theta : Tensor, num_posterior_samples : int):
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
            figsize=(5, 3),
        )
        plt.show()

    """
    TARP Test
    Generates parameters θ_i, simulates data x_i ~ p(x | θ_i),
    infers posteriors p(θ | x_i), and computes Expected Credible Percentiles (ECP)
    of the true parameters under the inferred posteriors.

    The ECP distribution should be uniform.
    Deviations indicate bias or miscalibration of the posterior.
    """
    @staticmethod
    def tarp_test(model : Model, x : Tensor, theta : Tensor, num_posterior_samples : int):
        # the tarp method returns the ECP values for a given set of alpha coverage levels.
        ecp, alpha = run_tarp(
            theta,
            x,
            model.posterior,
            references=None,  # will be calculated automatically.
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=False,  # `True` can give speed-ups, but can cause memory issues.
            num_workers=4,
        )
        # Similar to SBC, we can check then check whether the distribution of ecp is close to
        # that of alpha.
        atc, ks_pval = check_tarp(ecp, alpha)
        print(atc, "Should be close to 0")
        print(ks_pval, "Should be larger than 0.05")
        plot_tarp(ecp, alpha)


    @staticmethod
    def _flatten_stats(stats_2d: torch.Tensor) -> torch.Tensor:
        # stats_2d: (2, D) -> (2D,)
        return stats_2d.reshape(-1)

    """
    Misspecification Test
    Model misspecification occurs when the true data-generating process
    differs from the assumed model, such that no parameter value
    can generate data consistent with the observations.

    Misspecification leads to systematically biased or misleading posteriors.
    """
    @staticmethod
    def misspecification_test(model: Model, x_train: Tensor, x_o: Tensor):

        # summaries: (N, 2, D) -> (N, 2D)
        summaries = torch.stack([
            Diagnostics._flatten_stats(Diagnostics._summary_stats(x))
            for x in x_train
        ], dim=0)  # (N, 2D)

        summary_o = Diagnostics._flatten_stats(Diagnostics._summary_stats(x_o))  # (2D,)

        # (optionnel mais utile) normaliser pour aider le flow
        mu = summaries.mean(dim=0, keepdim=True)
        std = summaries.std(dim=0, keepdim=True).clamp_min(1e-8)
        summaries_z = (summaries - mu) / std
        summary_o_z = (summary_o - mu.squeeze(0)) / std.squeeze(0)

        trainer = MarginalTrainer(density_estimator="NSF")
        trainer.append_samples(summaries_z)
        est = trainer.train()

        logp_train = est.log_prob(summaries_z).detach().cpu()
        logp_obs = est.log_prob(summary_o_z.unsqueeze(0)).item()

        p_value = (logp_train <= logp_obs).float().mean().item()
        reject_H0 = p_value < 0.05

        print(f"p-value: {p_value:.4f}, Reject H0 (misspecified): {reject_H0}")

        plt.figure(figsize=(6, 4))
        plt.hist(logp_train.numpy(), bins=50, alpha=0.6, label=r"$\log p(s(x))$")
        plt.axvline(logp_obs, color="red", label=r"$\log p(s(x_o))$")
        plt.xlabel(r"$\log p(s)$")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()



    """
    Misspecification Test using MMD
    Uses Maximum Mean Discrepancy (MMD) to measure the distance
    between the distribution of observed data and data simulated
    from the inferred posterior predictive distribution.

    Large MMD values indicate model misspecification,
    i.e. the model cannot reproduce the observed data distribution.
    """
    @staticmethod
    def misspecification_test_mmd(model : Model, x_train : Tensor, x_o : Tensor):
        summaries = torch.stack([
            Diagnostics._flatten_stats(Diagnostics._summary_stats(x))
            for x in x_train
        ])

        summary_o = Diagnostics._flatten_stats(
            Diagnostics._summary_stats(x_o)
        ).unsqueeze(0)

        p_val, (mmds_baseline, mmd) = calc_misspecification_mmd(
            inference=None,
            x_obs=summary_o,
            x=summaries,
            mode="x_space"
        )
        print("MMD p-value:", p_val) # needs to be larger than 0.05 to be sure there is no missspecification

        plt.figure(figsize=(6, 4), dpi=80)
        plt.hist(mmds_baseline.numpy(), bins=50, alpha=0.5, label="baseline")
        plt.axvline(mmd.item(), color="k", label=r"MMD(x, $x_o$)")
        plt.ylabel("Count")
        plt.xlabel("MMD")
        plt.legend()
        plt.show()

    """
    LC2ST Test (Local Classifier Two-Sample Test)
    Trains a classifier to distinguish observed data from data
    simulated using the inferred posterior predictive distribution.

    If the classifier performs better than chance, the model
    fails to reproduce the observed data, indicating misspecification
    or poor posterior quality.
    """
    @staticmethod
    def lc2st_test(model : Model, x_o : Tensor, n_samples : int, n_points : int):
        # x_o: single observation WITH batch dimension, shape: (1, n_points, features)
        posterior = model.posterior
        # Simulate calibration data. Should be at least in the thousands.
        prior_predictives, prior_samples = model.simulate_data(n_samples, n_points)

        # Generate one posterior sample for every prior predictive.
        post_samples_cal = posterior.sample_batched(
            (1,),
            x=prior_predictives,
            max_sampling_batch_size=10
        )[0]
        # Train the L-C2ST classifier.
        lc2st = LC2ST(
            thetas=prior_samples,
            xs=prior_predictives,
            posterior_samples=post_samples_cal,
            classifier="mlp",
            num_ensemble=1,
        )
        _ = lc2st.train_under_null_hypothesis()
        _ = lc2st.train_on_observed_data()

        # Note: x_o must have a batch-dimension. I.e. `x_o.shape == (1, observation_shape)`.
        # Ensure x_o is a tensor with batch dimension
        if not isinstance(x_o, torch.Tensor):
            x_o = torch.as_tensor(x_o)
        if x_o.dim() == 2:  # If shape is (n_points, features), add batch dim
            x_o = x_o.unsqueeze(0)  # -> (1, n_points, features)

        post_samples_star = posterior.sample((10_000,), x=x_o)
        probs_data, scores_data = lc2st.get_scores(
            theta_o=post_samples_star,
            x_o=x_o,
            return_probs=True,
            trained_clfs=lc2st.trained_clfs
        )
        probs_null, scores_null = lc2st.get_statistics_under_null_hypothesis(
            theta_o=post_samples_star,
            x_o=x_o,
            return_probs=True,
        )
        conf_alpha = 0.05
        p_value = lc2st.p_value(post_samples_star, x_o)
        reject = lc2st.reject_test(post_samples_star, x_o, alpha=conf_alpha)

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        quantiles = np.quantile(scores_null, [0, 1-conf_alpha])
        ax.hist(scores_null, bins=50, density=True, alpha=0.5, label="Null")
        ax.axvline(scores_data, color="red", label="Observed")
        ax.axvline(quantiles[0], color="black", linestyle="--", label="95% CI")
        ax.axvline(quantiles[1], color="black", linestyle="--")
        ax.set_xlabel("Test statistic")
        ax.set_ylabel("Density")
        ax.set_title(f"p-value = {p_value:.3f}, reject = {reject}")
        fig.savefig("plots/lc2st/plot1.png") # doesnt't show but save the fig as lc2st is very slow and is executed offline

        pp_plot_lc2st(
            probs=[probs_data],
            probs_null=probs_null,
            conf_alpha=0.05,
            labels=["Classifier probabilities \n on observed data"],
            colors=["red"],
        )
        fig = plt.gcf()
        fig.savefig("plots/lc2st/plot2.png", bbox_inches="tight")
        plt.close(fig)

    """
    Plot many 1D posteriors in a grid to verify the accuracy of the predictions
    """
    @staticmethod
    def many_posteriors(
        model : Model,
        parameter_component_index : int,
        n_cols: int = 6,
        n_rows: int = 5,
        bins: int = 40,
        figsize_per_plot=(3.0, 2.4),
        savepath: str | None = None,
    ):
        n_plots = n_cols * n_rows
        n_points = 1000
        n_samples = 1000

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize_per_plot[0] * n_cols,
                     figsize_per_plot[1] * n_rows),
            squeeze=False,
        )

        for i in range(n_plots):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            true_parameter, observed_sample = model.get_random_true_parameter(n_points)
            samples = model.draw_parameters_from_predicted_posterior(observed_sample, n_samples)

            ax.hist(
                samples[:,parameter_component_index],
                bins=bins,
                density=True,
                alpha=0.6,
                color="green",
            )

            ax.axvline(
                true_parameter[parameter_component_index],
                color="red",
                linestyle="--",
                linewidth=2,
            )

            ax.tick_params(labelsize=Plotter.tick_fontsize)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for j in range(n_plots, n_plots):
            axes[j // n_cols, j % n_cols].axis("off")

        # Global legend (once)
        handles = [
            plt.Line2D([], [], color="green", alpha=0.6, linewidth=8, label="posterior"),
            plt.Line2D([], [], color="red", linestyle="--", linewidth=2, label="True value"),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=2,
            fontsize=Plotter.legend_fontsize,
            frameon=False,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if savepath is None:
            plt.show()
        else:
            fig.savefig(savepath, dpi=150)
            plt.close(fig)