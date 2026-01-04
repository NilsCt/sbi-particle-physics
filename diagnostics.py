import matplotlib.pyplot as plt
import torch
from torch import Tensor
from model import Model
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

    """
    Posterior Predictive Checks (PPC)
    Generates parameters θ_i, simulates data x_i ~ p(x | θ_i),
    infers posteriors p(θ | x_i), samples θ'_j ~ p(θ | x_i),
    and simulates posterior predictive data x'_j ~ p(x | θ'_j).

    Compares observed data x_i with posterior predictive data x'_j
    to assess whether the inferred posteriors can reproduce the observed data.
    """
    @staticmethod
    def posterior_predictive_checks(model : Model, x_o : Tensor, n_samples : int, n_points : int):
        D = 5  # simulator output was 5-dimensional
        x_pp, theta_pp = model.simulate_data_from_predicted_posterior(x_o, n_samples, n_points)
        _ = pairplot(
            samples=x_pp,
            points=x_o[0],
            limits=torch.tensor([[-2.0, 5.0]] * 5),
            figsize=(8, 8),
            upper="scatter",
            upper_kwargs=dict(marker=".", s=5),
            fig_kwargs=dict(
                points_offdiag=dict(marker="+", markersize=20),
                points_colors="red",
            ),
            labels=[rf"$x_{d}$" for d in range(D)],
        )

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

    """
    Misspecification Test
    Model misspecification occurs when the true data-generating process
    differs from the assumed model, such that no parameter value
    can generate data consistent with the observations.

    Misspecification leads to systematically biased or misleading posteriors.
    """
    @staticmethod
    def misspecification_test(model : Model, x_train : Tensor, x_o : Tensor):
        # x_train: baseline samples from the model (e.g., 1000-5000 samples)
        # x_o: single observation to test (shape: (n_points, features))
        trainer = MarginalTrainer(density_estimator='NSF')
        trainer.append_samples(x_train)
        est = trainer.train()

        p_value, reject_H0 = calc_misspecification_logprob(x_train, x_o, est)
        print(f"p-value: {p_value:.4f}, Reject H0 (misspecified): {reject_H0}")

        plt.figure(figsize=(6, 4), dpi=80)
        plt.hist(est.log_prob(x_train).detach().numpy(), bins=50, alpha=0.5, label=r'log p($x_{train}$)')
        plt.axvline(est.log_prob(x_o).detach().item(), color="red", label=r'$\log p(x_{o_{mis}})$)')
        plt.ylabel('Count')
        plt.xlabel(r'$\log p(x)$')
        plt.legend()
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
        p_val, (mmds_baseline, mmd) = calc_misspecification_mmd(
            inference=model.neural_network, x_obs=x_o, x=x_train, mode="embedding"
        )

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
        plt.show()

        pp_plot_lc2st(
            probs=[probs_data],
            probs_null=probs_null,
            conf_alpha=0.05,
            labels=["Classifier probabilities \n on observed data"],
            colors=["red"],
        )