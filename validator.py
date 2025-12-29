import matplotlib.pyplot as plt
from sbi.analysis import plot_summary
import torch
from model import Model
import numpy as np
from sbi.diagnostics import run_sbc
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics import run_tarp, check_tarp
from sbi.analysis.plot import plot_tarp
from sbi.diagnostics.misspecification import calc_misspecification_logprob
from sbi.inference.trainers.marginal import MarginalTrainer
from sbi.diagnostics.misspecification import calc_misspecification_mmd
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.inference import NPE
from sbi.diagnostics.lc2st import LC2ST
from sbi.analysis.plot import pp_plot_lc2st
from sbi.analysis import pairplot

class Validator:
    # test the performance of the model

    """Simulation based calibration (SBC)
    Génère des paramètres theta_i, puis simule des données x_i ~ p(x|theta_i)
    puis prédit les postérieurs p(theta|x_i) et regarde où se situe theta_i dans le posterior
    
    puis histogramme des rangs uniformes, test KS d'uniformité,  xi^2
    teste les biais systématiques, la surconfiance ou sous-confiance des postérieurs
    """
    @staticmethod
    def simulation_based_calibration(model, x, theta,num_posterior_samples):
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

    """Posterior Predictive Checks (PPC)
    Génère des paramètres theta_i, puis simule des données x_i ~ p(x|theta_i)
    puis prédict les postérieurs et génère theta'_j ~ p(theta|x_i) et simule des données x'_j ~ p(x|theta'_j)
    puis compare les distributions de x_i et x'_j

    teste si les données simulées à partir des postérieurs ressemblent aux données observées
    """
    @staticmethod
    def posterior_predictive_checks(model, x_o, n_samples, n_points):
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

    """Expected Coverage Test (ECT)
    Génère des paramètres theta_i, puis simule des données x_i ~ p(x|theta_i)
    puis prédit les postérieurs p(theta|x_i) et vérifie que les theta_i soient dans les intervalles de crédibilité

    teste la surconfiance ou sous-confiance des postérieurs
    """
    @staticmethod
    def expected_coverage_test(model, x, theta, num_posterior_samples):
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

    """TARP Test
    Génère des paramètres theta_i, puis simule des données x_i ~ p(x|theta_i)
    puis prédit les postérieurs p(theta|x_i) et calcule les Expected Credible Percentiles (ECP)
    puis compare la distribution des ECP à la distribution uniforme
    """
    @staticmethod
    def tarp_test(model, x, theta, num_posterior_samples):
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

    """Misspecification Test
    missspecification = true data-generating process differs from the assumed model
    cad pour aucune valeur des paramètres le modèle ne peut reproduire les données observées
    """
    @staticmethod
    def misspecification_test(model, x_train, x_o):
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

    """Misspecification Test with MMD
    mmd = maximum mean discrepancy
    """
    @staticmethod
    def misspecification_test_mmd(model, x_train, x_o):
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

    """LC2ST Test
    Local Classifier 2-Sample Test
    Compare les données observées avec des données simulées à partir des postérieurs prédits
    teste la capacité du modèle à reproduire les données observées
    """
    @staticmethod
    def lc2st_test(model, n_samples, n_points, x_o):
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