import matplotlib.pyplot as plt
import torch
from torch import Tensor
import numpy as np
from sbi_particle_physics.config import (
    AXIS_FONTSIZE,
    TICK_FONTSIZE,
    LEGEND_FONTSIZE,
    ENCODED_DATA_LABELS,
    C9
)
from pathlib import Path
from sbi_particle_physics.objects.model import Model
from scipy.stats import chi2

class ImperfectionsDiagnostics:
    """
    Test, quantify and visualize the modeling of the data imperfections
    """
    
    @staticmethod
    def _style_ax(ax, xlabel, ylabel="Candidates / bin"):
        ax.set_xlabel(xlabel, fontsize=AXIS_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.3)

    @staticmethod
    def _hist_step(ax, x: np.ndarray, bins: int, range_: tuple, label: str | None = None, color: str = "black", density: bool = False, lw: float = 1.5):
        ax.hist(x, bins=bins, range=range_, histtype="step", linewidth=lw, color=color, label=label, density=density)

    @staticmethod
    def angular_distributions(data: Tensor, q2_bin: tuple[float, float], bins: int = 25):
        data = data.detach().cpu().numpy()
        q2, ctl, ctk, phi = data.T
        q2_min, q2_max = q2_bin
        mask = (q2 > q2_min) & (q2 < q2_max)
        ctl = ctl[mask]
        ctk = ctk[mask]
        phi = phi[mask]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        ImperfectionsDiagnostics._hist_step(axes[0], ctl, bins=bins, range_=(-1, 1))
        ImperfectionsDiagnostics._style_ax(axes[0], xlabel=r"$\cos\theta_\ell$")
        ImperfectionsDiagnostics._hist_step(axes[1], ctk, bins=bins, range_=(-1, 1))
        ImperfectionsDiagnostics._style_ax(axes[1], xlabel=r"$\cos\theta_K$")
        ImperfectionsDiagnostics._hist_step(axes[2], phi, bins=bins, range_=(-np.pi, np.pi))
        ImperfectionsDiagnostics._style_ax(axes[2], xlabel=r"$\phi$")
        fig.suptitle(rf"${q2_min:.1f} < q^2 < {q2_max:.1f}\ \mathrm{{GeV}}^2$", fontsize=AXIS_FONTSIZE)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.show()

    @staticmethod
    def q2_distribution(data: Tensor,bins: int = 30, range_: tuple | None = None):
        q2 = data[:, 0].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        ImperfectionsDiagnostics._hist_step(ax, q2, bins=bins, range_=range_ if range_ is not None else (q2.min(), q2.max()))
        ImperfectionsDiagnostics._style_ax(ax, xlabel=r"$q^2\ [\mathrm{GeV}^2]$")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.show()


    @staticmethod
    def get_datasets(model : Model, n_points : int) -> dict:
        model.simulator.imperfections.use_acceptance = False
        model.simulator.imperfections.use_resolution = False
        model.simulator.imperfections.use_background = False
        data = model.simulate_data_with_parameters(model.to_tensor([[C9]]), n_points=n_points).squeeze(0)
        raw_data = model.normalizer.denormalize_data(data)
        ideal = raw_data.clone()
        after_acceptance = model.simulator.imperfections._apply_acceptance(raw_data.clone())
        after_resolution = model.simulator.imperfections._apply_resolution(raw_data.clone())
        after_background = model.simulator.imperfections._apply_background(raw_data.clone())[:n_points]
        model.simulator.imperfections.use_acceptance = True
        model.simulator.imperfections.use_resolution = True
        model.simulator.imperfections.use_background = True
        imperfect = model.simulator.imperfections.apply(raw_data.clone())[:n_points]
        datasets = {
                "Ideal": ideal,
                "Acceptance": after_acceptance,
                "Resolution": after_resolution,
                "Background": after_background,
                "Full": imperfect,
        }
        return datasets



    @staticmethod
    def angular_distributions_compare(datasets: dict[str, Tensor], q2_bin: tuple[float, float], bins: int = 25, colors: dict[str, str] | None = None, density : bool = False):
        q2_min, q2_max = q2_bin
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        if colors is None:
            colors = {k: None for k in datasets}
        for label, data in datasets.items():
            data = data.detach().cpu().numpy()
            q2, ctl, ctk, phi = data.T
            mask = (q2 > q2_min) & (q2 < q2_max)
            ImperfectionsDiagnostics._hist_step(axes[0], ctl[mask], bins=bins, range_=(-1, 1), label=label, color=colors.get(label), density=density)
            ImperfectionsDiagnostics._hist_step(axes[1], ctk[mask], bins=bins, range_=(-1, 1), label=label, color=colors.get(label), density=density)
            ImperfectionsDiagnostics._hist_step(axes[2], phi[mask], bins=bins, range_=(-np.pi, np.pi), label=label, color=colors.get(label), density=density)
        ImperfectionsDiagnostics._style_ax(axes[0], xlabel=r"$\cos\theta_\ell$")
        ImperfectionsDiagnostics._style_ax(axes[1], xlabel=r"$\cos\theta_K$")
        ImperfectionsDiagnostics._style_ax(axes[2], xlabel=r"$\phi$")
        fig.suptitle(rf"${q2_min:.1f} < q^2 < {q2_max:.1f}\ \mathrm{{GeV}}^2$", fontsize=AXIS_FONTSIZE)
        axes[2].legend(fontsize=LEGEND_FONTSIZE)
        plt.show()

    @staticmethod
    def q2_distribution_compare(datasets: dict[str, Tensor], bins: int = 40, range_: tuple | None = None, colors: dict[str, str] | None = None, density : bool = False):
        fig, ax = plt.subplots(figsize=(6, 4))
        if colors is None:
            colors = {k: None for k in datasets}
        for label, data in datasets.items():
            q2 = data[:, 0].detach().cpu().numpy()
            ImperfectionsDiagnostics._hist_step(ax, q2, bins=bins, range_=range_ if range_ is not None else (q2.min(), q2.max()), label=label,color=colors.get(label), density=density)
        ImperfectionsDiagnostics._style_ax(ax, xlabel=r"$q^2\ [\mathrm{GeV}^2]$")
        ax.legend(fontsize=LEGEND_FONTSIZE)
        plt.show()



    @staticmethod
    def chi2_test_1d(x_sim: Tensor, x_obs: Tensor, bins: int, range_: tuple, label: str):
        """
        Chi^2 goodness-of-fit test between simulated and observed data
        (shape-only, normalized histograms)
        """
        x_sim = x_sim.detach().cpu().numpy()
        x_obs = x_obs.detach().cpu().numpy()
        h_sim, edges = np.histogram(x_sim, bins=bins, range=range_)
        h_obs, _     = np.histogram(x_obs, bins=bins, range=range_)
        h_sim = h_sim / h_sim.sum()
        h_obs = h_obs / h_obs.sum()

        eps = 1e-12 # uncertainties (Poisson propagated through normalization)
        err_sim = np.sqrt(h_sim + eps) / np.sqrt(len(x_sim))
        err_obs = np.sqrt(h_obs + eps) / np.sqrt(len(x_obs))
        err2 = err_sim**2 + err_obs**2
        mask = err2 > 0
        chi2_val = np.sum((h_sim[mask] - h_obs[mask])**2 / err2[mask])
        ndof = np.sum(mask) - 1
        p_value = 1.0 - chi2.cdf(chi2_val, ndof)

        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.figure(figsize=(6, 4))
        plt.errorbar(centers, h_obs, yerr=err_obs, fmt="o", label="Observed", capsize=2)
        plt.step(centers, h_sim, where="mid", label="Simulated", linewidth=1.8)
        plt.xlabel(label)
        plt.ylabel("Normalized counts")
        plt.title(rf"$\chi^2/{ndof} = {chi2_val:.1f}/{ndof},\ p={p_value:.3f}$")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        plt.close()
        print(f"Chi2 for {label}: chi2_val {chi2_val}, ndof {ndof}, p_value {p_value}")
    
    @staticmethod
    def chi2_test(x_sim: Tensor, x_obs: Tensor, bins: int, ranges: tuple):
        ImperfectionsDiagnostics.chi2_test_1d(x_sim[:,0], x_obs[:,0], bins, ranges[0], label=r"$q^2$")
        ImperfectionsDiagnostics.chi2_test_1d(x_sim[:,1], x_obs[:,1], bins, ranges[1], label=r"$\cos\theta_\ell$")
        ImperfectionsDiagnostics.chi2_test_1d(x_sim[:,2], x_obs[:,2], bins, ranges[2], label=r"$\cos\theta_K$")
        ImperfectionsDiagnostics.chi2_test_1d(x_sim[:,3], x_obs[:,3], bins, ranges[3], label=r"$\phi$")


    @staticmethod
    def _rbf_kernel(x: Tensor, y: Tensor, sigma: float) -> Tensor:
        x2 = (x**2).sum(dim=1, keepdim=True)
        y2 = (y**2).sum(dim=1, keepdim=True).T
        xy = x @ y.T
        dist2 = x2 + y2 - 2 * xy
        return torch.exp(-dist2 / (2 * sigma**2))


    @staticmethod
    def _mmd2(x: Tensor, y: Tensor, sigma: float) -> Tensor:
        Kxx = ImperfectionsDiagnostics._rbf_kernel(x, x, sigma)
        Kyy = ImperfectionsDiagnostics._rbf_kernel(y, y, sigma)
        Kxy = ImperfectionsDiagnostics._rbf_kernel(x, y, sigma)
        n = x.shape[0]
        m = y.shape[0]
        return (
            (Kxx.sum() - torch.diagonal(Kxx).sum()) / (n * (n - 1)) +
            (Kyy.sum() - torch.diagonal(Kyy).sum()) / (m * (m - 1)) -
            2 * Kxy.mean()
        )

    
    @staticmethod
    def _median_heuristic(z: Tensor) -> float:
        with torch.no_grad():
            dists = torch.cdist(z, z)
            median = torch.median(dists[dists > 0])
        return median.item()
    

    @staticmethod
    def mmd_test(x_sim: Tensor, x_obs: Tensor):
        x_sim = x_sim.detach().cpu()
        x_obs = x_obs.detach().cpu()
        N_sim = x_sim.shape[0]
        z = torch.cat([x_sim.reshape(-1, x_sim.shape[-1]), x_obs], dim=0) # Kernel width from pooled data (important!)
        sigma = ImperfectionsDiagnostics._median_heuristic(z)
        mmds = []
        for i in range(N_sim):
            mmd_i = ImperfectionsDiagnostics._mmd2(x_sim[i], x_obs, sigma)
            mmds.append(mmd_i)
        mmds = torch.stack(mmds)
        mmd_mean = mmds.mean().item() # summary statistics
        mmd_median = mmds.median().item()

        plt.figure(figsize=(6, 4))
        plt.hist(mmds.numpy(), bins=40, alpha=0.7, label="MMD(sim, obs)")
        plt.axvline(mmd_median, color="k", lw=2, label="Median")
        plt.xlabel("MMD")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"MMD test mmds {mmds}, mmd_mean {mmd_mean}, mmd_median {mmd_median}")


    @staticmethod
    def compare_real_vs_simulated(x_real: Tensor, x_sim: Tensor, q2_bin: tuple[float, float] | None = None, bins: int = 30, labels: tuple[str, str] = ("LHCb data", "Simulation")):
        """
        Compare real LHCb data to simulated data for fixed C9.
        Plots distributions of q2, cos(theta_l), cos(theta_K), phi.
        """
        x_real = x_real.detach().cpu().numpy()
        x_sim = x_sim.detach().cpu().numpy()
        q2_r, ctl_r, ctk_r, phi_r = x_real.T
        q2_s, ctl_s, ctk_s, phi_s = x_sim.T
        if q2_bin is not None:
            q2_min, q2_max = q2_bin
            mask_r = (q2_r > q2_min) & (q2_r < q2_max)
            mask_s = (q2_s > q2_min) & (q2_s < q2_max)
            q2_r, ctl_r, ctk_r, phi_r = q2_r[mask_r], ctl_r[mask_r], ctk_r[mask_r], phi_r[mask_r]
            q2_s, ctl_s, ctk_s, phi_s = q2_s[mask_s], ctl_s[mask_s], ctk_s[mask_s], phi_s[mask_s]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
        axes[0].hist(q2_r, bins=bins, histtype="step", density=True, lw=2, label=labels[0])
        axes[0].hist(q2_s, bins=bins, histtype="step", density=True, lw=2, label=labels[1])
        axes[0].set_xlabel(r"$q^2\ [\mathrm{GeV}^2]$")
        axes[0].set_ylabel("Density")
        axes[1].hist(ctl_r, bins=bins, range=(-1, 1), histtype="step", density=True, lw=2)
        axes[1].hist(ctl_s, bins=bins, range=(-1, 1), histtype="step", density=True, lw=2)
        axes[1].set_xlabel(r"$\cos\theta_\ell$")
        axes[2].hist(ctk_r, bins=bins, range=(-1, 1), histtype="step", density=True, lw=2)
        axes[2].hist(ctk_s, bins=bins, range=(-1, 1), histtype="step", density=True, lw=2)
        axes[2].set_xlabel(r"$\cos\theta_K$")
        axes[3].hist(phi_r, bins=bins, range=(-np.pi, np.pi), histtype="step", density=True, lw=2)
        axes[3].hist(phi_s, bins=bins, range=(-np.pi, np.pi), histtype="step", density=True, lw=2)
        axes[3].set_xlabel(r"$\phi$")
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=12)
        if q2_bin is not None:
            fig.suptitle(rf"${q2_bin[0]:.1f} < q^2 < {q2_bin[1]:.1f}\ \mathrm{{GeV}}^2$", fontsize=14)
        plt.show()
