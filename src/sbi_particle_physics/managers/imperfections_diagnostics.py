import matplotlib.pyplot as plt
import torch
from torch import Tensor
import numpy as np
from pathlib import Path
from scipy.stats import chi2

from sbi_particle_physics.config import (
    AXIS_FONTSIZE,
    TICK_FONTSIZE,
    LEGEND_FONTSIZE,
    PLOTS_DIR,
    C9,
    RED_COLOR,
    BLUE_COLOR,
    GREEN_COLOR
)
from sbi_particle_physics.objects.model import Model


class ImperfectionsDiagnostics:
    OBSERVABLES = [
        ("q2", r"$q^2\ [\mathrm{GeV}^2]$", None),
        ("ctl", r"$\cos\theta_\ell$", (-1, 1)),
        ("ctk", r"$\cos\theta_K$", (-1, 1)),
        ("phi", r"$\phi$", (-np.pi, np.pi)),
        ("mB", r"$m_B\ [\mathrm{GeV}]$", (5.15, 5.40)),
    ]

    @staticmethod
    def _style_ax(ax, xlabel, ylabel="Density"):
        ax.set_xlabel(xlabel, fontsize=AXIS_FONTSIZE+3)
        ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE+3)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE-1)
        ax.grid(alpha=0.3)

    @staticmethod
    def _hist(ax, x, bins, range_, label=None, density=True, color : str = RED_COLOR, lw=2):
        ax.hist(x, bins=bins, range=range_, histtype="step", density=density, linewidth=lw, label=label, color=color)

    @staticmethod
    def _plot_observables(datasets: dict[str, Tensor], bins: int = 30, q2_bin: tuple[float, float] | None = None, density: bool = True):
        n_obs = len(ImperfectionsDiagnostics.OBSERVABLES)
        fig, axes = plt.subplots(1, n_obs, figsize=(4.5 * n_obs, 4), constrained_layout=True)
        colors = [RED_COLOR, BLUE_COLOR, GREEN_COLOR]
        colors = ["blue", "red", "green"]
        if n_obs == 1:
            axes = [axes]

        j = 0
        for label, data in datasets.items():
            data = data.detach().cpu().numpy()
            if q2_bin is not None:
                q2_min, q2_max = q2_bin
                mask = (data[:, 0] > q2_min) & (data[:, 0] < q2_max)
                data = data[mask]

            for i, (_, xlabel, default_range) in enumerate(ImperfectionsDiagnostics.OBSERVABLES):
                values = data[:, i]
                range_ = default_range if default_range is not None else (values.min(), values.max())
                ImperfectionsDiagnostics._hist(axes[i], values, bins=bins, range_=range_, label=label, density=density, color=colors[j % len(colors)], lw=2.5)
            j += 1
        for i, (_, xlabel, _) in enumerate(ImperfectionsDiagnostics.OBSERVABLES):
            ImperfectionsDiagnostics._style_ax(axes[i], xlabel)
            if i!= 4: axes[i].legend(fontsize=LEGEND_FONTSIZE+3)
        if q2_bin is not None:
            fig.suptitle(rf"${q2_bin[0]:.1f} < q^2 < {q2_bin[1]:.1f}\ \mathrm{{GeV}}^2$",fontsize=AXIS_FONTSIZE,)
        plt.savefig(PLOTS_DIR / "viva" / "imperfect.pdf")
        plt.show()

    @staticmethod
    def _plot_observables_2(
        datasets: dict[str, Tensor],
        bins: int = 30,
        q2_bin: tuple[float, float] | None = None,
        density: bool = True,
    ):
        colors = ["blue", "red", "green"]

        for i, (name, xlabel, default_range) in enumerate(ImperfectionsDiagnostics.OBSERVABLES):
            fig, ax = plt.subplots(1, 1, figsize=(5.2, 4), constrained_layout=True)

            j = 0
            for label, data in datasets.items():
                data = data.detach().cpu().numpy()

                if q2_bin is not None:
                    q2_min, q2_max = q2_bin
                    mask = (data[:, 0] > q2_min) & (data[:, 0] < q2_max)
                    data = data[mask]

                values = data[:, i]
                range_ = default_range if default_range is not None else (values.min(), values.max())
                print("rrange ", range_)

                ImperfectionsDiagnostics._hist(
                    ax,
                    values,
                    bins=bins,
                    range_=range_,
                    label=label,
                    density=density,
                    color=colors[j % len(colors)],
                    lw=2.5,
                )
                j += 1

            ImperfectionsDiagnostics._style_ax(ax, xlabel)
            if i!= 4: ax.legend(fontsize=LEGEND_FONTSIZE + 3)

            if q2_bin is not None:
                fig.suptitle(
                    rf"${q2_bin[0]:.1f} < q^2 < {q2_bin[1]:.1f}\ \mathrm{{GeV}}^2$",
                    fontsize=AXIS_FONTSIZE,
                )

            safe_name = name.lower().replace(" ", "_")
            if q2_bin is None:
                save_path = PLOTS_DIR / "viva" / f"imperfect_{safe_name}.pdf"
            else:
                save_path = PLOTS_DIR / "viva" / f"imperfect_{safe_name}_q2_{q2_bin[0]:.1f}_{q2_bin[1]:.1f}.pdf"

            plt.savefig(save_path)
            plt.show()
            plt.close(fig)

    @staticmethod
    def compare_datasets(datasets: dict[str, Tensor], bins: int = 30, q2_bin: tuple[float, float] | None = None, density: bool = True):
        ImperfectionsDiagnostics._plot_observables(datasets, bins=bins, q2_bin=q2_bin, density=density)

    @staticmethod
    def compare_real_vs_simulated(x_real: Tensor, x_sim: Tensor, q2_bin: tuple[float, float] | None = None, bins: int = 30):
        datasets = {"Toy data": x_real, "Simulation": x_sim}
        ImperfectionsDiagnostics._plot_observables_2(datasets, bins=bins, q2_bin=q2_bin)

    @staticmethod
    def compare_real_vs_simulated_vs_accepted(x_real: Tensor, x_sim: Tensor, x_acc: Tensor, q2_bin: tuple[float, float] | None = None, bins: int = 30,):
        datasets = {
            "LHCb data": x_real,
            "No acceptance": x_sim,
            "With acceptance": x_acc,
        }
        ImperfectionsDiagnostics._plot_observables(datasets, bins=bins, q2_bin=q2_bin)


    @staticmethod
    def chi2_test(x_sim: Tensor, x_obs: Tensor, bins: int = 30):
        x_sim = x_sim.detach().cpu().numpy()
        x_obs = x_obs.detach().cpu().numpy()
        for i, (_, label, default_range) in enumerate(ImperfectionsDiagnostics.OBSERVABLES):
            sim = x_sim[:, i]
            obs = x_obs[:, i]
            range_ = default_range if default_range is not None else (min(sim.min(), obs.min()), max(sim.max(), obs.max()))
            h_sim, edges = np.histogram(sim, bins=bins, range=range_)
            h_obs, _ = np.histogram(obs, bins=bins, range=range_)
            h_sim = h_sim / h_sim.sum()
            h_obs = h_obs / h_obs.sum()
            eps = 1e-12
            err_sim = np.sqrt(h_sim + eps) / np.sqrt(len(sim))
            err_obs = np.sqrt(h_obs + eps) / np.sqrt(len(obs))
            err2 = err_sim**2 + err_obs**2
            mask = err2 > 0
            chi2_val = np.sum((h_sim[mask] - h_obs[mask]) ** 2 / err2[mask])
            ndof = np.sum(mask) - 1
            p_value = 1.0 - chi2.cdf(chi2_val, ndof)

            print(f"{label} : chi2/ndof = {chi2_val:.2f}/{ndof}, p = {p_value:.3f}")

    @staticmethod
    def get_datasets(model: Model, n_points: int) -> dict:
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

        return {
            "Ideal": ideal,
            "Acceptance": after_acceptance,
            "Resolution": after_resolution,
            "Background": after_background,
            "Full": imperfect,
        }
