import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np
from sbi_particle_physics.config import (
    AXIS_FONTSIZE,
    TICK_FONTSIZE,
    LEGEND_FONTSIZE,
    ENCODED_DATA_LABELS
)
from pathlib import Path

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

    