import matplotlib.pyplot as plt
import torch
from torch import Tensor
from sbi_particle_physics.objects.model import Model
from sbi_particle_physics.managers.plotter import Plotter
import numpy as np
from sbi_particle_physics.config import AXIS_FONTSIZE, LEGEND_FONTSIZE, TICK_FONTSIZE, ENCODED_DATA_LABELS


class DataDiagnostics:
    """
    Test, quantify and visualize the relevance of the data generated
    """

    @staticmethod
    def _unicity(x: torch.Tensor, tol: float = 1e-3) -> dict:
        x_quant = torch.round(x / tol) * tol
        unique = torch.unique(x_quant, dim=0)
        return unique.shape[0] / x.shape[0]

    @staticmethod
    def data_unicity(data : Tensor, parameters : Tensor):
        """
        Taux d'unicité
        Compte le nombre de points différents
        (0.95< would be ok, 0.9-0.8 suspect)
        """
        parameters_uni = DataDiagnostics._unicity(parameters)
        data_unis = []
        for i in range(data.shape[0]):
            data_unis.append(DataDiagnostics._unicity(data[i]))
        data_unis = torch.as_tensor(data_unis, device=data.device)
        print(f"Parameters unicity rate {parameters_uni} (normal to be low if many parameters)")
        print(f"Data average unicity rate {data_unis.mean()} (>0.95 needed), data min unicity rate {data_unis.min()}")


    @staticmethod
    def _autocorr_1d(x: Tensor, max_lag: int = 100) -> Tensor:
        x = x - x.mean()
        var = x.var(unbiased=False)
        if var == 0: return torch.zeros(max_lag, device=x.device)
        ac = [torch.tensor(1.0, device=x.device)]
        for k in range(1, max_lag):
            ac_k = (x[:-k] * x[k:]).mean() / var
            ac.append(ac_k)
        return torch.stack(ac)

    @staticmethod
    def _autocorr(x: Tensor, max_lag: int = 100) -> Tensor:
        return torch.stack([DataDiagnostics._autocorr_1d(x[:, i], max_lag) for i in range(x.shape[1])])

    @staticmethod
    def _dataset_autocorr(x: Tensor, max_lag: int = 100) -> dict:
        acs = torch.stack([DataDiagnostics._autocorr(x[i], max_lag) for i in range(x.shape[0])])
        return acs.mean(dim=0), acs.std(dim=0)

    @staticmethod
    def _integrated_autocorr_time(ac, cutoff=0.05):
        mask = ac[1:] > cutoff
        return 1 + 2 * ac[1:][mask].sum()

    @staticmethod
    def _ess_from_autocorr(ac, n_points, cutoff=0.05):
        tau = DataDiagnostics._integrated_autocorr_time(ac, cutoff)
        return n_points / tau
    
    @staticmethod
    def _decorrelation_lag(ac, threshold=0.1):
        idx = torch.where(ac < threshold)[0]
        return idx[0].item() if len(idx) > 0 else None

    @staticmethod
    def data_autocorrelation(data : Tensor, lag_zoom : int = 20):
        """
        Autocorrélation, Effective Sample Size (ESS)
        Tests si des points consécutifs sont corrélés
        """
        mean_ac, std_ac = DataDiagnostics._dataset_autocorr(data)
        d, max_lag = mean_ac.shape
        for i in range(d):
            plt.figure(figsize=(6,4))
            plt.plot(mean_ac[i,0:lag_zoom], label="mean")
            plt.fill_between(
                range(max_lag)[0:lag_zoom],
                mean_ac[i,0:lag_zoom] - std_ac[i,0:lag_zoom],
                mean_ac[i,0:lag_zoom] + std_ac[i,0:lag_zoom],
                alpha=0.3
            )
            plt.axhline(0, color="black", lw=0.5)
            plt.title(ENCODED_DATA_LABELS[i], fontsize=AXIS_FONTSIZE)
            plt.xlabel("Lag", fontsize=AXIS_FONTSIZE)
            plt.ylabel("$\\rho$", fontsize=AXIS_FONTSIZE)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=LEGEND_FONTSIZE)
            plt.show()

        ess_per_obs = [
            DataDiagnostics._ess_from_autocorr(mean_ac[i], n_points=data.shape[1]).item()
            for i in range(mean_ac.shape[0])
        ]
        ess_min = min(ess_per_obs)
        decorrelation_lag = [
            DataDiagnostics._decorrelation_lag(mean_ac[i]) for i in range(mean_ac.shape[0])
        ]
        print("ESS min", ess_min)
        print("ESS's", ess_per_obs)
        print("Decorrelation lag", decorrelation_lag)