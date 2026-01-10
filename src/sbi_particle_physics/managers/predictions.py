import torch
from torch import Tensor

class Predictions:
    """
    Responsible for the calculations of the estimators, their uncertainties and averages
    """

    @staticmethod
    def _uncertainty(many_posterior_samples : Tensor) -> Tensor:
        samples = many_posterior_samples.squeeze(-1)
        sample_dim = 0 if samples.ndim == 1 else 1
        q16, q84 = torch.quantile(
            samples,
            torch.tensor([0.16, 0.84], device=samples.device),
            dim=sample_dim
        )
        return 0.5 * (q84 - q16) # half-width of the 68% interval (better if asymetry, non gaussian posterior, conventionnal for sbi)

    @staticmethod
    def calculate_estimator(posterior_samples : Tensor) -> tuple[Tensor, Tensor]:
        samples = posterior_samples.squeeze(-1)
        mean = samples.mean(dim=-1)
        uncertainty = Predictions._uncertainty(posterior_samples)
        return mean, uncertainty
    
    @staticmethod
    def calculate_estimator_summary(posterior_samples : Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        samples = posterior_samples.squeeze(-1)
        mean = samples.mean(dim=-1)
        median = samples.median(dim=-1).values
        std = samples.std(dim=-1, unbiased=False)
        q5, q16, q84, q95 = torch.quantile(
            samples,
            torch.tensor([0.05, 0.16, 0.84, 0.95], device=samples.device),
            dim=-1
        )
        width_68 = q84 - q16
        return mean, median, q5, q16, q84, q95, std, width_68
    
    @staticmethod
    def average_uncertainty(many_posterior_samples : Tensor) -> float:
        uncertainties = Predictions._uncertainty(many_posterior_samples)
        return uncertainties.mean().item()

    @staticmethod
    def log_contraction(prior_samples: Tensor, posterior_samples: Tensor, eps: float = 1e-12) -> Tensor:
        # Compute log(prior/posterior widths) (around 0: posterior not very informative)
        prior_width = Predictions._uncertainty(prior_samples)
        posterior_width = Predictions._uncertainty(posterior_samples)
        return torch.log(prior_width / (posterior_width + eps))

    @staticmethod
    def _entropy_from_samples(samples: Tensor, xmin: Tensor, xmax: Tensor, bins: int = 50, eps: float = 1e-12) -> Tensor:
        samples = samples.squeeze(-1)
        if samples.ndim == 1: samples = samples.unsqueeze(0)
        N, S = samples.shape
        hist = torch.stack([torch.histc(samples[i], bins=bins, min=xmin.item(), max=xmax.item()) for i in range(N)])
        probs = hist / hist.sum(dim=1, keepdim=True)
        probs = probs.clamp(min=eps)
        dx = (xmax - xmin) / bins
        entropy = -torch.sum(probs * torch.log(probs), dim=1) + torch.log(dx)
        return entropy.squeeze()
    
    @staticmethod
    def information_gain(prior_samples: Tensor, posterior_samples: Tensor, bins: int = 50, eps: float = 1e-12) -> Tensor:
        # Information gain measured as entropy reduction: H(prior) - H(posterior)
        xmin = torch.min(prior_samples.min(), posterior_samples.min())
        xmax = torch.max(prior_samples.max(), posterior_samples.max())
        H_prior = Predictions._entropy_from_samples(prior_samples, xmin=xmin, xmax=xmax, bins=bins, eps=eps)
        H_post = Predictions._entropy_from_samples(posterior_samples, xmin=xmin, xmax=xmax, bins=bins, eps=eps)
        return H_prior - H_post
