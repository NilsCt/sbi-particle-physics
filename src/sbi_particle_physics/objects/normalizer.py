import torch
from torch import Tensor
import numpy as np
from typing import Self

class Normalizer:
    """
    Normalize and format data and parameters to make the neural network more efficient

    For now the parameters are unchanged
    The data is normalized and formated such that
    (q^2, \\cos \\theta_d, \\cos \\theta_l, \\phi) -> (q^2, \\cos \\theta_d, \\cos \\theta_l, \\cos \\phi, \\sin \\phi)
    to encode the periodicity (phi near -pi should be similar to phi near pi)
    """

    def __init__(self, data_mean : Tensor, data_std : Tensor):
        #formated_data = Normalizer.format_phi(raw_data)
        #self.data_mean = formated_data.mean(dim=(0,1)) # shape (point_dim) (average along q^2, cos theta_d,...)
        #self.data_std = formated_data.std(dim=(0,1))

        self.data_mean = data_mean
        self.data_std = data_std
        self.parameters_mean = None # I don't normalize parameters, but here for potential future use
        self.parameters_std = None
 
    
    @staticmethod
    def calculate_stats(raw_data : Tensor) -> tuple[Tensor, Tensor]:
        formated_data = Normalizer._format_phi(raw_data)
        return formated_data.mean(dim=(0,1)), formated_data.std(dim=(0,1))
    
    @staticmethod
    def create_normalizer(raw_data : Tensor) -> Self:
        data_mean, data_std = Normalizer.calculate_stats(raw_data)
        return Normalizer(data_mean, data_std)
    
    @staticmethod
    def _format_phi(raw_x : Tensor) -> Tensor:
        features = raw_x[..., 0:3]
        phi = raw_x[..., 3:4]
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        return torch.cat([features, cos_phi, sin_phi], dim=-1)

    @staticmethod
    def _unformat_phi(x_with_encoded_phi : Tensor) -> Tensor:
        features = x_with_encoded_phi[..., 0:3]
        cos_phi = x_with_encoded_phi[..., 3:4]
        sin_phi = x_with_encoded_phi[..., 4:5]
        phi = torch.atan2(sin_phi, cos_phi) # knows how to recognize angles in parts I II III IV
        return torch.cat([features, phi], dim=-1)

    def normalize_data(self, raw_x : Tensor) -> Tensor:
        formated_x = Normalizer._format_phi(raw_x)
        return (formated_x - self.data_mean) / self.data_std

    def denormalize_data(self, x: Tensor) -> Tensor:
        x_denormalized = x * self.data_std + self.data_mean
        return Normalizer._unformat_phi(x_denormalized)
    
    def normalize_parameters(self, raw_parameters : Tensor) -> Tensor: # I kept these functions in case I need to format the parameters later
        return raw_parameters

    def denormalize_parameters(self, parameters : Tensor) -> Tensor:
        return parameters