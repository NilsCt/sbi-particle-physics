import torch
import numpy as np

class Normalizer:
    # Normalize data and use suitable transformations if needed

    def __init__(self, raw_data, raw_parameters):
        formated_data = Normalizer.format_phi(raw_data)
        self.data_mean = formated_data.mean(dim=(0,1)) # shape (point_dim) (on ne mélange pas q^2, cos theta_d,...)
        self.data_std = formated_data.std(dim=(0,1))
        #self.parameters_mean = raw_parameters.mean(dim=0) # shape (parameter_dim)
        #self.parameters_std = raw_parameters.std(dim=0)

        #self.data_mean = 0
        #self.data_std = 1
        self.parameters_mean = 0 # commun de ne pas normaliser les paramètres
        self.parameters_std = 1
 
    def normalize_parameters(self, raw_parameters): # j'ai laissé les fonctions au cas ou plus tard j'en ai besoin
        return (raw_parameters - self.parameters_mean) / self.parameters_std

    def denormalize_parameters(self, parameters):
        return parameters * self.parameters_std + self.parameters_mean
    
    @staticmethod
    def format_phi(raw_x):
        features = raw_x[..., 0:3]
        phi = raw_x[..., 3:4]
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        return torch.cat([features, cos_phi, sin_phi], dim=-1)

    @staticmethod
    def unformat_phi(x_with_encoded_phi):
        features = x_with_encoded_phi[..., 0:3]
        cos_phi = x_with_encoded_phi[..., 3:4]
        sin_phi = x_with_encoded_phi[..., 4:5]
        phi = torch.atan2(sin_phi, cos_phi) # gère le cadran
        return torch.cat([features, phi], dim=-1)

    def normalize_data(self, raw_x):
        formated_x = Normalizer.format_phi(raw_x)
        return (formated_x - self.data_mean) / self.data_std

    def denormalize_data(self, x):
        x_denormalized = x * self.data_std + self.data_mean
        return Normalizer.unformat_phi(x_denormalized)