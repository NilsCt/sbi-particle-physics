import torch
import numpy as np

class Normalizer:
    def __init__(self, raw_data, raw_parameters):
        self.data_mean = raw_data.mean(dim=(0,1)) # shape (point_dim) (on ne mélange pas q^2, phi,...)
        self.data_std = raw_data.std(dim=(0,1))
        #self.parameters_mean = raw_parameters.mean(dim=0) # shape (parameter_dim)
        #self.parameters_std = raw_parameters.std(dim=0)

        #self.data_mean = 0
        #self.data_std = 1
        self.parameters_mean = 0 # todo si je veux normaliser les parametres il faut adapter le prior de sbi qui ne sait pas comment tirer les données
        self.parameters_std = 1

    def normalize_data(self, raw_x):
        return (raw_x - self.data_mean) / self.data_std

    def denormalize_data(self, x):
        return x * self.data_std + self.data_mean
 
    def normalize_parameters(self, raw_parameters):
        return (raw_parameters - self.parameters_mean) / self.parameters_std

    def denormalize_parameters(self, parameters):
        return parameters * self.parameters_std + self.parameters_mean