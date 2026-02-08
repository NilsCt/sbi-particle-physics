import numpy as np
import torch
from torch import Tensor
from pathlib import Path
import uproot
import awkward as ak

class RealData:
    """
    Responsible to load and format real LHCb data from root files
    """

    def load_a_file(file : Path):
        """
        Load real LHCb data from a root file
        """
        file = uproot.open(file)
        print(file.keys())
        tree = file["DecayTree"]
        print(tree.keys())

        branches = ["B_PT", "B_M", "B_ETA", "mu_plus_PT", "mu_minus_PT",] # charger certaines données
        data = tree.arrays(branches, library="ak") 
        print(data["B_PT"])
        B_pt = ak.to_numpy(data["B_PT"])
        B_mass = ak.to_numpy(data["B_M"])
        # éviter data = tree.arrays(library="ak") de charger tous l'arbre qui peut etre très lourd