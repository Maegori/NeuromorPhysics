import torch
import torch.nn as nn
import numpy as np
import os, sys

sys.path.append("../models/")
from pytorch_model_a1 import Model

weights_path = "../models/weights/"
alphas = [1, 2, 4, 6, 8, 10]
n = 16  # number of spins

for alpha in alphas:

    with open(weights_path + f"alpha={alpha}/b.csv", 'r') as f:
        biases = np.loadtxt(f, delimiter=',', dtype=np.float64)
    
    with open(weights_path + f"alpha={alpha}/W.csv", 'r') as f:
        weights = np.loadtxt(f, delimiter=',', dtype=np.float64).transpose()

    rbm = Model()
    sd = rbm.state_dict()

    sd['l1.bias'] = torch.from_numpy(biases)
    sd['l1.weight'] = torch.from_numpy(weights)

    torch.save(sd, weights_path + f"alpha={alpha}/rbm.pkl")

    print(f"alpha={alpha} weights saved")


