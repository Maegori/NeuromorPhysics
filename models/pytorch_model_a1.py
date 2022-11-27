import torch
import torch.nn as nn
import numpy as np

"""
This script is used to convert the weights from the csv files to PyTorch model files.
"""


class Model(nn.Module):
    def __init__(self):
        n = 16
        alpha = 1
        super(Model, self).__init__()
        self.l1 = nn.Linear(n, n * alpha)
        self.input_shape = (n,)

    def forward(self, x):
        y = self.l1(x)
        # y = torch.exp(torch.sum(torch.log(2 * torch.cosh(y)), 1))
        y=torch.sum(y)
        return y

