import torch
import numpy as np
import torch.nn as nn

from src.nn.MLP import MLP


class GaussianLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GaussianLayer, self).__init__()
        self.z_mu = torch.nn.Linear(input_dim, output_dim)
        self.z_sigma = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mu = self.z_mu(x)
        std = self.z_sigma(x)
        std = torch.exp(std)
        return mu, std


class GaussianNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GaussianNetwork, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim)
        self.gaussian_layer = GaussianLayer(hidden_dim, output_dim)

    def forward(self, x):
        x = self.mlp(x)
        mu, std = self.gaussian_layer.forward(x)
        return mu, std


def gaussianLL(mu, std, y):
    log_exp_terms = torch.pow(y - mu, 2) / (2 * torch.pow(std, 2))
    log_scaler_terms = -0.5 * (torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(np.pi)) + 2 * torch.log(std))
    LL = log_scaler_terms - log_exp_terms
    return LL
