import math

import torch
import torch.nn as nn


class MatrixMultiplication(nn.Module):
    """
        batch operation supporting matrix multiplication layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int):
        super(MatrixMultiplication, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return torch.einsum('bx, xy -> by', input, self.weight)


class Simulator(nn.Module):

    def __init__(self, state_dim, action_dim, noise=0.01):
        super(Simulator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.A = MatrixMultiplication(state_dim, state_dim)
        self.B = MatrixMultiplication(action_dim, state_dim)
        self.noise = noise

    def forward(self, x, u):
        return self.A(x) + self.B(u) + torch.randn((1, self.state_dim), device=x.device) * self.noise


class SimpleSSM(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(SimpleSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.A = MatrixMultiplication(state_dim, state_dim)
        self.B = MatrixMultiplication(action_dim, state_dim)

    def forward(self, x, u):
        return self.A(x) + self.B(u)
