import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

EPS = 1e-6
ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    """
    A simple Mixture Density Network implementation
    For the multivariate Gaussian cases, assuming the offdiagonal terms of the covariance matrices are 0.0.
    Following the original MDN paper https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf.

    The implementation is borrowed from https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_gaussians: int):
        super(MDN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        self.pi = nn.Sequential(
            nn.Linear(input_dim, num_gaussians),
            nn.Softmax(dim=1)
        )

        self.sigma = nn.Linear(input_dim, output_dim * num_gaussians)
        self.mu = nn.Linear(input_dim, output_dim * num_gaussians)

    def forward(self, x):
        pi = self.pi(x)

        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.output_dim)

        sigma = F.softplus(self.sigma(x)) + EPS  # Adopted from Deep ensemble paper (2nd footnote, 3p)
        sigma = sigma.view(-1, self.num_gaussians, self.output_dim)
        return pi, mu, sigma


def gaussian_probability(mu, sigma, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
    Arguments:
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(pi, mu, sigma, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(mu, sigma, target)
    nll = -torch.log(torch.sum(prob, dim=1) + EPS)
    return torch.mean(nll)


def sample(pi, mu, sigma):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)
