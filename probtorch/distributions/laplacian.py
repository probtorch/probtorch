import torch
from torch.autograd import Variable
import math
from probtorch.distributions.distribution import *
from probtorch.util import broadcast_size, expanded_size
from numbers import Number

__all__ = [
    "Laplace"
]


class Laplace(Distribution):
    def __init__(self,
                 mu,
                 sigma,
                 name="Laplace"):

        if mu is None:
            mu = 0
        if sigma is None:
            sigma = 0

        self._mu = mu
        self._sigma = sigma

        if broadcast_size(mu, sigma) != (mu + sigma).size(): raise AssertionError("Check the mu and sigma sizes!")
        super(Laplace, self).__init__(broadcast_size(mu, sigma),
                                      mu.data.type(),
                                      GradientType.REPARAMETERIZED)


    ########################################
    ############# Properties ###############
    ########################################
    @property
    def mean(self):
        """Distribution parameter for the location."""
        return self._mu

    @property
    def sigma(self):
        """Distribution parameter for sigma."""
        return self._sigma

    @property
    def size(self):
        """Distribution parameter for sigma."""
        return self._size


    ########################################
    ############# Methods ##################
    ########################################
    def _stddev(self):
        return torch.sqrt(2.) * self.sigma

    def _median(self):
        return self._mean()

    def _mode(self):
        return self._mean()

    # Please refer to https://en.wikipedia.org./wiki/Laplace_distribution.
    def _sample_generate(self, shape=None, seed=None):
        # Given a random variable X drawn from the uniform distribution in the interval
        # (-1/2,1/2], the following random variable has a Laplace distribution with parameters mu and sigma:
        # Y = mu - sigma * sgn(X) * ln (1 - 2*|X|)

        if shape is None: raise AssertionError("You must specify a shape for generating the samples!")
        uniform_generated = Variable(torch.Tensor(*shape).uniform_(-0.5, 0.5))

        return (self.mu - self.sigma * torch.sign(uniform_generated) *
                torch.log(1.0 + -1.0 * 2 * torch.abs.abs(uniform_generated)))

    def _entropy(self):
        # Using broadcasting rule.
        # Entropy= log(2*sigma*e)
        return 1. + torch.log(2.) + torch.log(self.sigma)

    def _log_prob(self, x):
        first_term = -(torch.log(2.) + torch.log(self.sgima))
        second_term = -(torch.abs(x-self.mu) / self.sigma)
        return first_term + second_term

    def _CDF(self, x):
        z = torch.abs(x-self.mu) / self.sigma
        return (0.5 + 0.5 * torch.sign(z) *
                (1. - torch.exp(-torch.abs(z))))
