"""Implements the Univariate Normal distribution."""

import torch
from torch.autograd import Variable
import math
from probtorch.distributions.distribution import *

__all__ = [
    "Normal"
]

class Normal(Distribution):
    "The univariate normal distribution, parameterized by mean and standard deviation."
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std
        # TODO: is there a cleaner way to do broadcast sizes?
        super(Normal, self).__init__((mean + std).size(),
                                     mean.data.type(),
                                     GradientType.REPARAMETERIZED)

    @property
    def mean(self):
        return self._mean
    
    @property
    def mode(self):
        return self._mean
    
    @property
    def std(self):
        return self._std
    
    @property
    def variance(self):
        return self._std**2

    def sample(self):
        eps = Variable(torch.randn(self._size)).type(self._type)
        return self._mean + self._std * eps

    def log_prob(self, value):
        return -0.5 * (torch.log(2 * math.pi * self._std**2)
                       + ((value - self._mean) / self._std)**2)
