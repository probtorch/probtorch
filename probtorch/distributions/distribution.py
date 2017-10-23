"""Base classes for probability distributions."""

import torch

__all__ = [
    "GradientType",
    "Distribution"
]


class GradientType(object):
    """Enumerates gradient implementations for distributions."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<GradientType: %s>" % self


GradientType.REPARAMETERIZED = GradientType("reparameterized")
GradientType.REINFORCE = GradientType("reinforce")
GradientType.NONE = GradientType("none")


class Distribution(object):
    LOG_0 = -12.0
    EPS = 1e-12

    def __init__(self, event_size, data_type, gradient_type):
        self._size = event_size
        self._type = data_type
        self._gradient_type = gradient_type

    @property
    def type(self):
        return self._type

    @property
    def gradient_type(self):
        return self._gradient_type

    @property
    def event_size(self):
        return self._size

    def sample(self, *sizes):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def prob(self, value):
        return torch.exp(self.log_prob(value))

    def log_cdf(self, value):
        return torch.log(self.cdf(value))

    def cdf(self, value):
        raise NotImplementedError

    def inv_cdf(self, value):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def covariance(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError
