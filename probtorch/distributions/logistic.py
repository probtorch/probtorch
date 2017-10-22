"""Implements the Logistic distribution."""

import torch
from torch.autograd import Variable
import math
from probtorch.distributions.distribution import *
from probtorch.util import broadcast_size, expanded_size
from numbers import Number

__all__ = [
    "Logistic"
]

class Logistic(Distribution):
    r"""The univariate Logistic distribution.

    .. math::
       f(x \mid \mu, s) =
       \frac{\exp \left(
                    -\frac{x-\mu}{s}
                  \right)}
            {s \left(
                 1 + \exp \left(
                            - \frac{x-\mu}{s}
                          \right)
                \right)^2}

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\frac{s^2\pi^2}{3}`
    ========  ==========================================

    Parameters:
        mu(:obj:`Variable`): Mean.
        s(:obj:`Variable`): Scale (s>0).
        size(tuple, optional): Sample size.

    Attributes:
        mean(:obj:`Variable`): Mean (mu).
        variance(:obj:`Variable`): Variance ((s*pi)**2/3)
    """

    def __init__(self, mu, s, size=None):

        self._mu = Variable(torch.Tensor([mu])) if isinstance(mu, Number) else mu
        self._s = Variable(torch.Tensor([s])) if isinstance(s, Number) else s
        assert(broadcast_size(mu, s) == (mu + s).size())
        super(Logistic, self).__init__(expanded_size(size, broadcast_size(mu, s)),
                                       mu.data.type(),
                                       GradientType.REPARAMETERIZED)

    @property
    def mu(self):
        return self._mu

    @property
    def s(self):
        return self._s

    @property
    def mean(self):
        return self._mu

    @property
    def variance(self):
        return (math.pi * self._s)**2 / 3

    def cdf(self, value):
        return 1.0 / (1 + torch.exp(-(value - self._mu) / self._s))

    def inv_cdf(self, value):
        return self._mu + self._s * (torch.log(value) - torch.log(1.0 - value))

    def sample(self):
        u = Variable(torch.Tensor(self._size)
                     .type(self._type)
                     .uniform_(self.EPS, 1.0 - self.EPS))
        return self._mu + self._s * (torch.log(u) - torch.log(1.0 - u))

    def log_prob(self, value):
        y = (value - self._mu) / self._s
        return -y - 2 * torch.log(1 + torch.exp(-y)) - torch.log(self._s)
