"""Implements the Univariate Kumaraswamy distribution."""

import torch
from torch.autograd import Variable
import math
from probtorch.distributions.distribution import *
from probtorch.util import broadcast_size, expanded_size
from numbers import Number

__all__ = [
    "Kumaraswamy"
]


class Kumaraswamy(Distribution):
    r"""The univariate Kumaraswamy distribution.

    .. math::
       f(x \mid a, b) = a b x^{a-1}(1-x^a)^{b-1}

    ========  ==========================================
    Support   :math:`x \in \left[0,1\right]`
    Mean      :math:`b\Gamma \left(1+\frac{1}{a}\right)\Gamma \left(b\right)/\Gamma \left(1+\frac{1}{a}+b\right)`
    Variance  No closed-form
    ========  ==========================================

    Parameters:
        a(:obj:`Variable`): Shape parameter.
        b(:obj:`Variable`): Shape parameter.

    Attributes:
        mean(:obj:`Variable`): Mean.
    """

    def __init__(self, a, b):
        self._a = a
        self._b = b
        s = a + b
        if isinstance(s, Number):
            super(Kumaraswamy, self).__init__((1,),
                                              'torch.FloatTensor',
                                              GradientType.REPARAMETERIZED)
        else:
            super(Kumaraswamy, self).__init__(s.size(),
                                              s.data.type(),
                                              GradientType.REPARAMETERIZED)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def mean(self):
        log = math.log if isinstance(self._b, Number) else torch.log
        lgamma = math.lgamma if isinstance(self._b, Number) else torch.lgamma
        exp = math.exp if isinstance(self._b, Number) else torch.exp
        return exp(log(self._b) +
                   lgamma(1.0 + 1.0 / self._a) +
                   lgamma(self._b) -
                   lgamma(1 + 1.0 / self._a + self._b))

    def cdf(self, value):
        return 1.0 - (1.0 - torch.clamp(value, 0.0, 1.0)**self._a)**self._b

    def inv_cdf(self, value):
        return (1 - (1 - value)**(1.0 / self._b))**(1.0 / self._a)

    def sample(self, *sizes):
        size = expanded_size(sizes, self._size)
        uniform = Variable(torch.Tensor(*size)
                           .type(self._type)
                           .uniform_(0.0, 1.0))
        return self.inv_cdf(uniform)

    def log_prob(self, value):
        log = math.log if isinstance(self._b, Number) else torch.log
        valid_value = torch.clamp(value, self.EPS, 1.0 - self.EPS)
        mask = (torch.gt(value, 0.0) * torch.lt(value, 1.0)).type(self._type)
        log_prob = mask * (log(self._a) + log(self._b) +
                           (self._a - 1) * torch.log(valid_value) +
                           (self._b - 1) *
                           torch.log1p(-valid_value ** self._a))
        log_prob_0 = (1.0 - mask) * self.LOG_0
        return log_prob + log_prob_0
