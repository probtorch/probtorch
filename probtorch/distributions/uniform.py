"""Implements the Uniform distribution."""
import math
from numbers import Number
import torch
from torch.autograd import Variable
from probtorch.distributions.distribution import *
from probtorch.util import expanded_size

__all__ = [
    "Uniform"
]


class Uniform(Distribution):
    r"""
    Continuous uniform distribution

    .. math::
       f(x \mid lower, upper) = \frac{1}{upper-lower}

    ========  =====================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`(lower + upper) / 2`
    Variance  :math:`(upper - lower)^2 / 12`
    ========  =====================================

    Parameters:
        lower(:obj:`Variable`) : Lower limit (default: 0.0).
        upper(:obj:`Variable`) : Upper limit (upper > lower, default: 1.0).
        size(tuple, optional): Sample size.

    Attributes:
        mean(:obj:`Variable`): Mean.
        std(:obj:`Variable`): Standard deviation.
        variance(:obj:`Variable`): Variance.
    """

    def __init__(self, lower=0.0, upper=1.0):
        # TODO: needs assert lower<upper
        self._lower = lower
        self._upper = upper
        scale = lower - upper
        if isinstance(scale, Number):
            super(Uniform, self).__init__((1,),
                                          'torch.FloatTensor',
                                          GradientType.REPARAMETERIZED)
        else:
            super(Uniform, self).__init__(scale.size(),
                                          scale.data.type(),
                                          GradientType.REPARAMETERIZED)

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def mean(self):
        return 0.5 * (self._lower + self._upper)

    @property
    def variance(self):
        return (self._upper - self._lower)**2 / 12.0

    @property
    def std(self):
        return (self._upper - self._lower) / math.sqrt(12.0)

    def sample(self, *sizes):
        size = expanded_size(sizes, self._size)
        uniform = Variable(torch.Tensor(*size)
                           .type(self._type)
                           .uniform_(0.0, 1.0))
        return self._lower + (self._upper - self._lower) * uniform

    def log_prob(self, value):
        width = self._upper - self._lower
        log = math.log if isinstance(width, Number) else torch.log
        mask = (torch.ge(value, self._lower) *
                torch.le(value, self._upper)).type(self._type)
        log_prob = -mask * log(width)
        log_prob_0 = (1.0 - mask) * self.LOG_0
        return log_prob + log_prob_0
