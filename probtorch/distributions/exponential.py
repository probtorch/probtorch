"""Implements the Univariate Exponential distribution."""
import math
import torch
from torch.autograd import Variable
from probtorch.distributions.distribution import *
from probtorch.util import expanded_size
from numbers import Number

__all__ = [
    "Exponential"
]


class Exponential(Distribution):
    r"""The univariate Exponential distribution.

    .. math::
       f(x \mid \lambda) = \lambda \exp \left(-\lambda x\right)

    ========  ==========================================
    Support   :math:`x \in \left[ 0,\infty\right)`
    Mean      :math:`\frac{1}{\lambda}`
    Variance  :math:`\frac{1}{\lambda^2}`
    ========  ==========================================


    Parameters:
        lam(:obj:`Variable`): Rate.
        size(tuple, optional): Sample size.

    Attributes:
        mean(:obj:`Variable`): Mean (1/lam).
        variance(:obj:`Variable`): Variance (equal to 1/lam**2)
    """

    def __init__(self, lam):
        self._lam = lam
        if isinstance(lam, Number):
            super(Exponential, self).__init__((1,),
                                              'torch.FloatTensor',
                                              GradientType.REPARAMETERIZED)
        else:
            super(Exponential, self).__init__(lam.size(),
                                              lam.data.type(),
                                              GradientType.REPARAMETERIZED)

    @property
    def lam(self):
        return self._lam

    @property
    def mean(self):
        return 1.0 / self._lam

    @property
    def variance(self):
        return 1.0 / self._lam**2

    def sample(self, *sizes):
        size = expanded_size(sizes, self._size)
        uniform = Variable(torch.Tensor(*size)
                           .type(self._type)
                           .uniform_(self.EPS, 1.0 - self.EPS))
        return - torch.log(uniform) / self._lam

    def log_prob(self, value):
        log = math.log if isinstance(self._lam, Number) else torch.log
        # TODO: apparently, this is slow:
        # https://discuss.pytorch.org/t/bytetensor-to-floattensor-is-slow/3672
        mask = torch.ge(value, 0.0).type(torch.DoubleTensor)
        log_prob = mask * (log(self._lam) - self._lam * value)
        log_prob_0 = (1 - mask) * self.LOG_0
        return log_prob + log_prob_0
