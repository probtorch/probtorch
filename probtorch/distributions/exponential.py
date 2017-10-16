"""Implements the Univariate Exponential distribution."""

import torch
from torch.autograd import Variable
import math
from probtorch.distributions.distribution import *
from probtorch.util import broadcast_size, expanded_size
from numbers import Number

__all__ = [
    "Exponential"
]
EPS= 1e-12
class Exponential(Distribution):
    R"""The univariate Exponential distribution.

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
    def __init__(self, lam, size=None):
        
        self._lam = Variable(torch.Tensor([lam])) if isinstance(lam, Number) else lam
        
        super(Exponential, self).__init__(expanded_size(size, 
                                        lam.size()),
                                     lam.data.type(),
                                     GradientType.REPARAMETERIZED)
    @property
    def lam(self):
        return self._lam
    
    @property
    def mean(self):
        return 1.0/self._lam

    @property
    def variance(self):
        return 1.0/self._lam**2

    def sample(self):
        uniform = Variable(torch.clamp(torch.rand(self._size).type(self._type),EPS,1-EPS))
        return - torch.log(uniform)/self._lam
        
    def log_prob(self, value):
        mask = torch.ge(value,0.0).type(torch.DoubleTensor)
        # TODO: apparently, this is slow: https://discuss.pytorch.org/t/bytetensor-to-floattensor-is-slow/3672
        return mask * (torch.log(self._lam) - self._lam * value) + (1-mask) * self.LOG_0
