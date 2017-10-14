"""Implements the Uniform distribution."""

import torch
from torch.autograd import Variable
import math
from probtorch.distributions.distribution import *
from probtorch.util import broadcast_size, expanded_size

__all__ = [
    "Uniform"
]

class Uniform(Distribution):
    R"""
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
    
    def __init__(self, lower=0.0, upper=1.0, size=None):
        #TODO: needs assert lower<upper
        self._lower = lower
        self._upper = upper
        # TODO: is there a cleaner way to do broadcast sizes?
        super(Uniform, self).__init__(expanded_size(size, 
                                        broadcast_size(lower, upper)),
                                     lower.data.type(),
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
        return (self._upper-self._lower)**2/12.0

    @property
    def std(self):
        return (self._upper-self._lower)/math.sqrt(12.0)

    def sample(self):
        uniform = Variable(torch.rand(self._size)).type(self._type)
        return self._lower + (self._upper-self._lower) * uniform

    def log_prob(self, value):
        mask= Variable((torch.ge(value.data, self._lower.data) \
            & torch.le(value.data, self._upper.data)).type(torch.DoubleTensor))
        return mask * -1.0 * torch.log(self._upper-self._lower) + (1.0 - mask) * self.LOG_0
         
