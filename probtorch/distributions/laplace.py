"""Implements the Univariate Laplace distribution."""

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
    R"""The univariate Laplace distribution.

    .. math::
       f(x \mid \mu, b) =
           \frac{1}{2b}
           \exp \left[ -\frac{|x-\mu|}{b}\right]

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`2b^2`
    ========  ==========================================

    Parameters:
        mu(:obj:`Variable`): Mean.
        b(:obj:`Variable`): Scale (b > 0).
        size(tuple, optional): Sample size. 

    Attributes:
        mean(:obj:`Variable`): Mean (mu).
        mode(:obj:`Variable`): Mode (mu).
        variance(:obj:`Variable`): Variance (equal to 2b**2)
    """
    def __init__(self, mu, b, size=None):
        
        self._mu = Variable(torch.Tensor([mu])) if isinstance(mu, Number) else mu
        self._b = Variable(torch.Tensor([b])) if isinstance(b, Number) else b

        assert(broadcast_size(mu, b) == (mu + b).size())
        super(Laplace, self).__init__(expanded_size(size, 
                                        broadcast_size(mu, b)),
                                     mu.data.type(),
                                     GradientType.REPARAMETERIZED)
    @property
    def mu(self):
        return self._mu
    
    @property
    def b(self):
        return self._b

    @property
    def mean(self):
        return self._mu

    @property
    def mode(self):
        return self._mu
   
    @property
    def variance(self):
        return 2*self._b**2

    def sample(self):
        uniform = Variable(torch.Tensor(self._size).uniform_(-0.5, 0.5)).type(self._type)
        #if U is uniform in (-.5,.5], loc-scale*sgn(U)*ln(1-2*abs(U)) is laplace
        return self._mu - self._b * torch.sign(uniform) * torch.log(1 - 2 * torch.abs(uniform))

    def log_prob(self, value):
        first_term = -(torch.log(2.*self._b))
        second_term = -(torch.abs(value-self._mu) / self._b)
        return first_term + second_term

