"""Implements the Univariate Laplace distribution."""

import torch
from torch.autograd import Variable
from probtorch.distributions.distribution import *
from probtorch.util import expanded_size
from numbers import Number

__all__ = [
    "Laplace"
]


class Laplace(Distribution):
    r"""The univariate Laplace distribution.

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

    def __init__(self, mu, b):

        self._mu = mu
        self._b = b
        mu0 = mu / b
        if isinstance(mu0, Number):
            super(Laplace, self).__init__((1,),
                                          'torch.FloatTensor',
                                          GradientType.REPARAMETERIZED)
        else:
            super(Laplace, self).__init__(mu0.size(),
                                          mu0.data.type(),
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
        return 2.0 * self._b**2

    def sample(self, *sizes):
        size = expanded_size(sizes, self._size)
        uniform = Variable(torch.Tensor(*size)
                           .type(self._type)
                           .uniform_(-0.5, 0.5))
        # if U is uniform in (-.5,.5], then
        # mu - b * sign(U) * log(1 - 2 * abs(U)) is Laplace
        return self._mu - (self._b *
                           torch.sign(uniform) *
                           torch.log(1.0 - 2.0 * torch.abs(uniform)))

    def log_prob(self, value):
        log = math.log if isinstance(self._b, Number) else torch.log
        log_normalizer = log(2.0 * self._b)
        log_weight = -torch.abs(value - self._mu) / self._b
        return log_weight - log_normalizer


    def entropy(self):
        """
        Using broadcasting rule it return the Shannon entropy (nats).

        :return:
            Entropy= log(2*sigma*e)
        """
        return 1. + torch.log(2.) + torch.log(self._b)

    def cdf(self, x):
        """
        Cumulative distribution function.
        Given the variable X, the CDF is:
        CDF(x): = P[X <= x]

        :param
                x: The variable

        :return:
                CDF(x)
        """
        #

        z = (x - self.mu) / self._b
        return (0.5 + 0.5 * torch.sign(z) *
                (1. - torch.exp(-torch.abs(z))))

    def log_cdf(self, x):
        """
        Cumulative distribution function.
        Given the variable X, the CDF is:
        CDF(x): = P[X <= x]

        :param
                x: The variable

        :return:
                CDF(x)
        """
        #

        z = (x - self.mu) / self._b
        cdf = (0.5 + 0.5 * torch.sign(z) *
               (1. - torch.exp(-torch.abs(z))))
        return torch.log(cdf)
