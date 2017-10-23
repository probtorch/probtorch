"""Implements the Univariate Normal distribution."""

import torch
from torch.autograd import Variable
import math
from probtorch.distributions.distribution import *
from probtorch.util import expanded_size
from numbers import Number

__all__ = [
    "Normal"
]


class Normal(Distribution):
    r"""The univariate normal distribution.

    .. math::
       f(x \mid \mu, \sigma) =
           \sqrt{\frac{1}{2\pi \sigma^2}}
           \exp \left[ -\frac{1}{2} \frac{(x-\mu)^2}{\sigma^2} \right]

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\sigma^2` or :math:`\frac{1}{\tau}`
    ========  ==========================================

    Normal distribution can be parameterized either in terms of standard
    deviation or precision. The link between the two parametrizations is
    given by :math:`\tau = 1 / \sigma^2`

    Parameters:
        mu(:obj:`Variable`): Mean.
        sigma(:obj:`Variable`, optional): Standard deviation (sigma > 0).
        tau(:obj:`Variable`, optional): Precision (tau > 0).
        size(tuple, optional): Sample size.

    Attributes:
        mean(:obj:`Variable`): Mean (mu).
        mode(:obj:`Variable`): Mode (mu).
        variance(:obj:`Variable`): Variance (equal to sigma**2 or 1/tau)

    Note:
        Only one of sigma or tau can be specified. When neither is specified,
        the default is sigma = tau = 1.0
    """

    def __init__(self, mu, sigma=None, tau=None):
        if sigma is not None and tau is not None:
            raise ValueError("Either sigma or tau may be specified, not both.")
        if sigma is None and tau is None:
            sigma = 1.0
            tau = 1.0
        if tau is None:
            tau = sigma**-2
        if sigma is None:
            sigma = tau**-0.5
        self._mu = mu
        self._sigma = sigma
        self._tau = tau
        _mu0 = mu / sigma
        if isinstance(_mu0, Number):
            super(Normal, self).__init__((1,),
                                         'torch.FloatTensor',
                                         GradientType.REPARAMETERIZED)
        else:
            super(Normal, self).__init__(_mu0.size(),
                                         _mu0.data.type(),
                                         GradientType.REPARAMETERIZED)

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    @property
    def mean(self):
        return self._mu

    @property
    def mode(self):
        return self._mu

    @property
    def variance(self):
        return self._sigma**2

    def sample(self, *sizes):
        size = expanded_size(sizes, self._size)
        eps = Variable(torch.randn(*size).type(self._type))
        return self._mu + self._sigma * eps

    def log_prob(self, value):
        # TODO: hopefully this goes away soon
        log = math.log if isinstance(self._sigma, Number) else torch.log
        return -0.5 * (log(2 * math.pi * self._sigma**2) +
                       ((value - self._mu) / self._sigma)**2)
