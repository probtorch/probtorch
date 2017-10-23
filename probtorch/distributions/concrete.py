"""Implements the Concrete distribution (a relaxation of the one-hot
Categorical)."""
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from probtorch.distributions.distribution import *
from probtorch.util import log_sum_exp, expanded_size
from numbers import Number

__all__ = [
    "Concrete"
]


class Concrete(Distribution):
    r"""The Gumbel-Softmax relaxation of the discrete distribution, as described
    in [1] and [2].

    Arguments:
        log_weights(:obj:`Variable`): Unnormalized log probabilities.

        temperature(:obj:`Variable`): Temperature parameter.
        size(:obj:`torch.Size`, optional): Sample size.

        log_pdf(bool): Controls whether probabilities are calculated \
        according to the probability density function of the relaxation, \
        or according to the probability mass function. Defaults to False.

    References:
        [1] Chris J Maddison, Andriy Mnih, Yee Whye Teh. The concrete
        distribution: A continuous relaxation of discrete random variables.
        ICLR 2017.

        [2] Eric Jang, Shixiang Gu, Ben Poole. Categorical Reparameterization
        with Gumbel-Softmax. ICLR 2017.
    """

    def __init__(self, log_weights, temperature, log_pdf=False):
        self._log_pdf = log_pdf
        self._temperature = temperature
        self._log_weights = log_weights
        self._log_probs = F.log_softmax(self._log_weights,
                                        self._log_weights.dim() - 1)
        # TODO: we should just be able to call log_weights.type()
        # but apparently the variable API does not expose this
        # call syntax.
        super(Concrete, self).__init__(self._log_weights.size(),
                                       self._log_weights.data.type(),
                                       GradientType.REPARAMETERIZED)

    @property
    def log_weights(self):
        return self._log_weights

    @property
    def temperature(self):
        return self._temperature

    @property
    def mean(self):
        return torch.exp(self._log_probs)

    def sample(self, *sizes):
        size = expanded_size(sizes, self._size)
        uniform = Variable(torch.rand(size).type(self._type))
        gumbel = - torch.log(- torch.log(uniform + self.EPS) + self.EPS)
        logits = (self._log_weights + gumbel) / self._temperature
        return F.softmax(logits, logits.dim() - 1)

    def log_pmf(self, value):
        """Returns the probability mass, which is the probability of the argmax
        of the value under the corresponding Discrete distribution."""
        if value.data.type() != 'torch.LongTensor':
            _, value = value.max(-1)
        if value.dim() < len(self._size[:-1]):
            value = value.expand(*self._size[:-1])
        log_probs = self._log_probs
        if value.dim() > len(self._size[:-1]):
            log_probs = self._log_probs.expand(*value.size(), self._size[-1])
        return log_probs.gather(-1, value.unsqueeze(-1)).squeeze(-1)

    def log_pdf(self, value):
        """Returns the marginal probability density for the relaxed value."""
        k = self._size[-1]
        log_value = torch.log(value)
        if isinstance(self._temperature, Number):
            log_temp = math.log(self._temperature)
        else:
            log_temp = torch.log(self._temperature)
        return (Variable(torch.lgamma(torch.Tensor([k]) - 1.)) +
                (k - 1.) * log_temp +
                torch.sum(self._log_weights, -1) +
                torch.sum(log_value * (self._temperature - 1.0), -1) -
                k * log_sum_exp(self._log_weights +
                                log_value * self._temperature, -1))

    def log_prob(self, value):
        if self._log_pdf:
            return self.log_pdf(value)
        else:
            return self.log_pmf(value)
