"""Implements the Concrete distribution (a relaxation of the one-hot Categorical)."""
import torch
from torch.autograd import Variable
from probtorch.distributions.distribution import *
from probtorch.util import log_sum_exp, log_softmax, softmax, expanded_size
from numbers import Number

__all__ = [
    "Concrete"
]

# TODO: does PyTorch have a default EPS somewhere?
EPS = 1e-12

class Concrete(Distribution):
    """The Gumbel-Softmax relaxation of the discrete distribution, as described
    in [1] and [2]. 

    Arguments:
        log_weights(:obj:`Variable`): Unnormalized log probabilities.
        
        temperature(:obj:`Variable`): Temperature parameter. 
        size(:obj:`torch.Size`, optional): Sample size.
        
        log_pdf(bool): Controls whether probabilities are calculated according \
        to the probability density function of the relaxation, or according to \
        the probability mass function. Defaults to False. 

    References:
        [1] Chris J Maddison, Andriy Mnih, Yee Whye Teh. The concrete distribution: 
        A continuous relaxation of discrete random variables. ICLR 2017.
        
        [2] Eric Jang, Shixiang Gu, Ben Poole. Categorical Reparameterization with 
        Gumbel-Softmax. ICLR 2017.
    """
    def __init__(self, log_weights, temperature, size=None, log_pdf=False):
        self._log_pdf = log_pdf
        if isinstance(temperature, Number):
            temperature = Variable(torch.Tensor([temperature]))
        self._temperature = temperature 
        if not size is None:
            size = expanded_size(size, log_weights.size())
            log_weights = log_weights.expand(*size)
        self._log_weights = log_weights
        self._log_probs = log_softmax(self._log_weights)
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

    def sample(self):
        uniform = Variable(torch.rand(self._size).type(self._type))
        gumbel = - torch.log(- torch.log(uniform + EPS) + EPS)
        logits = (self._log_weights + gumbel) / self._temperature
        return softmax(logits)

    def log_pmf(self, value):
        if value.data.type() != 'torch.LongTensor':
            _, value = value.max(-1)
        if value.dim() < len(self.size[:-1]):
            value = value.expand(*self.size[:-1])
        # score according to marginal probabilities
        return self._log_probs.gather(-1, value.unsqueeze(-1)).squeeze(-1)

    def log_pdf(self, value):
        # calculate probabilities under pdf 
        k = Variable(torch.Tensor([self._size[-1]]))
        log_value = torch.log(value)
        return (Variable(torch.lgamma(k.data - 1.))
                + (k - 1.) * torch.log(self._temperature)
                + torch.sum(self._log_weights, -1)
                + torch.sum(log_value * (self._temperature - 1.0), -1)
                - k * log_sum_exp(self._log_weights 
                                  + log_value * self._temperature, -1))

    def log_prob(self, value):
        if self._log_pdf:
            return self.log_pdf(value)
        else:
            return self.log_pmf(value)