"""Implements the Concrete distribution (a relaxation of the one-hot Categorical)."""
import torch
from torch.autograd import Variable
from probtorch.distributions.distribution import *
from probtorch.util import log_sum_exp, log_softmax, softmax

__all__ = [
    "Concrete"
]

# TODO: does PyTorch have a default EPS somewhere?
EPS = 1e-12

class Concrete(Distribution):
    "The concrete distribution (also known as the Gumbel-softmax distribution)"

    def __init__(self, log_weights, temperature, log_pdf=False):
        self._log_pdf = log_pdf
        self._log_weights = log_weights 
        self._log_probs = log_softmax(log_weights)
        self._temperature = temperature
        # TODO: we should just be able to call log_weights.type()
        # but apparently the variable API does not expose this 
        # call syntax.
        super(Concrete, self).__init__(log_weights.size(),
                                       log_weights.data.type(),
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