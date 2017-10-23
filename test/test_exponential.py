import math
from scipy.stats import expon, kstest
from probtorch.distributions.exponential import Exponential
import torch
import numpy as np
from common import TestCase, run_tests, SAMPLE_COUNT
from torch.autograd import Variable


class TestExponential(TestCase):
    def test_logprob(self):

        lam = torch.exp(Variable(torch.randn(100)))
        value = Variable(torch.randn(100))
        dist = Exponential(lam)

        # test log probability
        res1 = dist.log_prob(value).data
        res2 = expon.logpdf(value.data.numpy(),
                            scale=1.0 / lam.data.numpy())
        res2[np.isinf(res2)] = dist.LOG_0
        self.assertEqual(res1, res2)

    def test_sample(self):
        # TODO: this only works for scalar continuous distributions,
        # just to make sure things are ok until we write a better sample test

        lam = math.exp(torch.randn(1)[0])
        dist = Exponential(lam)
        samples = dist.sample(SAMPLE_COUNT).data
        _, p = kstest(samples.numpy(), 'expon', (0, 1.0 / lam))
        assert p > 0.05

    def test_size(self):
        for dims in range(1, 4):
            sizes = range(1, 1 + dims)
            # ensure that sample size is handled correctly
            for k in range(dims):
                lam = Variable(torch.exp(torch.randn(*sizes[k:])))
                dist = Exponential(lam)
                value = dist.sample(*sizes[:k])
                self.assertEqual(sizes, value.size())
            # ensure that log_prob broadcasts values if needed
            lam = Variable(torch.exp(torch.randn(*sizes)))
            for k in range(dims):
                dist = Exponential(lam)
                value = Variable(torch.randn(*sizes[k:]))
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())
            # ensure that log_prob broadcasts parameters if needed
            value = Variable(torch.randn(*sizes))
            for k in range(dims):
                lam = Variable(torch.exp(torch.randn(*sizes[k:])))
                dist = Exponential(lam)
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())


if __name__ == '__main__':
    run_tests()
