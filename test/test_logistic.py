import math
from scipy.stats import logistic, kstest
import numpy as np
from probtorch.distributions.logistic import Logistic
import torch
from common import TestCase, run_tests, SAMPLE_COUNT
from torch.autograd import Variable


class TestLogistic(TestCase):
    def test_logprob(self):
        mu = Variable(torch.randn(100))
        s = torch.exp(Variable(torch.randn(100)))
        value = Variable(torch.randn(100))
        dist = Logistic(mu, s)

        # test log probability
        res1 = dist.log_prob(value).data
        res2 = logistic.logpdf(value.data.numpy(),
                               mu.data.numpy(),
                               s.data.numpy())
        res2[np.isinf(res2)] = dist.LOG_0
        self.assertEqual(res1, res2)

    def test_cdf(self):
        mu = Variable(torch.randn(100))
        s = torch.exp(Variable(torch.randn(100)))
        value = Variable(torch.randn(100))
        dist = Logistic(mu, s)

        # test cdf
        res1 = dist.cdf(value).data
        res2 = logistic.cdf(value.data.numpy(),
                            mu.data.numpy(),
                            s.data.numpy())
        self.assertEqual(res1, res2)

    def test_inv_cdf(self):
        mu = Variable(torch.randn(100))
        s = torch.exp(Variable(torch.randn(100)))
        value = Variable(torch.rand(100))
        dist = Logistic(mu, s)

        # test inverse cdf
        res1 = dist.inv_cdf(value).data
        res2 = logistic.ppf(value.data.numpy(),
                            mu.data.numpy(),
                            s.data.numpy())
        self.assertEqual(res1, res2)

    def test_sample(self):
        # TODO: this only works for scalar continuous distributions,
        # just to make sure things are ok until we write a better sample test
        mu = torch.randn(1)[0]
        s = math.exp(torch.randn(1)[0])
        dist = Logistic(mu, s)
        samples = dist.sample(SAMPLE_COUNT).data
        _, p = kstest(samples.numpy(), 'logistic', (mu, s))
        assert p > 0.05

    def test_size(self):
        for dims in range(1, 4):
            sizes = range(1, 1 + dims)
            # ensure that sample size is handled correctly
            for k in range(dims):
                mu = Variable(torch.randn(*sizes[k:]))
                s = Variable(torch.exp(torch.randn(*sizes[k:])))
                dist = Logistic(mu, s)
                value = dist.sample(*sizes[:k])
                self.assertEqual(sizes, value.size())
            # ensure that log_prob broadcasts values if needed
            mu = Variable(torch.randn(*sizes))
            s = Variable(torch.exp(torch.randn(*sizes)))
            for k in range(dims):
                dist = Logistic(mu, s)
                value = Variable(torch.randn(*sizes[k:]))
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())
            # ensure that log_prob broadcasts parameters if needed
            value = Variable(torch.randn(*sizes))
            for k in range(dims):
                mu = Variable(torch.randn(*sizes[k:]))
                s = Variable(torch.exp(torch.randn(*sizes[k:])))
                dist = Logistic(mu, s)
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())


if __name__ == '__main__':
    run_tests()
