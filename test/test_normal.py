from scipy.stats import norm, kstest
import math
from probtorch.distributions.normal import Normal
import torch
from common import TestCase, run_tests, SAMPLE_COUNT
from torch.autograd import Variable


class TestNormal(TestCase):
    def test_logprob(self):
        mu = Variable(torch.randn(100))
        sigma = torch.exp(Variable(torch.randn(100)))
        value = Variable(torch.randn(100))
        dist = Normal(mu, sigma)

        # test log probability
        res1 = dist.log_prob(value).data
        res2 = norm.logpdf(value.data.numpy(),
                           mu.data.numpy(),
                           sigma.data.numpy())
        self.assertEqual(res1, res2)

    def test_sample(self):
        # TODO: this only works for scalar continuous distributions,
        # just to make sure things are ok until we write a better sample test
        mu = torch.randn(1)[0]
        sigma = math.exp(torch.randn(1)[0])
        dist = Normal(mu, sigma)
        samples = dist.sample(SAMPLE_COUNT).data
        _, p = kstest(samples.numpy(), 'norm', (mu, sigma))
        assert p > 0.05

    def test_size(self):
        for dims in range(1, 4):
            sizes = range(1, 1 + dims)
            # ensure that sample size is handled correctly
            for k in range(dims):
                mu = Variable(torch.randn(*sizes[k:]))
                sigma = Variable(torch.exp(torch.randn(*sizes[k:])))
                dist = Normal(mu, sigma)
                value = dist.sample(*sizes[:k])
                self.assertEqual(sizes, value.size())
            # ensure that log_prob broadcasts values if needed
            mu = Variable(torch.randn(*sizes))
            sigma = Variable(torch.exp(torch.randn(*sizes)))
            for k in range(dims):
                dist = Normal(mu, sigma)
                value = Variable(torch.randn(*sizes[k:]))
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())
            # ensure that log_prob broadcasts parameters if needed
            value = Variable(torch.randn(*sizes))
            for k in range(dims):
                mu = Variable(torch.randn(*sizes[k:]))
                sigma = Variable(torch.exp(torch.randn(*sizes[k:])))
                dist = Normal(mu, sigma)
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())


if __name__ == '__main__':
    run_tests()
