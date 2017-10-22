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

if __name__ == '__main__':
    run_tests()
