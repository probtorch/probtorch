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

        lam = torch.exp(Variable(torch.randn(1)))
        dist = Exponential(lam, size=(SAMPLE_COUNT,))
        samples = dist.sample().data
        _, p = kstest(samples.numpy(), 'expon', (0, 1.0 / lam.data.numpy()))
        assert p > 0.05

if __name__ == '__main__':
    run_tests()
