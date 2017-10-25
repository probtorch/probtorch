import math
from scipy.stats import beta, kstest
from probtorch.distributions.kumaraswamy import Kumaraswamy
import torch
import numpy as np
from common import TestCase, run_tests, SAMPLE_COUNT
from torch.autograd import Variable


class TestKumaraswamy(TestCase):
    # There is no implementation of Kumaraswamy in scipy, so we're checking
    # Kumaraswamy(1,b) against Beta(1,b) and Kumaraswamy(a,1) against Beta(a,1)

    def test_logprob(self):
        # Beta(1,b) = Kumaraswamy(1,b)
        a = torch.exp(Variable(torch.randn(10)))
        b = torch.ones_like(a)
        value = Variable(torch.randn(10))
        dist = Kumaraswamy(a, b)

        # test log probability
        res1 = dist.log_prob(value).data
        res2 = beta.logpdf(value.data.numpy(),
                           a.data.numpy(),
                           b.data.numpy())
        res2[np.isinf(res2)] = dist.LOG_0
        self.assertEqual(res1, res2)

        # Beta(a,1) = Kumaraswamy(a,1)
        b = torch.exp(Variable(torch.randn(100)))
        a = torch.ones_like(b)
        value = Variable(torch.randn(100))
        dist = Kumaraswamy(a, b)

        # test log probability
        res1 = dist.log_prob(value).data
        res2 = beta.logpdf(value.data.numpy(),
                           a.data.numpy(),
                           b.data.numpy())
        res2[np.isinf(res2)] = dist.LOG_0
        self.assertEqual(res1, res2)

    def test_sample(self):
        # TODO: this only works for scalar continuous distributions,
        # just to make sure things are ok until we write a better sample test

        # Beta(a,1) = Kumaraswamy(a,1)
        a = math.exp(torch.randn(1)[0])
        b = 1.0
        dist = Kumaraswamy(a, b)
        samples = dist.sample(SAMPLE_COUNT).data
        _, p = kstest(samples.numpy(), 'beta', (a, b))
        assert p > 0.05

        # Beta(1,b) = Kumaraswamy(1,b)
        # NOTE: There is an underflow problem when a<1 (which is the case at
        # this test). When a and b are not Variables, samples becomes
        # FloatTensor and this test fails. To pass, we explicitly set samples
        # to use double precision by making a and b Variables containing
        # DoubleTenor.
        b = Variable(torch.exp(torch.DoubleTensor(1).normal_()))
        a = Variable(torch.DoubleTensor([1.0]))
        dist = Kumaraswamy(a, b)
        samples = dist.sample(SAMPLE_COUNT).data
        _, p = kstest(samples.numpy(),
                      'beta', (a.data.numpy(), b.data.numpy()))
        assert p > 0.05

    def test_size(self):
        for dims in range(1, 4):
            sizes = range(1, 1 + dims)
            # ensure that sample size is handled correctly
            for k in range(dims):
                a = Variable(torch.exp(torch.randn(*sizes[k:])))
                b = Variable(torch.exp(torch.randn(*sizes[k:])))
                dist = Kumaraswamy(a, b)
                value = dist.sample(*sizes[:k])
                self.assertEqual(sizes, value.size())

            # ensure that log_prob broadcasts values if needed
            a = Variable(torch.exp(torch.randn(*sizes)))
            b = Variable(torch.exp(torch.randn(*sizes)))
            for k in range(dims):
                dist = Kumaraswamy(a, b)
                value = Variable(torch.randn(*sizes[k:]))
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())

            # ensure that log_prob broadcasts parameters if needed
            value = Variable(torch.randn(*sizes))
            for k in range(dims):
                a = Variable(torch.exp(torch.randn(*sizes[k:])))
                b = Variable(torch.exp(torch.randn(*sizes[k:])))
                dist = Kumaraswamy(a, b)
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())


if __name__ == '__main__':
    run_tests()
