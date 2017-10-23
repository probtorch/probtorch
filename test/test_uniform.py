from scipy.stats import uniform, kstest
import numpy as np
from probtorch.distributions.uniform import Uniform
import torch
from common import TestCase, run_tests, SAMPLE_COUNT
from torch.autograd import Variable


class TestUniform(TestCase):
    def test_logprob(self):
        loc = Variable(torch.randn(100))
        scale = Variable(torch.randn(100).abs())
        value = Variable(torch.randn(100))
        dist = Uniform(loc, loc + scale)

        # test log probability
        res1 = dist.log_prob(value).data
        res2 = uniform.logpdf(value.data.numpy(),
                              loc.data.numpy(),
                              scale.data.numpy())
        res2[np.isinf(res2)] = dist.LOG_0
        self.assertEqual(res1, res2)

    def test_sample(self):
        # TODO: this only works for scalar continuous distributions,
        # just to make sure things are ok until we write a better sample test
        loc = torch.randn(1)[0]
        scale = torch.randn(1).abs()[0]
        dist = Uniform(loc, loc + scale)
        samples = torch.zeros(SAMPLE_COUNT)
        for n in range(SAMPLE_COUNT):
            samples[n] = dist.sample().data[0]
        _, p = kstest(samples.numpy(), 'uniform', (loc, scale))
        assert p > 0.05

    def test_size(self):
        for dims in range(1, 4):
            sizes = range(1, 1 + dims)
            # ensure that sample size is handled correctly
            for k in range(dims):
                loc = Variable(torch.randn(*sizes[k:]))
                scale = Variable(torch.randn(*sizes[k:]).abs())
                dist = Uniform(loc, loc + scale)
                value = dist.sample(*sizes[:k])
                self.assertEqual(sizes, value.size())
            # ensure that log_prob broadcasts values if needed
            loc = Variable(torch.randn(*sizes))
            scale = Variable(torch.randn(*sizes).abs())
            for k in range(dims):
                dist = Uniform(loc, loc + scale)
                value = Variable(torch.randn(*sizes[k:]))
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())
            # ensure that log_prob broadcasts parameters if needed
            value = Variable(torch.randn(*sizes))
            for k in range(dims):
                loc = Variable(torch.randn(*sizes[k:]))
                scale = Variable(torch.randn(*sizes[k:]).abs())
                dist = Uniform(loc, loc + scale)
                log_prob = dist.log_prob(value)
                self.assertEqual(sizes, log_prob.size())


if __name__ == '__main__':
    run_tests()
