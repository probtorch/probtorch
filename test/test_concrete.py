from probtorch.distributions import Concrete
import torch
from common import TestCase, run_tests
from torch.autograd import Variable

class TestConcrete(TestCase):
    def test_size(self):
        for dims in range(1, 4):
            sizes = range(1, 1 + dims)
            # ensure that sample size is handled correctly
            for k in range(dims):
                log_weights = Variable(torch.randn(*sizes[k:]))
                dist = Concrete(log_weights, 0.66)
                value = dist.sample(*sizes[:k])
                self.assertEqual(sizes, value.size())
            # ensure that log_prob broadcasts values if needed
            log_weights = Variable(torch.randn(*sizes))
            for k in range(dims):
                dist = Concrete(log_weights, 0.66)
                value = Variable(torch.exp(torch.randn(*sizes[k:])))
                log_prob = dist.log_prob(value)
                if dims == 1:
                    self.assertEqual(sizes, log_prob.size())
                else:
                    self.assertEqual(sizes[:-1], log_prob.size())
            # ensure that log_prob broadcasts parameters if needed
            value = Variable(torch.exp(torch.randn(*sizes)))
            for k in range(dims):
                log_weights = Variable(torch.randn(*sizes[k:]))
                dist = Concrete(log_weights, 0.66)
                log_prob = dist.log_prob(value)
                if dims == 1:
                    self.assertEqual(sizes, log_prob.size())
                else:
                    self.assertEqual(sizes[:-1], log_prob.size())

if __name__ == '__main__':
    run_tests()
