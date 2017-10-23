import torch
from random import randint, sample
from torch.autograd import Variable
from common import TestCase, run_tests
from probtorch import util

zero = Variable(torch.Tensor([0.0]))

class TestWrap2D(TestCase):
    def test_partial_sum(self):
        for dims in range(1, 5):
            size = [randint(1, 5) for _ in range(dims)]
            r = Variable(torch.rand(*size))
            # compute partial sum by iterative summation over keep_dims complement
            keep_dims = sample(range(dims), randint(0, dims - 1))
            res = r
            if len(keep_dims) == 0:
                res = res.sum()
            else:
                drop_dims = list(set(range(dims)) - set(keep_dims))
                for d in sorted(drop_dims, reverse=True):
                    res = res.sum(d)
            self.assertEqual(res, util.partial_sum(r, keep_dims=keep_dims))

    def test_log_sum_exp(self):
        for dims in range(1, 5):
            sizes = range(1, 1 + dims)
            r = Variable(torch.rand(*sizes))
            self.assertEqual(r.exp().sum().log(),
                             util.log_sum_exp(r))
            for dim in range(dims):
                # print('dims=%d, dim=%d' % (dims,dim))
                # print(r.exp().sum(dim).log())
                # print(util.log_sum_exp(r, dim))
                self.assertEqual(r.exp().sum(dim).log(),
                                 util.log_sum_exp(r, dim))
                for keepdim in [True, False]:
                    self.assertEqual(r.exp().sum(dim, keepdim).log(),
                                     util.log_sum_exp(r, dim, keepdim))

    def test_log_mean_exp(self):
        for dims in range(1, 5):
            sizes = range(1, 1 + dims)
            r = Variable(torch.rand(*sizes))
            self.assertEqual(r.exp().sum().log(),
                             util.log_sum_exp(r))
            for dim in range(dims):
                # print('dims=%d, dim=%d' % (dims,dim))
                # print(r.exp().sum(dim).log())
                # print(util.log_sum_exp(r, dim))
                self.assertEqual(r.exp().mean(dim).log(),
                                 util.log_mean_exp(r, dim))
                for keepdim in [True, False]:
                    self.assertEqual(r.exp().mean(dim, keepdim).log(),
                                     util.log_mean_exp(r, dim, keepdim))

if __name__ == '__main__':
    run_tests()
