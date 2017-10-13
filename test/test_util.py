import torch
from torch.autograd import Variable
from common import TestCase, run_tests
from probtorch import util

zero = Variable(torch.Tensor([0.0]))

class TestWrap2D(TestCase):
    def test_softmax(self):
        for dims in range(1, 5):
            sizes = range(1, 1+dims)
            r = Variable(torch.rand(*sizes))
            for dim in range(dims):
                # ensure that values sum to 1 along specified dim
                sr = util.softmax(r, dim)
                srs = sr.sum(dim).data
                self.assertEqual(srs, torch.ones(srs.size()))
            # ensure that default normalizes along last dim
            self.assertEqual(sr, util.softmax(r))

    def test_log_softmax(self):
        for dims in range(1, 5):
            sizes = range(1, 1+dims)
            r = Variable(torch.rand(*sizes))
            for dim in range(dims):
                lsr = util.log_softmax(r, dim)
                # ensure that values sum to 1 along specified dim
                sr = torch.exp(lsr)
                srs = sr.sum(dim).data
                self.assertEqual(srs, torch.ones(srs.size()))
            # ensure that default normalizes along last dim
            self.assertEqual(lsr, util.log_softmax(r))

    def test_log_sum_exp(self):
        for dims in range(1, 5):
            sizes = range(1, 1+dims)
            r = Variable(torch.rand(*sizes))
            self.assertEqual(r.exp().sum().log(), 
                             util.log_sum_exp(r))
            for dim in range(dims):
                #print('dims=%d, dim=%d' % (dims,dim))
                #print(r.exp().sum(dim).log())
                #print(util.log_sum_exp(r, dim))
                self.assertEqual(r.exp().sum(dim).log(), 
                                 util.log_sum_exp(r, dim))
                for keepdim in [True, False]:
                    #print('dims=%d, dim=%d keepdim=%s' % (dims, dim, keepdim))
                    #print(r.exp().sum(dim, keepdim).log())
                    #print(util.log_sum_exp(r, dim, keepdim))
                    self.assertEqual(r.exp().sum(dim, keepdim).log(), 
                                     util.log_sum_exp(r, dim, keepdim))

    def test_log_mean_exp(self):
        for dims in range(1, 5):
            sizes = range(1, 1+dims)
            r = Variable(torch.rand(*sizes))
            self.assertEqual(r.exp().sum().log(), 
                             util.log_sum_exp(r))
            for dim in range(dims):
                #print('dims=%d, dim=%d' % (dims,dim))
                #print(r.exp().sum(dim).log())
                #print(util.log_sum_exp(r, dim))
                self.assertEqual(r.exp().mean(dim).log(), 
                                 util.log_mean_exp(r, dim))
                for keepdim in [True, False]:
                    #print('dims=%d, dim=%d keepdim=%s' % (dims, dim, keepdim))
                    #print(r.exp().sum(dim, keepdim).log())
                    #print(util.log_sum_exp(r, dim, keepdim))
                    self.assertEqual(r.exp().mean(dim, keepdim).log(), 
                                     util.log_mean_exp(r, dim, keepdim))

if __name__ == '__main__':
    run_tests()