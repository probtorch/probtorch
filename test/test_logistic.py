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
        res2[np.isinf(res2)]=dist.LOG_0
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
        mu = Variable(torch.randn(1))
        s = torch.exp(Variable(torch.randn(1)))
        dist = Logistic(mu, s)
        samples=torch.zeros(SAMPLE_COUNT)
        for n in range(SAMPLE_COUNT):
            samples[n] = dist.sample().data[0]
        _,p = kstest(samples.numpy(), 
                     'logistic',
                     (mu.data.numpy(),s.data.numpy()))
        assert p>0.05

if __name__ == '__main__':
    run_tests()
        