from scipy.stats import norm, kstest
import numpy as np
from probtorch.distributions.normal import Normal
import torch
from common import TestCase, run_tests, SAMPLE_COUNT
from torch.autograd import Variable

class TestNormal(TestCase):
        
    def test_logprob(self):
        mean = Variable(torch.randn(100))
        std = torch.exp(Variable(torch.randn(100)))
        value = Variable(torch.randn(100))
        dist = Normal(mean, std)

        # test log probability
        res1 = dist.log_prob(value).data
        res2 = norm.logpdf(value.data.numpy(), mean.data.numpy(), std.data.numpy())
        self.assertEqual(res1, res2)
    
    def test_sample(self): 
        # TODO: this only works for scalar continuous distributions,
        # just to make sure things are ok until we write a better sample test
        mean = Variable(torch.randn(1))
        std = torch.exp(Variable(torch.randn(1)))
        dist = Normal(mean, std)
        samples=torch.zeros(SAMPLE_COUNT)
        for n in range(SAMPLE_COUNT):
            samples[n] = dist.sample().data[0]
        _,p =kstest(samples.numpy(),'norm',(mean.data.numpy(),std.data.numpy()))
        assert p>0.05
	
	#legacy code 
	#def _test_sample(self, dist, mean, variance,
    #                 count=100, prec=None):
    #    sample_sum = dist.sample()
    #    for n in range(count-1):
    #        sample_sum += dist.sample()
    #    sample_mean = sample_sum / count
    #    # TODO: implement a frequentist test that accounts 
    #    # for the dimesionality of the variable, and test
    #    # for convergence of the empirical variance. 
    #    self.assertEqual(sample_mean, mean, prec=4*(variance/count)**0.5)
	
if __name__ == '__main__':
    run_tests()
        