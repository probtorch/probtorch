from scipy.stats import laplace, kstest
from probtorch.distributions.laplace import Laplace
import torch
from common import TestCase, run_tests, SAMPLE_COUNT
from torch.autograd import Variable

class TestLaplace(TestCase):
        
    def test_logprob(self):
        mu = Variable(torch.randn(100))
        b = torch.exp(Variable(torch.randn(100)))
        value = Variable(torch.randn(100))
        dist = Laplace(mu, b)

        # test log probability
        res1 = dist.log_prob(value).data
        res2 = laplace.logpdf(value.data.numpy(),
                           mu.data.numpy(), 
                           b.data.numpy())
        self.assertEqual(res1, res2)
    
    def test_sample(self): 
        # TODO: this only works for scalar continuous distributions,
        # just to make sure things are ok until we write a better sample test
        mu = Variable(torch.randn(1))
        b = torch.exp(Variable(torch.randn(1)))
        dist = Laplace(mu, b, size=(SAMPLE_COUNT,))
        samples = dist.sample().data
        _,p = kstest(samples.numpy(), 
                     'laplace', (mu.data.numpy(), b.data.numpy()))
        assert p > 0.05
	
if __name__ == '__main__':
    run_tests()
        