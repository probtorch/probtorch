from scipy.stats import uniform, kstest
import numpy as np
from probtorch.distributions.uniform import Uniform
import torch
from common import TestCase, run_tests, SAMPLE_COUNT
from torch.autograd import Variable

class TestUniform(TestCase):
        
    def test_logprob(self):
        loc = Variable(torch.randn(100))
        scale = torch.exp(Variable(torch.randn(100)))
        value = Variable(torch.randn(100))
        dist = Uniform(loc, loc+scale)

        # test log probability
        res1 = dist.log_prob(value).data
        res2 = uniform.logpdf(value.data.numpy(),loc.data.numpy(), scale.data.numpy())
        res2[np.isinf(res2)]=dist.LOG_0
        self.assertEqual(res1, res2)
    
    def test_sample(self): 
        # TODO: this only works for scalar continuous distributions,
        # just to make sure things are ok until we write a better sample test
        loc = Variable(torch.randn(1))
        scale = torch.exp(Variable(torch.randn(1)))
        dist = Uniform(loc, loc+scale)
        samples=torch.zeros(SAMPLE_COUNT)
        for n in range(SAMPLE_COUNT):
            samples[n] = dist.sample().data[0]
        _,p =kstest(samples.numpy(),'uniform',(loc.data.numpy(),scale.data.numpy()))
        assert p>0.05

if __name__ == '__main__':
    run_tests()
        