import torch
from torch.autograd import Variable
import probtorch
from probtorch.distributions import Normal
from probtorch.util import log_mean_exp
from common import TestCase, run_tests


class TestLogBatchMarginal(TestCase):
    def test_normal(self):
        S = 3  # sample size
        B = 5  # batch size
        D = 2  # hidden dim
        mu = Variable(torch.randn(S, B, D))
        sigma = torch.exp(Variable(torch.randn(S, B, D)))
        q = probtorch.Trace()
        q.normal(mu=mu, sigma=sigma, name='z')
        z = q['z']
        value = z.value

        log_joint, log_prod_mar = q.log_batch_marginal(sample_dim=0, batch_dim=1, nodes=['z'])

        # compare result
        log_probs = Variable(torch.zeros(B, S, B, D))
        for b1 in range(B):
            for s in range(S):
                for b2 in range(B):
                    d = Normal(mu[s, b2], sigma[s, b2])
                    log_probs[b1, s, b2] = d.log_prob(value[s, b1])
        log_joint_2 = log_mean_exp(log_probs.sum(3), 2).transpose(0, 1)
        log_prod_mar_2 = log_mean_exp(log_probs, 2).sum(2).transpose(0, 1)

        self.assertEqual(log_joint, log_joint_2)
        self.assertEqual(log_prod_mar, log_prod_mar_2)


if __name__ == '__main__':
    run_tests()