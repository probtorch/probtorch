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
        mu1 = Variable(torch.randn(S, B, D))
        mu2 = Variable(torch.randn(S, B, D))
        sigma1 = torch.exp(Variable(torch.randn(S, B, D)))
        sigma2 = torch.exp(Variable(torch.randn(S, B, D)))
        q = probtorch.Trace()
        q.normal(mu=mu1, sigma=sigma1, name='z1')
        q.normal(mu=mu2, sigma=sigma2, name='z2')
        z1 = q['z1']
        z2 = q['z2']
        value1 = z1.value
        value2 = z2.value

        log_joint, log_mar, log_prod_mar = q.log_batch_marginal(sample_dim=0,
                                                                batch_dim=1,
                                                                nodes=['z1', 'z2'])
        # compare result
        log_probs1 = Variable(torch.zeros(B, S, B, D))
        log_probs2 = Variable(torch.zeros(B, S, B, D))
        for b1 in range(B):
            for s in range(S):
                for b2 in range(B):
                    d1 = Normal(mu1[s, b2], sigma1[s, b2])
                    d2 = Normal(mu2[s, b2], sigma2[s, b2])
                    log_probs1[b1, s, b2] = d1.log_prob(value1[s, b1])
                    log_probs2[b1, s, b2] = d2.log_prob(value2[s, b1])

        log_mar_z1 = log_mean_exp(log_probs1.sum(3), 2).transpose(0, 1)
        log_mar_z2 = log_mean_exp(log_probs2.sum(3), 2).transpose(0, 1)
        log_prod_mar_z1 = log_mean_exp(log_probs1, 2).sum(2).transpose(0, 1)
        log_prod_mar_z2 = log_mean_exp(log_probs2, 2).sum(2).transpose(0, 1)

        log_joint_2 = log_mean_exp(log_probs1.sum(3) + log_probs2.sum(3), 2).transpose(0, 1)
        log_mar_2 = log_mar_z1 + log_mar_z2
        log_prod_mar_2 = log_prod_mar_z1 + log_prod_mar_z2

        self.assertEqual(log_mar, log_mar_2)
        self.assertEqual(log_prod_mar, log_prod_mar_2)
        self.assertEqual(log_joint, log_joint_2)

if __name__ == '__main__':
    run_tests()
