import torch
from torch.autograd import Variable
import probtorch
from probtorch.objectives.average_encoding import disentangle, realism, mutual_info
from common import TestCase, run_tests
from scipy.stats import norm


class TestQstar(TestCase):
    def test_averageEncoding_normal(self):
        S = 3  # sample size
        B = 5  # batch size
        D = 2  # hidden dim
        mu = Variable(torch.randn(S, B, D))
        sigma = torch.exp(Variable(torch.randn(S, B, D)))
        q = probtorch.Trace()
        p = probtorch.Trace()
        q.normal(mu=mu, sigma=sigma, name='z')
        p.normal(mu=0.0, sigma=1.0, value=q['z'], name='z')
        z_node = q['z']
        z_node_p = p['z']
        value = z_node.value

        res1_joint, res1_sep = q.log_pair(sample_dim=0, batch_dim=1, nodes=['z'])

        # test every element in Q is computed correctly: Q[s,b1,b2,d] = q(z_{d}^{sb2}|x^b1)
        res2 = torch.zeros(S, B, B, D)
        for s in range(S):
            for b1 in range(B):
                for b2 in range(B):
                    for d in range(D):
                        log_pdf = norm.logpdf(value[s, b1, d].data[0], mu[s, b2, d].data[0], sigma[s, b2, d].data[0])
                        res2[s, b1, b2, d] = log_pdf


        # test the individual losses
        res2 = res2.exp()
        log_prob_pz = z_node_p.log_prob.data
        log_prob_qz = z_node.log_prob.data

        disentangle_loss1 = res2.prod(-1).sum(-1).div(B).div(res2.sum(2).prod(-1)).mul(B ** D).log().mean()
        realism_loss1 = res2.sum(2).prod(-1).div(B ** D).div(log_prob_pz.exp().prod(-1)).log().mean()
        mi_loss1 = log_prob_qz.exp().prod(-1).div(res2.prod(-1).sum(-1).div(B)).log().mean()

        disentangle_loss2 = disentangle(q, p, res1_joint, res1_sep, 0, 1).data[0]
        realism_loss2 = realism(q, p, res1_sep, 0, 1).data[0]
        mi_loss2 = mutual_info(q, p, res1_joint, 0, 1).data[0]

        self.assertEqual(disentangle_loss1, disentangle_loss2)
        self.assertEqual(realism_loss1, realism_loss2)
        self.assertEqual(mi_loss1, mi_loss2)


if __name__ == '__main__':
    run_tests()
