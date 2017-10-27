from numbers import Number
from probtorch.util import sum_log_prob, log_mean_exp


def elbo(q, p, sample_dim=None, batch_dim=None, alpha=0.1):
    r"""Calculates an importance weighted Monte Carlo estimate of the
    semi-supervised evidence lower bound (ELBO)

    .. math:: \frac{1}{B} \sum_{b=1}^B
              \log \left[
                \frac{1}{S} \sum_{s=1}^S
                \frac{p(x^{(b)}, y^{(b)}, z^{(s,b)})}{q(z^{(s,b)} | x^{(b)})}
              \right]
              + \frac{\alpha}{B} \sum_{b=1}^B
              \log \left[
                \frac{1}{S} \sum_{s=1}^S
                \frac{q(y^{(b)}, z^{(s,b)} | x^{(b)})}
                     {q(z^{(s,b)} | x^{(b)})}
              \right]

    The sets of variables :math:`x`, :math:`y` and :math:`z` refer to:

        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.

        :math:`y`: The set of conditioned nodes in `q`.

        :math:`z`: The set of sampled nodes in `q`.

    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        alpha(float, default 0.1): Coefficient for the ML term.
    """
    z = [n for n in q.sampled() if n in p]
    log_qz = sum_log_prob(q, sample_dim, batch_dim, z)
    log_p = sum_log_prob(p, sample_dim, batch_dim)
    log_pq = (log_p - log_qz)
    log_qy = sum_log_prob(q, sample_dim, batch_dim, q.conditioned())
    if sample_dim is None:
        return log_pq.mean() + alpha * log_qy.mean()
    else:
        if isinstance(log_qy, Number):
            return log_mean_exp(log_pq, 0).mean()
        else:
            return (log_mean_exp(log_pq, 0).mean() +
                    alpha * log_mean_exp(log_qy, 0).mean())
