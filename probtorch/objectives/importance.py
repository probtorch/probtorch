from numbers import Number
from probtorch.util import log_mean_exp


def elbo(q, p, sample_dim=None, batch_dim=None, alpha=0.1,
         size_average=True, reduce=True):
    r"""Calculates an importance weighted Monte Carlo estimate of the
    semi-supervised evidence lower bound (ELBO)

    .. math:: \frac{1}{B} \sum_{b=1}^B
              \log \left[
                \frac{1}{S} \sum_{s=1}^S
                \frac{p(x^{(b)}, y^{(b)}, z^{(s,b)})}
                     {q(z^{(s,b)} | x^{(b)})}
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

        :math:`y`: The set of conditioned nodes in `q`, which may or may
        not also be present in `q`.

        :math:`z`: The set of sampled nodes present in both `q` and `p`.

    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        alpha(float, default 0.1): Coefficient for the ML term.
        size_average (bool, optional): By default, the objective is averaged
            over items in the minibatch. When set to false, the objective is
            instead summed over the minibatch.
        reduce (bool, optional): By default, the objective is averaged or
           summed over items in the minibatch. When reduce is False, losses
           are returned without averaging or summation.
    """
    z = [n for n in q.sampled() if n in p]
    log_pxyz = p.log_joint(sample_dim, batch_dim)
    log_qz = q.log_joint(sample_dim, batch_dim, z)
    log_qy = q.log_joint(sample_dim, batch_dim, q.conditioned())
    log_pq = (log_pxyz - log_qz)
    if sample_dim is None:
        objective = log_pq + alpha * log_qy
    else:
        objective = log_mean_exp(log_pq, 0)
        if not isinstance(log_qy, Number):
            objective = objective + alpha * log_mean_exp(log_qy, 0)
    if reduce:
        objective = objective.mean() if size_average else objective.sum()
    return objective
