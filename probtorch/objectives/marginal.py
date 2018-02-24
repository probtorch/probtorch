from numbers import Number
from torch.nn.functional import softmax
from probtorch.objectives.montecarlo import log_like, ml


def elbo(q, p, sample_dim=None, batch_dim=None, alpha=0.1, beta=(1.0, 1.0, 1.0, 1.0),
         size_average=True, reduce=True):
    r"""Calculates an importance sampling estimate of the semi-supervised
    evidence lower bound (ELBO), Bob Combo

    .. math::
       E_{q(z | x, y)} \left[ \log p(x | y, z) \right]
       - \beta_{1} E_{q(z | x, y)} 
            \left[ \log \frac{q_{avg}(z)}{\prod_{i=1}^{D}q_{avg}(z_{d})} 
                   - \log \frac{p(z)}{\prod_{i=1}^{D} p(z_{d})} \right] \\
       - \beta_{2} E_{q(z | x, y)} \left[ \log \prod_{d=1}^{D}\frac{q_{avg}(z_{d})}{p(z_{d})} \right]
       - \beta_{3} E_{q(z | x, y)} \left[ \log \frac{q(z | x, y)}{q_{avg}(z)} \right] \\
       - \beta_{4} E_{q(z | x, y)} \left[ \log \frac{q(y | x)}{p(y)} \right]
       + (\beta_{4} + \alpha) E_{q(z | x)}\left[ \log \frac{q(y, z| x)}{q(z | x)} \right]

    The sets of variables :math:`x`, :math:`y` and :math:`z` refer to:
        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.
        :math:`y`: The set of conditioned nodes in `q`, which may or may
        not also be present in `q`.
        :math:`z`: The set of sampled nodes present in both `q` and `p`.
    Importance sampling is used to approximate the expectation over
    :math:`q(z| x,y)`.
    Distribution :math:`q_{avg}(z)` is the average encoding distribution of the elements in the batch:
    .. math:: q_{avg}(z^{(s,b)}) = \frac{1}{B} \sum_{b'=1}^{B}q(z^{(s,b)} | x^{(b')}, y^{(s,b')})
    Or in the supervised case:
    .. math:: q_{avg}(z^{(s,b)}) = \frac{1}{B} \sum_{b'=1}^{B}q(z^{(s,b)} | x^{(b')}, y^{(b')})
    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        alpha(float, default 0.1): Coefficient for the ML term.
        beta(float, default 1.0):  Coefficient for the KL term.
        gamma(tuple of int): Containing coefficients for total correlation,
            kl to the prior, and mutual information
        size_average (bool, optional): By default, the objective is averaged
            over items in the minibatch. When set to false, the objective is
            instead summed over the minibatch.
        reduce (bool, optional): By default, the objective is averaged or
           summed over items in the minibatch. When reduce is False, losses
           are returned without averaging or summation.
    """
    log_weights = q.log_joint(sample_dim, batch_dim, q.conditioned())
    return (log_like(q, p, sample_dim, batch_dim, log_weights,
                     size_average=size_average, reduce=reduce) -
            kl(q, p, sample_dim, batch_dim, log_weights, beta,
               size_average=size_average, reduce=reduce) +
            (beta[3] + alpha) * ml(q, sample_dim, batch_dim, log_weights,
                                   size_average=size_average, reduce=reduce))


def kl(q, p, sample_dim=None, batch_dim=None, log_weights=None, beta=(1.0, 1.0, 1.0, 1.0),
         size_average=True, reduce=True):
    r"""
    Computes a Monte Carlo estimate of the unnormalized KL divergence
    described for variable z.
    .. math::
       E_{q(z | x, y)} \left[ \log \frac{q(z | x, y)}{p(z)} \right]
       \simeq
       \frac{1}{S} \frac{1}{B} \sum_{s=1}^S \sum_{b=1}^B
       \left[\beta_{1}  \log \frac{q_{avg}(z^{(s,b)})}{\prod_{d=1}^{D}q_{avg}(z_{d}^{(s,b)})} + \\
             \beta_{2} \log \prod_{d=1}^{D} \frac{q_{avg}(z_{d}^{(s,b)})}{p(z_{d}^{(s,b)})} +
             \beta_{3} \log \frac{q(z^{(s,b)} | x^{(b)}, y^{(s,b)})}{q_{avg}(z^{(s,b)})}
             \beta_{4} \log \frac{q(y^{(b)} | x^{(b)})}{p(y^(b))}
       \right]
    The sets of variables :math:`x`, :math:`y` and :math:`z` refer to:
        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.
        :math:`y`: The set of conditioned nodes in `q`, which may or may
        not also be present in `q`.
        :math:`z`: The set of sampled nodes present in both `q` and `p`.
    Importance sampling is used to approximate the expectation over
    :math:`q(z| x,y)`.
    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        log_weights(:obj:`Variable` or number, optional): Log weights for
            samples. Calculated when not specified.
        beta(tuple of int): Containing coefficients for total correlation,
            kl to the prior, and mutual information
        size_average (bool, optional): By default, the objective is averaged
            over items in the minibatch. When set to false, the objective is
            instead summed over the minibatch.
        reduce (bool, optional): By default, the objective is averaged or
           summed over items in the minibatch. When reduce is False, losses
           are returned without averaging or summation.
    """
    y = q.conditioned()
    if log_weights is None:
        log_weights = q.log_joint(sample_dim, batch_dim, y)
    log_qy = log_weights
    log_py = p.log_joint(sample_dim, batch_dim, y)
    z = [n for n in q.sampled() if n in p]
    _, log_avg_pzd_prod = p.log_batch_marginal(sample_dim, batch_dim, z)
    log_avg_qz, log_avg_qzd_prod = q.log_batch_marginal(sample_dim, batch_dim, z)
    log_pz = p.log_joint(sample_dim, batch_dim, z)
    log_qz = q.log_joint(sample_dim, batch_dim, z)
    objective = (beta[0] * ((log_avg_qz - log_avg_qzd_prod) -
                            (log_pz - log_avg_pzd_prod)) +
                 beta[1] * (log_avg_qzd_prod - log_avg_pzd_prod) +
                 beta[2] * (log_qz - log_avg_qz) +
                 beta[3] * (log_qy - log_py))
    if sample_dim is not None:
        if isinstance(log_weights, Number):
            objective = objective.mean(0)
        else:
            weights = softmax(log_weights, 0)
            objective = (weights * objective).sum(0)
    if reduce:
        objective = objective.mean() if size_average else objective.sum()
        # mask should 0 for unobserved and 0 for 1 for observed
    return objective

