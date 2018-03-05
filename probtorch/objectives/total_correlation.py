from numbers import Number
from torch.nn.functional import softmax
from probtorch.objectives.montecarlo import log_like, ml


def elbo(q, p, N, sample_dim=None, batch_dim=None, alpha=0.1, beta=1.0,
         size_average=True, reduce=True):
    r"""Calculates an importance sampling estimate of the semi-supervised
    evidence lower bound (ELBO), https://arxiv.org/abs/1802.04942

    .. math::
      \mathcal{L}_{\beta-\text{TC}}
      = E_{q(z, x)\left[ \log p(x|z) \right]}
        - D_{\text{KL}}\left(q(z,x) \| q(z)\tilde{p}(x)\right)
        - \beta * D_{\text{KL}}\left(q(z) \| \prod_j q(z_j)\right)
        - \sum_j D_{\text{KL}}\left(q(z_j) \| p(z_j)\right)

    The sets of variables :math:`x`, :math:`y` and :math:`z` refer to:
        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.
        :math:`y`: The set of conditioned nodes in `q`, which may or may
        not also be present in `q`.
        :math:`z`: The set of sampled nodes present in both `q` and `p`.

    Importance sampling is used to approximate the expectation over
    :math:`q(z)`.
    Distribution :math:`q(z)` is the average encoding distribution of the elements in the batch:

    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        alpha(float, default 0.1): Coefficient for the ML term.
        beta(float, default 1.0):  Coefficient for the TC term.
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
            kl(q, p, N, sample_dim, batch_dim, log_weights, beta,
               size_average=size_average, reduce=reduce) +
            alpha * ml(q, sample_dim, batch_dim, log_weights,
                       size_average=size_average, reduce=reduce))


def kl(q, p, N, sample_dim=None, batch_dim=None, log_weights=None, beta=1.0,
       size_average=True, reduce=True):
    r"""
    Computes a Monte Carlo estimate of the unnormalized KL divergence
    described for variable z.

    .. math::
       TBC

    The sets of variables :math:`x`, :math:`y` and :math:`z` refer to:
        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.
        :math:`y`: The set of conditioned nodes in `q`, which may or may
        not also be present in `q`.
        :math:`z`: The set of sampled nodes present in both `q` and `p`.
    Importance sampling is used to approximate the expectation over
    :math:`q(z)`.
    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        log_weights(:obj:`Variable` or number, optional): Log weights for
            samples. Calculated when not specified.
        beta(int): Coefficient for total correlation
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
    log_qz = q.log_joint(sample_dim, batch_dim, z)
    log_pz = p.log_joint(sample_dim, batch_dim, z)
    log_marginals, log_marginals_d = q.log_batch_marginal2(N)
    objective = log_qz - log_pz + \
        (beta - 1) * log_marginals - \
        (beta - 1) * log_marginals_d
    if sample_dim is not None:
        if isinstance(log_weights, Number):
            objective = objective.mean(0)
        else:
            weights = softmax(log_weights, 0)
            objective = (weights * objective).sum(0)
    if reduce:
        objective = objective.mean() if size_average else objective.sum()
    return objective
