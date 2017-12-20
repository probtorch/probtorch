from probtorch.util import log_mean_exp

def elbo(q, p, sample_dim=None, batch_dim=None, alpha=0.1,
         beta=1.0, gamma=1.0, size_average=True, reduce=True):
    r"""
    Calculates a Monte Carlo estimate of the average encoding
    evidence lower bound (ELBO)

    .. math::
       E_{q(z | x)} \left[ \log p(x | z) \right]
       - \alpha E_{q(z | x)} \left[ \log \frac{q_{avg}(z)}{ \prod_{d=1}^{D} q_{avg}(z_{d})} \right] \\
       - \beta E_{q(z | x)}\left[ \log \prod_{d=1}^{D}\frac{q_{avg}(z_{d})}{p(z_{d})} \right]
       - \gamma E_{q(z | x)}\left[ \log \frac{q(z | x)}{q_{avg}(z)} \right]

    The sets of variables :math:`x` and :math:`z` refer to:

        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.

        :math:`z`: The set of sampled nodes present in both `q` and `p`.

    Distribution :math:`q_{avg}(z)` is the average encoding distribution of the elements in the batch:

    .. math:: q_{avg}(z) = \frac{1}{B} \sum_{b=1}^{B}q(z | x^{(b)})


    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        alpha(float, default 1.0): Coefficient for the disentanglement term.
        beta(float, default 1.0):  Coefficient for the realism term.
        gamma(float, default 1.0):  Coefficient for the mutual-information term.
        size_average (bool, optional): By default, the objective is averaged
            over items in the minibatch. When set to false, the objective is
            instead summed over the minibatch.
        reduce (bool, optional): By default, the objective is averaged or
           summed over items in the minibatch. When reduce is False, losses
           are returned without averaging or summation.
    """

    disentangle_loss = 0.0
    realism_loss = 0.0
    mutual_info_loss = 0.0

    zs = [n for n in q.sampled() if n in p]
    for z in zs:
        log_pairs = q.log_pair(sample_dim, batch_dim, z)
        log_avg_qz = log_mean_exp(log_pairs.sum(-1), 2)
        log_avg_qzd_prod = log_mean_exp(log_pairs, 2).sum(-1)
        log_pz = p[z].log_prob.sum(-1)
        log_qz = q[z].log_prob.sum(-1)
        disentangle_loss += log_avg_qz.sub(log_avg_qzd_prod)
        realism_loss += log_avg_qzd_prod.sub(log_pz)
        mutual_info_loss += log_qz.sub(log_avg_qz)

    if reduce:
        disentangle_loss = disentangle_loss.mean() if size_average else disentangle_loss.sum()
        realism_loss = realism_loss.mean() if size_average else realism_loss.sum()
        mutual_info_loss = mutual_info_loss.mean() if size_average else mutual_info_loss.sum()

    return (log_like(q, p, sample_dim, batch_dim,
                     size_average=size_average, reduce=reduce) -
            alpha * disentangle_loss - beta * realism_loss - gamma * mutual_info_loss)

    # TODO: compute the losses separately via the function below
    # return (log_like(q, p, sample_dim, batch_dim,
    #                 size_average=size_average, reduce=reduce) -
    #         alpha * disentangle(q, p, sample_dim, batch_dim,
    #                 size_average=size_average, reduce=reduce) -
    #         beta * realism(q, p, sample_dim, batch_dim,
    #                 size_average=size_average, reduce=reduce) -
    #         gamma * mutual_info(q, p, sample_dim, batch_dim,
    #                 size_average=size_average, reduce=reduce))

def log_like(q, p, sample_dim=None, batch_dim=None,
             size_average=True, reduce=True):
    r"""Computes a Monte Carlo estimate of the log-likelihood.

    .. math::
       E_{q(z | x)}[\log p(x | z)]
       \simeq \frac{1}{S} \frac{1}{B} \sum_{s=1}^S \sum_{b=1}^B
              \log p(x^{(b)} | z^{(s,b)})

    The sets of variables :math:`x` and :math:`z` refer to:

        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.

        :math:`z`: The set of sampled nodes present in both `q` and `p`.


    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        size_average (bool, optional): By default, the objective is averaged
            over items in the minibatch. When set to false, the objective is
            instead summed over the minibatch.
        reduce (bool, optional): By default, the objective is averaged or
           summed over items in the minibatch. When reduce is False, losses
           are returned without averaging or summation.
    """
    x = [n for n in p.conditioned() if n not in q]
    objective = p.log_joint(sample_dim, batch_dim, x)
    if reduce:
        objective = objective.mean() if size_average else objective.sum()
    return objective

def disentangle(q, p, sample_dim=None, batch_dim=None,
             size_average=True, reduce=True):
    r"""Computes a Monte Carlo estimate of the KL divergence term between the joint
    average encoding distribution and the product of average encoding distribution
    in each dimension independently.

    .. math::
       E_{q(z | x)}\left[ \log \frac{q_{avg}(z)}{ \prod_{d=1}^{D} q_{avg}(z_{d})} \right]
       \simeq \frac{1}{S} \frac{1}{B} \sum_{s=1}^S \sum_{b=1}^B
              \left[ \log \frac{q_{avg}(z^{(s,b)})}{\prod_{d=1}^{D} q_{avg}(z_{d}^{(s,b)})} \right] \\
              = \frac{1}{S} \frac{1}{B} \sum_{s=1}^S \sum_{b=1}^B
              \left[ \log \frac{\frac{1}{B}\sum_{b'=1}^{B}q(z^{(s,b)} | x^{(b')})}
              {\prod_{d=1}^{D} \left[ \frac{1}{B}\sum_{b''=1}^{B} q(z_{d}^{(s,b)} | x^{(b'')})
               \right] } \right]

    The sets of variables :math:`x` and :math:`z` refer to:

        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.

        :math:`z`: The set of sampled nodes present in both `q` and `p`.


    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        size_average (bool, optional): By default, the objective is averaged
            over items in the minibatch. When set to false, the objective is
            instead summed over the minibatch.
        reduce (bool, optional): By default, the objective is averaged or
           summed over items in the minibatch. When reduce is False, losses
           are returned without averaging or summation.
    """

    zs = [n for n in q.sampled() if n in p]
    objective = 0.0
    for z in zs:
        log_pairs = q.log_pair(sample_dim, batch_dim, z)
        log_avg_qz = log_mean_exp(log_pairs.sum(-1), 2)
        log_avg_qzd_prod = log_mean_exp(log_pairs, 2).sum(-1)
        objective = objective + log_avg_qz.sub(log_avg_qzd_prod)
    if reduce:
        objective = objective.mean() if size_average else objective.sum()

    return objective

def realism(q, p, sample_dim=None, batch_dim=None,
             size_average=True, reduce=True):
    r"""Computes a Monte Carlo estimate of the KL divergence term between the
    average encoding distribution and the prior in each dimension independently.

    .. math::
       E_{q(z | x)}\left[\log \prod_{d=1}^{D}\frac{q_{avg}(z_{d})}{p(z_{d})} \right]
       \simeq \frac{1}{S} \frac{1}{B} \sum_{s=1}^S \sum_{b=1}^B
              \left[ \log \prod_{d=1}^{D}\frac{q_{avg}(z_{d}^{(s,b)})}{p(z_{d}^{(s,b)})} \right] \\
              = \frac{1}{S} \frac{1}{B} \sum_{s=1}^S \sum_{b=1}^B
              \left[ \log \prod_{d=1}^{D} \frac{\frac{1}{B}\sum_{b''=1}^{B} q(z_{d}^{(s,b)} | x^{(b'')})}
              { p(z_{d}^{(s,b)}) } \right]

    The sets of variables :math:`x` and :math:`z` refer to:

        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.

        :math:`z`: The set of sampled nodes present in both `q` and `p`.


    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        size_average (bool, optional): By default, the objective is averaged
            over items in the minibatch. When set to false, the objective is
            instead summed over the minibatch.
        reduce (bool, optional): By default, the objective is averaged or
           summed over items in the minibatch. When reduce is False, losses
           are returned without averaging or summation.
    """

    zs = [n for n in q.sampled() if n in p]
    objective = 0.0
    for z in zs:
        log_pairs = q.log_pair(sample_dim, batch_dim, z)
        log_avg_qzd_prod = log_mean_exp(log_pairs, 2).sum(-1)
        log_pz = p[z].log_prob.sum(-1)
        objective = objective + log_avg_qzd_prod.sub(log_pz)
    if reduce:
        objective = objective.mean() if size_average else objective.sum()

    return objective


def mutual_info(q, p, sample_dim=None, batch_dim=None,
                size_average=True, reduce=True):
    r"""Computes a Monte Carlo estimate of the mutual information term between
    teh observed and hidden variable.

    .. math::
       E_{q(z | x)}\left[ \log \frac{q(z | x)}{q_{avg}(z)} \right]
       \simeq \frac{1}{S} \frac{1}{B} \sum_{s=1}^S \sum_{b=1}^B
              \left[ \log \frac{q(z^{(s,b)} | x^{(b)})}{q_{avg}(z^{(s,b)})} \right] \\
              = \frac{1}{S} \frac{1}{B} \sum_{s=1}^S \sum_{b=1}^B
              \left[ \log \frac{q(z^{(s,b)} | x^{(b)})}
              {\frac{1}{B}\sum_{b'=1}^{B} q(z^{(s,b)} | x^{(b')})} \right]

    The sets of variables :math:`x` and :math:`z` refer to:

        :math:`x`: The set of conditioned nodes that are present in `p` but
        are not present in `q`.

        :math:`z`: The set of sampled nodes present in both `q` and `p`.


    Arguments:
        q(:obj:`Trace`): The encoder trace.
        p(:obj:`Trace`): The decoder trace.
        sample_dim(int, optional): The dimension containing individual samples.
        batch_dim(int, optional): The dimension containing batch items.
        size_average (bool, optional): By default, the objective is averaged
            over items in the minibatch. When set to false, the objective is
            instead summed over the minibatch.
        reduce (bool, optional): By default, the objective is averaged or
           summed over items in the minibatch. When reduce is False, losses
           are returned without averaging or summation.
    """
    zs = [n for n in q.sampled() if n in p]
    objective = 0.0
    for z in zs:
        log_pairs = q.log_pair(sample_dim, batch_dim, z)
        log_avg_qz = log_mean_exp(log_pairs.sum(-1), 2)
        log_qz = q[z].log_prob.sum(-1)
        objective = objective + log_qz.sub(log_avg_qz)
    if reduce:
        objective = objective.mean() if size_average else objective.sum()

    return objective


