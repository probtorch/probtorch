from probtorch.util import batch_sum, log_sum_exp
import torch

def log_like(q, p, sample_dim=None, batch_dim=None):
    r"""Calculates a Monte Carlo estimate of the log-likelihood.

    The expectation in this estimate is calculated with respect to
    unobserved variables in the encoder trace.

    When `batch_dim` is specified, this function returns the mean
    of the estimates for each item in the batch.

    When `sample_dim` is specified, this function computes an
    importance sampling estimate for each item.
    """
    log_P = 0.0
    log_W = None
    for v in p:
        if (v not in q):
            log_p = batch_sum(p[v].log_prob,
                              sample_dim=sample_dim,
                              batch_dim=batch_dim)
            log_P = log_P + log_p
    if sample_dim is None:
        return log_P.mean()
    else:
        for v in q:
            if q[v].observed:
                log_w = batch_sum(q[v].log_prob,
                                  sample_dim=sample_dim,
                                  batch_dim=batch_dim)
                if log_W is None:
                    log_W = 0.0
                log_W = log_W + log_w
            if log_W is None:
                return log_P.mean(0).mean()
            else:
                W = torch.exp(log_W - log_sum_exp(log_W, 0, True))
                return (W * log_P).sum(0).mean()

def kl(q, p, alpha=0.0, sample_dim=None, batch_dim=None):
    r"""Calculates a Monte Carlo estimate of the KL divergence.

    The expectation in this estimate is calculated with respect to
    unobserved variables in the encoder trace.

    When `batch_dim` is specified, this function returns the mean
    of the estimates for each item in the batch.

    When `sample_dim` is specified, this function computes an
    importance sampling estimate for each item.
    """
    log_P = 0.0
    log_Q = 0.0
    log_W = None
    for v in q:
        if (v in p):
            log_p = batch_sum(p[v].log_prob,
                              sample_dim=sample_dim,
                              batch_dim=batch_dim)
            log_q = batch_sum(q[v].log_prob,
                              sample_dim=sample_dim,
                              batch_dim=batch_dim)
            log_P = log_P + log_p
            log_Q = log_Q + log_q
        if q[v].observed:
            if log_W is None:
                log_W = 0.0
            log_W = log_W + log_q
    if sample_dim is None:
        if log_W is None:
            return (log_Q - log_P).mean()
        else:
            return (log_Q - log_P - (1.0 + alpha) * log_W).mean()
    else:
        if log_W is None:
            return (log_Q - log_P).mean(0).mean()
        else:
            W = torch.exp(log_W - log_sum_exp(log_W, 0, True))
            return (W * (log_Q - log_P)).sum(0).mean() - (1.0 + alpha) * log_W.mean(0).mean()

def elbo(q, p, alpha=0.0, sample_dim=None, batch_dim=None):
    r"""Calculates a Monte Carlo estimate of the evidence lower bound.

    The expectation in this estimate is calculated with respect to
    unobserved variables in the encoder trace.

    When `batch_dim` is specified, this function returns the mean
    of the estimates for each item in the batch.

    When `sample_dim` is specified, this function computes an
    importance sampling estimate for each item.
    """
    return log_like(q, p, sample_dim, batch_dim) - \
        kl(q, p, alpha, sample_dim, batch_dim)
