from probtorch.util import batch_sum, log_mean_exp

def elbo(q, p, alpha=0.0, sample_dim=None, batch_dim=None):
    r"""Calculates an importance weighted estimate of the evidence
    lower bound.

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
    for v in p:
        log_p = batch_sum(p[v].log_prob,
                          sample_dim=sample_dim,
                          batch_dim=batch_dim)
        log_P = log_P + log_p
    for v in q:
        if (v in p):
            log_q = batch_sum(q[v].log_prob,
                              sample_dim=sample_dim,
                              batch_dim=batch_dim)
            log_Q = log_Q + log_q
        if q[v].observed:
            if log_W is None:
                log_W = 0.0
            log_W = log_W + log_q
    if sample_dim is None:
        if log_W is None:
            return (log_P - log_Q).mean()
        else:
            return (log_P - log_Q + (1.0 + alpha) * log_W).mean()
    else:
        if log_W is None:
            return log_mean_exp(log_P - log_Q, 0).mean()
        else:
            return log_mean_exp(log_P - log_Q + (1.0 + alpha) * log_W, 0).mean()
