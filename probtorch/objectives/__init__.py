from . import montecarlo
from . import importance
from probtorch.util import batch_sum

def log_joint(p, sample_dim=None, batch_dim=None):
    log_P = 0.0
    for v in p:
        log_p = batch_sum(p[v].log_prob, sample_dim, batch_dim)
        log_P = log_P + log_p
    return log_P

def log_observed(p, sample_dim=None, batch_dim=None):
    log_P = 0.0
    for v in p:
        if p[v].observed:
            log_p = batch_sum(p[v].log_prob, sample_dim, batch_dim)
            log_P = log_P + log_p
    return log_P


def log_sampled(p, sample_dim=None, batch_dim=None):
    log_P = 0.0
    for v in p:
        if not p[v].observed:
            log_p = batch_sum(p[v].log_prob, sample_dim, batch_dim)
            log_P = log_P + log_p
    return log_P
