"Helper functions that don't have a better place yet"
import torch
from numbers import Number
import math
from functools import wraps


__all__ = ['broadcast_size',
           'expanded_size',
           'batch_sum',
           'partial_sum',
           'log_sum_exp',
           'log_mean_exp']


def expand_inputs(f):
    """Decorator that expands all input tensors to add a sample dimensions"""
    @wraps(f)
    def g(*args, **kwargs):
        num_samples = kwargs.get('num_samples', None)
        if num_samples is not None:
            new_args = []
            new_kwargs = {}
            for arg in args:
                if hasattr(arg, 'expand'):
                    new_args.append(arg.expand(num_samples, *arg.size()))
                else:
                    new_args.append(arg)
            for k in kwargs:
                arg = kwargs[k]
                if hasattr(arg, 'expand'):
                    new_kwargs[k] = arg.expand(num_samples, *arg.size())
                else:
                    new_kwargs[k] = arg
                new_kwargs['num_samples'] = num_samples
            return f(*new_args, **new_kwargs)
        else:
            return f(*args, **kwargs)
    return g


def broadcast_size(a, b):
    """Returns the broadcasted size given two Tensors or Variables"""
    a_size = torch.Size((1,)) if isinstance(a, Number) else a.size()
    b_size = torch.Size((1,)) if isinstance(b, Number) else b.size()
    # order a and b by number of dimensions
    if len(b_size) > len(a_size):
        a_size, b_size = b_size, a_size
    # pad b with 1's if needed
    b_size = torch.Size((1,) * (len(a_size) - len(b_size))) + b_size
    c_size = a_size[0:0]
    for a, b in zip(a_size, b_size):
        if a == 1:
            c_size += (b,)
        elif b == 1:
            c_size += (a,)
        else:
            if a != b:
                raise ValueError("Broadcasting dimensions must be either equal"
                                 "or 1.")
            c_size += (a,)
    return c_size


def expanded_size(expand_size, orig_size):
    """Returns the expanded size given two sizes"""
    # strip leading 1s from original size
    if not expand_size:
        return orig_size
    if orig_size == (1,):
        return expand_size
    else:
        return expand_size + orig_size


def batch_sum(v, sample_dims=None, batch_dims=None):
    if sample_dims is None:
        sample_dims = ()
    elif isinstance(sample_dims, int):
        sample_dims = (sample_dims,)
    if batch_dims is None:
        batch_dims = ()
    elif isinstance(batch_dims, int):
        batch_dims = (batch_dims,)
    assert set(sample_dims).isdisjoint(set(batch_dims))
    keep_dims = tuple(sorted(set(sample_dims).union(set(batch_dims))))
    v_sum = partial_sum(v, keep_dims=keep_dims)
    # ToDo: Can we do this more elegantly?
    if len(keep_dims) == 2 and sample_dims[0] > batch_dims[0]:
        return v_sum.permute(1, 0)
    else:
        return v_sum


def partial_sum(v, keep_dims=[]):
    """Sums variable or tensor along all dimensions except those specified
    in `keep_dims`"""
    if len(keep_dims) == 0:
        return v.sum()
    else:
        keep_dims = sorted(keep_dims)
        drop_dims = list(set(range(v.dim())) - set(keep_dims))
        result = v.permute(*(keep_dims + drop_dims))
        size = result.size()[:len(keep_dims)] + (-1,)
        return result.contiguous().view(size).sum(-1)


def log_mean_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().mean(dim, keepdim).log()
    """
    if dim is None:
        s = value.view(-1).size(0)
    else:
        s = value.size(dim)
    return log_sum_exp(value, dim, keepdim) - math.log(s)


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
