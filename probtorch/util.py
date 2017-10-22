"Helper functions that don't have a better place yet"
import torch
from torch.nn import functional as F
from functools import wraps
from numbers import Number
import math

__all__ = ['broadcast_size',
           'expanded_size',
           'batch_sum',
           'partial_sum',
           'log_sum_exp',
           'log_softmax',
           'softmax']

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
                raise ValueError("Broadcasting dimensions",
                                 "must be either equal or 1.")
            c_size += (a,)
    return c_size

def expanded_size(expand_size, orig_size):
    """Returns the expanded size given two sizes"""
    # strip leading 1s from original size
    if expand_size is None:
        return orig_size
    if orig_size == (1,):
        return expand_size
    else:
        return expand_size + orig_size

def batch_sum(v, sample_dim=None, batch_dim=None):
    keepdims = []
    if sample_dim is not None:
        keepdims.append(sample_dim)
    if batch_dim is not None:
        keepdims.append(batch_dim)
    return partial_sum(v, keepdims)

def partial_sum(v, keep_dims=[]):
    """Sums variable or tensor of all dimensions except the dimensions
    specified in keep_dims"""
    if not keep_dims:
        return v.sum()
    else:
        # check if we need to permute
        if any([k != d for k, d in enumerate(keep_dims)]):
            dims = list(keep_dims) + [d for d in range(v.dim())
                                      if d not in keep_dims]
            v = v.clone()
            v.permute(dims)
        size = list(v.size())[:len(keep_dims)] + [-1, ]
        return v.view(size).sum(-1)

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
    # TODO: torch.max(value, dim=None) threw an error
    # at time of writing -> post an issue with PyTorch devs
    # TODO: torch.max(v, dim=0, keepdim=False) does not eliminate
    # first dimension -> post an issue with PyTorch devs
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        # TODO: this works when value is a variable,
        # but not when it is a tensor, since `torch.sum`
        # returns a float for tensors, and `torch.log`
        # does not accept float input.
        return m + torch.log(torch.sum(torch.exp(value - m)))

def wrap_2d_unitary(f):
    @wraps(f)
    def g(input, dim=-1, **kwds):
        if dim == -1:
            if input.dim() <= 2:
                # 1d and 2d inputs work as expected
                return f(input)
            else:
                input_size = input.size()
                input_2d = input.contiguous().view(-1, input_size[-1])
                return f(input_2d, **kwds).view(*input_size)
        else:
            tinput = input.transpose(dim, -1)
            tinput_size = tinput.size()
            tinput_2d = tinput.contiguous().view(-1, tinput_size[-1])
            toutput = f(tinput_2d, **kwds).view(*tinput_size)
            return toutput.transpose(dim, -1)
    return g

log_softmax = wrap_2d_unitary(F.log_softmax)
softmax = wrap_2d_unitary(F.softmax)
