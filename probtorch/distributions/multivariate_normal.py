import math
from numbers import Number

import torch
from torch.autograd import Variable
from probtorch.distributions.distribution import Distribution, GradientType

__all__ = [
    "MultivariateNormal"
]

def _get_batch_shape(bmat, bvec):
    """
    Given a batch of matrices and a batch of vectors, compute the combined `batch_shape`.
    """
    try:
        vec_shape = torch._C._infer_size(bvec.shape, bmat.shape[:-1])
    except RuntimeError:
        raise ValueError("Incompatible batch shapes: vector {}, matrix {}".format(bvec.shape, bmat.shape))
    return torch.Size(vec_shape[:-1])


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    n = bvec.size(-1)
    batch_shape = _get_batch_shape(bmat, bvec)

    # to conform with `torch.bmm` interface, both bmat and bvec should have `.dim() == 3`
    bmat = bmat.expand(batch_shape + (n, n)).contiguous().view((-1, n, n))
    bvec = bvec.unsqueeze(-1).expand(batch_shape + (n, 1)).contiguous().view((-1, n, 1))
    return torch.bmm(bmat, bvec).squeeze(-1).view(batch_shape + (n,))


def _batch_potrf_lower(bmat):
    """
    Applies a Cholesky decomposition to all matrices in a batch of arbitrary shape.
    """
    n = bmat.size(-1)
    cholesky = torch.stack([C.potrf(upper=False) for C in bmat.unsqueeze(0).contiguous().view((-1, n, n))])
    return cholesky.view(bmat.shape)


def _batch_diag(bmat):
    """
    Returns the diagonals of a batch of square matrices.
    """
    return bmat.contiguous().view(bmat.shape[:-2] + (-1,))[..., ::bmat.size(-1) + 1]


def _batch_mahalanobis(L, x):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both L and x.
    """
    # TODO: use `torch.potrs` or similar once a backwards pass is implemented.
    flat_L = L.unsqueeze(0).contiguous().view((-1,) + L.shape[-2:])
    L_inv = torch.stack([torch.inverse(Li.t()) for Li in flat_L]).view(L.shape)
    return (x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0).sum(-1)


class MultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`

    Example:

        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        -0.2102
        -0.5429
        [torch.FloatTensor of size 2]

    Args:
        loc (Tensor or Variable): mean of the distribution
        covariance_matrix (Tensor or Variable): positive-definite covariance matrix

    """

    def __init__(self, loc, covariance_matrix):
        event_shape = torch.Size(loc.shape[-1:])
        if covariance_matrix.dim() < 2:
            raise ValueError("covariance_matrix must be two-dimensional")
        self.covariance_matrix = covariance_matrix
        batch_shape = _get_batch_shape(covariance_matrix, loc)
        self._batch_shape = batch_shape
        self.loc = loc
        super(MultivariateNormal, self).__init__(event_shape,
                                                 'torch.FloatTensor',
                                                 GradientType.REPARAMETERIZED)

    @property
    def scale_tril(self):
        if not hasattr(self, '_scale_tril'):
            self._scale_tril =  _batch_potrf_lower(self.covariance_matrix)
        return self._scale_tril

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape + self._batch_shape + self._size)
        eps = Variable(self.loc.data.new(*shape).normal_())
        return self.loc + _batch_mv(self.scale_tril, eps)

    def log_prob(self, value):
        diff = value - self.loc
        M = _batch_mahalanobis(self.scale_tril, diff)
        log_det = _batch_diag(self.scale_tril).abs().log().sum(-1)
        return -0.5 * (M + self.loc.size(-1) * math.log(2 * math.pi)) - log_det

