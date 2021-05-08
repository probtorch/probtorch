from collections import OrderedDict, MutableMapping
from .util import batch_sum, partial_sum, log_mean_exp
import abc
from enum import Enum
import re
import math

__all__ = ["Stochastic", "Factor", "RandomVariable", "Trace"]


class Provenance(Enum):
    SAMPLED = 0
    OBSERVED = 1
    REUSED = 2

class Stochastic(object):
    """Stochastic variables wrap Pytorch Variables to associate a log probability
    density or mass.

    Attributes:
        value(:obj:Variable): The value of the variable.
        log_prob(:obj:Variable): The log probability mass or density.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def value(self):
        """Holds the value of a stochastic variable"""

    @abc.abstractproperty
    def log_prob(self):
        """Holds the log probability of a stochastic variable"""

    @abc.abstractproperty
    def mask(self):
        """Holds a mask for batch items"""


class RandomVariable(Stochastic):
    """Random variables wrap a PyTorch Variable to associate a distribution
    and a log probability density or mass.

    Parameters:
        dist(:obj:`Distribution`): The distribution of the variable.
        value(:obj:`Variable`): The value of the variable.
        observed(bool): Indicates whether the value was sampled or observed.
    """

    def __init__(self, dist, value, provenance=Provenance.SAMPLED, mask=None,
                 use_pmf=True):
        self._dist = dist
        self._value = value
        if use_pmf and hasattr(dist, 'log_pmf'):
            self._log_prob = dist.log_pmf(value)
        else:
            self._log_prob = dist.log_prob(value)
        assert isinstance(provenance, Provenance)
        self._provenance = provenance
        self._mask = mask
        self._reparameterized = dist.has_rsample

    @property
    def dist(self):
        return self._dist

    @property
    def value(self):
        return self._value

    @property
    def observed(self):
        return self._provenance == Provenance.OBSERVED

    @property
    def provenance(self):
        return self._provenance

    @property
    def log_prob(self):
        return self._log_prob

    @property
    def mask(self):
        return self._mask

    @property
    def reparameterized(self):
        return self._reparameterized

    def __repr__(self):
        return "%s RandomVariable containing: %s" % (type(self._dist).__name__,
                                                     repr(self._value))


class Factor(Stochastic):
    """A Factor wraps a log probability density or mass without associating a
    value. The value attribute of a Factor node is `None`.

    Parameters:
        log_prob(:obj:`Variable`): The log-probability.
    """

    def __init__(self, log_prob, mask=None):
        self._value = None
        self._log_prob = log_prob
        self._mask = mask

    @property
    def value(self):
        return self._value

    @property
    def log_prob(self):
        return self._log_prob

    @property
    def mask(self):
        return self._mask

    def __repr__(self):
        return "Factor with log probability: %s" % repr(self._log_prob)


class Loss(Stochastic):
    """A Loss associates a log probability of the form `-loss(value, target)`,
    with the provided value.

    Parameters:
        loss(function): A PyTorch loss function.
        value(:obj:`Variable`): The value.
        target(:obj:`Variable`): The target value.
    """

    def __init__(self, loss, value, target, mask=None):
        self._loss = loss
        self._value = value
        self._log_prob = -loss(value, target)
        self._mask = mask

    @property
    def value(self):
        return self._value

    @property
    def log_prob(self):
        return self._log_prob

    @property
    def loss(self):
        return self._log_prob

    @property
    def mask(self):
        return self._mask

    def __repr__(self):
        return "Loss with log probability: %s" % repr(self._log_prob)


class Trace(MutableMapping):
    """A dictionary-like container for stochastic nodes. Entries are ordered
    and can be retrieved by key, which by convention is a string, or by index.
    A Trace is write-once, in the sense that an entry cannot be removed or
    reassigned.
    """

    def __init__(self):
        # TODO: Python 3 dicts are ordered as of 3.6,
        # so could we use a normal dict instead?
        self._nodes = OrderedDict()
        self._counters = {}
        self._mask = None

    def __getitem__(self, name):
        return self._nodes.get(name, None)

    def __setitem__(self, name, node):
        if not isinstance(node, Stochastic):
            raise TypeError("Argument node must be an instance of "
                            "probtorch.Stochastic")
        if name in self._nodes:
            raise ValueError("Trace already contains a node with "
                             "name: " + name)
        if (node.log_prob != node.log_prob).sum() > 0:
            raise ValueError("NaN log prob encountered in node"
                             "with name: " + name)
        self._nodes[name] = node

    def __delitem__(self, name):
        raise NotImplementedError("Nodes may not be deleted from a "
                                  "Trace.")

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __contains__(self, name):
        return name in self._nodes

    def __repr__(self):
        item_reprs = []
        for n in self:
            node = self[n]
            if isinstance(node, RandomVariable):
                dname = type(node.dist).__name__
            else:
                dname = type(node).__name__
            if isinstance(node, Factor):
                dtype = node.log_prob.type()
                dsize = 'x'.join([str(d) for d in node.log_prob.size()])
            else:
                dtype = node.value.type()
                dsize = 'x'.join([str(d) for d in node.value.size()])
            val_repr = "[%s of size %s]" % (dtype, dsize)
            node_repr = "%s(%s)" % (dname, val_repr)
            item_reprs.append("%s: %s" % (repr(n), node_repr))
        return "Trace{%s}" % ", ".join(item_reprs)

    def iloc(self, pos):
        """Indexes entries by integer position."""
        return list(self._nodes.values())[pos]

    def append(self, node):
        """Appends a node, storing it according to the name attribute. If the
        node does not have a name attribute, then a unique name is generated.
        """
        if not isinstance(node, Stochastic):
            raise TypeError("Argument node must be an instance of"
                            "probtorch.Stochastic")
        # construct a new node name
        if isinstance(node, RandomVariable):
            node_name = type(node.dist).__name__.lower()
        else:
            node_name = type(node).__name__.lower()
        while True:
            node_count = self._counters.get(node_name, 0)
            name = '%s_%d' % (node_name, node_count)
            self._counters[node_name] = node_count + 1
            if name not in self._nodes:
                break
        self._nodes[name] = node

    def extend(self, nodes):
        """Appends multiple nodes"""
        for node in nodes:
            self.append(node)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    def factor(self, log_prob, name=None):
        """Creates a new Factor node"""
        node = Factor(log_prob, mask=self._mask)
        if name is None:
            self.append(node)
        else:
            self[name] = node

    def loss(self, objective, value, target, name=None):
        """Creates a new Loss node"""
        self[name] = Loss(objective, value, target, mask=self._mask)

    def variable(self, Dist, *args, **kwargs):
        """Creates a new RandomVariable node"""
        name = kwargs.pop('name', None)
        value = kwargs.pop('value', None)
        provenance = kwargs.pop('provenance', None)
        dist = Dist(*args, **kwargs)
        if value is None:
            if dist.has_rsample:
                value = dist.rsample()
            else:
                value = dist.sample()
            provenance = Provenance.SAMPLED
        else:
            if not provenance:
                provenance = Provenance.OBSERVED
            if isinstance(value, RandomVariable):
                value = value.value
        node = RandomVariable(dist, value, provenance, mask=self._mask)
        if name is None:
            self.append(node)
        else:
            self[name] = node
        return value

    def factors(self):
        """Returns a generator over Factor nodes"""
        for name in self._nodes:
            if isinstance(self._nodes[name], Factor):
                yield name

    def losses(self):
        """Returns a generator over Loss nodes"""
        for name in self._nodes:
            if isinstance(self._nodes[name], Loss):
                yield name

    def variables(self):
        """Returns a generator over RandomVariable nodes"""
        for name in self._nodes:
            if isinstance(self._nodes[name], RandomVariable):
                yield name

    def reused(self):
        """Returns a generator over reused RandomVariable nodes"""
        for name in self._nodes:
            node = self._nodes[name]
            if isinstance(node, RandomVariable) and\
               node.provenance == Provenance.REUSED:
                yield name

    def observed(self):
        """Returns a generator over observed RandomVariable nodes"""
        for name in self._nodes:
            node = self._nodes[name]
            if isinstance(node, RandomVariable) and node.observed:
                yield name

    def sampled(self):
        """Returns a generator over sampled RandomVariable nodes"""
        for name in self._nodes:
            node = self._nodes[name]
            if isinstance(node, RandomVariable) and\
               node.provenance == Provenance.SAMPLED:
                yield name

    def conditioned(self):
        """Returns a generator over all conditioned nodes, consisting of
        observed RandomVariable nodes, Factor nodes and Loss nodes.
        """
        for name in self._nodes:
            node = self._nodes[name]
            if not isinstance(node, RandomVariable) or node.observed:
                yield name

    def log_joint(self, sample_dims=None, batch_dim=None, nodes=None,
                  reparameterized=True):
        """Returns the log joint probability, optionally for a subset of nodes.

        Arguments:
            nodes(iterable, optional): The subset of nodes to sum over. When \
            unspecified, the sum over all nodes is returned.
            sample_dims(tuple): The dimensions that enumerate samples.
            batch_dim(int): The dimension that enumerates batch items.
        """
        if nodes is None:
            nodes = self._nodes
        log_prob = 0.0
        for n in nodes:
            if n in self._nodes:
                node = self._nodes[n]
                log_p = batch_sum(node.log_prob,
                                  sample_dims,
                                  batch_dim)
                if batch_dim is not None and node.mask is not None:
                    log_p = log_p * node.mask
                log_prob = log_prob + log_p
        return log_prob

    def log_batch_marginal(self, sample_dim=None, batch_dim=None, nodes=None, bias=1.0):
        """Computes log batch marginal probabilities. Returns the log marginal joint
        probability, the log product of marginals for individual variables, and the
        log product over both variables and individual dimensions."""
        if batch_dim is None:
            return self.log_joint(sample_dim, batch_dim, nodes)
        if nodes is None:
            nodes = self._nodes
        log_pw_joints = 0.0
        log_marginals = 0.0
        log_prod_marginals = 0.0
        for n in nodes:
            if n in self._nodes:
                node = self._nodes[n]
                if not isinstance(node, RandomVariable):
                    raise ValueError(('Batch averages can only be computed '
                                      'for random variables.'))
                # convert values of size (*, B, **) to size (B, *, 1, **)
                value = node.value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
                if hasattr(node.dist, 'log_pmf'):
                    # log pairwise probabilities of size (B, B, *, **)
                    log_pw = node.dist.log_pmf(value).transpose(1, batch_dim + 1)
                else:
                    # log pairwise probabilities of size (B, B, *, **)
                    log_pw = node.dist.log_prob(value).transpose(1, batch_dim + 1)
                if sample_dim is None:
                    keep_dims = (0, 1)
                else:
                    keep_dims = (0, 1, sample_dim + 2)
                batch_size = node.value.size(batch_dim)
                # log pairwise joint probabilities (B, B) or (B, B, S)
                log_pw_joint = partial_sum(log_pw, keep_dims)

                if node.mask is not None:
                    log_pw_joint = log_pw_joint * node.mask
                log_pw_joints = log_pw_joints + log_pw_joint

                # perform bias correction for diagonal terms
                log_pw_joint[range(batch_size), range(batch_size)] -= math.log(bias)
                log_pw[range(batch_size), range(batch_size)] -= math.log(bias)
                # log average over pairs (B) or (S, B)
                log_marginal = log_mean_exp(log_pw_joint, 1).transpose(0, batch_dim)
                # log product over marginals (B) or (S, B)
                log_prod_marginal = batch_sum(log_mean_exp(log_pw, 1),
                                              sample_dim + 1, 0)
                if node.mask is not None:
                    log_marginal = log_marginal * node.mask
                    log_prod_marginal = log_prod_marginal * node.mask
                log_marginals = log_marginals + log_marginal
                log_prod_marginals = log_prod_marginals + log_prod_marginal
        # perform bias correction for log pairwise joint
        log_pw_joints[range(batch_size), range(batch_size)] -= math.log(bias)
        log_pw_joints = log_mean_exp(log_pw_joints, 1).transpose(0, batch_dim)
        return log_pw_joints, log_marginals, log_prod_marginals


def _autogen_trace_methods():
    import torch as _torch
    from torch import distributions as _distributions
    import inspect as _inspect
    import re as _re

    # monkey patch relaxed distribtions
    def relaxed_bernoulli_log_pmf(self, value):
        return (value > self.probs).type('torch.FloatTensor')

    def relaxed_categorical_log_pmf(self, value):
        _, max_index = value.max(-1)
        return self.base_dist._categorical.log_prob(max_index)

    _distributions.RelaxedBernoulli.log_pmf = relaxed_bernoulli_log_pmf

    _distributions.RelaxedOneHotCategorical.log_pmf = relaxed_categorical_log_pmf

    def camel_to_snake(name):
        s1 = _re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return _re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    for name, obj in _inspect.getmembers(_distributions):
        if hasattr(obj, "__bases__") and issubclass(obj, _distributions.Distribution):
            f_name = camel_to_snake(name).lower()
            doc="""Generates a random variable of type torch.distributions.%s""" % name
            try:
                # try python 3 first
                asp = _inspect.getfullargspec(obj.__init__)
            except Exception as e:
                # python 2
                asp = _inspect.getargspec(obj.__init__)

            arg_split = -len(asp.defaults) if asp.defaults else None
            args = ', '.join(asp.args[:arg_split])

            if arg_split:
                pairs = zip(asp.args[arg_split:], asp.defaults)
                kwargs = ', '.join(['%s=%s' % (n, v) for n, v in pairs])
                args = args + ', ' + kwargs

            env = {'obj': obj, 'torch': _torch}
            s = ("""def f({0}, name=None, value=None):
                    return self.variable(obj, {1}, name=name, value=value)""")
            input_args = ', '.join(asp.args[1:])
            exec(s.format(args, input_args), env)
            f = env['f']
            f.__name__ = f_name
            f.__doc__ = doc
            setattr(Trace, f_name, f)

    # add alias for relaxed_one_hot_categorical
    setattr(Trace, 'concrete', Trace.relaxed_one_hot_categorical)


_autogen_trace_methods()
