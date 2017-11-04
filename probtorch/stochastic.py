from collections import OrderedDict, MutableMapping
from .util import batch_sum
import abc

__all__ = ["Stochastic", "Factor", "RandomVariable", "Trace"]


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


class RandomVariable(Stochastic):
    """Random variables wrap a PyTorch Variable to associate a distribution
    and a log probability density or mass.

    Parameters:
        dist(:obj:`Distribution`): The distribution of the variable.
        value(:obj:`Variable`): The value of the variable.
        observed(bool): Indicates whether the value was sampled or observed.
    """

    def __init__(self, dist, value, observed=False):
        self._dist = dist
        self._value = value
        self._log_prob = dist.log_prob(value)
        self._observed = observed

    @property
    def dist(self):
        return self._dist

    @property
    def value(self):
        return self._value

    @property
    def observed(self):
        return self._observed

    @property
    def log_prob(self):
        return self._log_prob

    def __repr__(self):
        return "%s RandomVariable containing: %s" % (type(self._dist).__name__,
                                                     repr(self._value.data))


class Factor(Stochastic):
    """A Factor wraps a log probability density or mass without associating a
    value. The value attribute of a Factor node is `None`.

    Parameters:
        log_prob(:obj:`Variable`): The log-probability.
    """

    def __init__(self, log_prob):
        self._value = None
        self._log_prob = log_prob

    @property
    def value(self):
        return self._value

    @property
    def log_prob(self):
        return self._log_prob

    def __repr__(self):
        return "Factor with log probability: %s" % repr(self._log_prob.data)


class Loss(Stochastic):
    """A Loss associates a log probability of the form `-loss(value, target)`,
    with the provided value.

    Parameters:
        loss(function): A PyTorch loss function.
        value(:obj:`Variable`): The value.
        target(:obj:`Variable`): The target value.
    """

    def __init__(self, loss, value, target):
        self._loss = loss
        self._value = value
        self._log_prob = -loss(value, target)

    @property
    def value(self):
        return self._value

    @property
    def log_prob(self):
        return self._log_prob

    @property
    def loss(self):
        return self._log_prob

    def __repr__(self):
        return "Loss with log probability: %s" % repr(self._log_prob.data)


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

    def __getitem__(self, name):
        return self._nodes[name]

    def __setitem__(self, name, node):
        if not isinstance(node, Stochastic):
            raise TypeError("Argument node must be an instance of "
                            "probtorch.Stochastic")
        if name in self._nodes:
            raise ValueError("Trace already contains a node with "
                             "name: " + name)
        if (node.log_prob.data != node.log_prob.data).sum() > 0:
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
            dname = type(node.dist).__name__
            dtype = node.value.data.type()
            dsize = 'x'.join([str(d) for d in node.value.data.size()])
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

    def factor(self, log_prob, name=None):
        """Creates a new Factor node"""
        node = Factor(log_prob)
        if name is None:
            self.append(node)
        else:
            self[name] = node

    def loss(self, objective, value, target, name=None):
        """Creates a new Loss node"""
        self[name] = Loss(objective, value, target)

    def variable(self, Dist, *args, **kwargs):
        """Creates a new RandomVariable node"""
        name = kwargs.pop('name', None)
        value = kwargs.pop('value', None)
        dist = Dist(*args, **kwargs)
        if value is None:
            value = dist.sample()
            observed = False
        else:
            observed = True
            if isinstance(value, RandomVariable):
                value = value.value
        node = RandomVariable(dist, value, observed)
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

    def observed(self):
        """Returns a generator over RandomVariable nodes"""
        for name in self._nodes:
            node = self._nodes[name]
            if isinstance(node, RandomVariable) and node.observed:
                yield name

    def sampled(self):
        """Returns a generator over RandomVariable nodes"""
        for name in self._nodes:
            node = self._nodes[name]
            if isinstance(node, RandomVariable) and not node.observed:
                yield name

    def conditioned(self):
        """Returns a generator over all conditioned nodes, consisting of
        observed RandomVariable nodes, Factor nodes and Loss nodes.
        """
        for name in self._nodes:
            node = self._nodes[name]
            if not isinstance(node, RandomVariable) or node.observed:
                yield name

    def log_joint(self, sample_dim=None, batch_dim=None, nodes=None):
        """Returns the log joint probability, optionally for a subset of nodes.

        Arguments:
            nodes(iterable, optional): The subset of nodes to sum over. When \
            unspecified, the sum over all nodes is returned.
            sample_dim(int): The dimension that enumerates samples.
            batch_dim(int): The dimension that enumerates batch items.
        """
        if nodes is None:
            nodes = self._nodes
        log_prob = 0.0
        for n in nodes:
            if n in self._nodes:
                log_p = batch_sum(self._nodes[n].log_prob,
                                  sample_dim,
                                  batch_dim)
                log_prob = log_prob + log_p
        return log_prob


def _autogen_trace_methods():
    from . import distributions as _distributions
    from .distributions.distribution import Distribution as _Distribution
    import inspect as _inspect

    def param_doc(doc):
        keep = False
        doc_body = ["Arguments:"]
        for line in doc.split('\n'):
            if keep and not line.split():
                break
            elif not keep and (line.strip().lower() == 'parameters:'):
                keep = True
            elif keep:
                doc_body.append("    " + line.strip())
        return "\n".join(doc_body) + "\n"

    for name, obj in _inspect.getmembers(_distributions):
        if hasattr(obj, "__bases__") and _Distribution in obj.__bases__:
            f_name = name.lower()
            doc_head = ("Creates a %s-distributed random variable node and "
                        "returns its value.\n\nFor more information, refer to the "
                        ":class:`~probtorch.distributions.%s` distribution.\n\n" % (f_name, name))
            doc_body = param_doc(obj.__doc__)
            doc_foot = (
                "    value(:obj:`Variable`, optional): Value of the RandomVariable instance.\n"
                "        When specified, the variable is observed. When not specified a value is\n"
                "        sampled and the variable is not observed.\n"
                "    name(string, optional): The name for the RandomVariable. When not\n"
                "        specified, a unique name is dynamically generated.")
            doc = doc_head + doc_body + doc_foot
            try:
                asp = _inspect.getfullargspec(obj.__init__)  # try python3 first
            except Exception as e:
                asp = _inspect.getargspec(obj.__init__)  # python 2

            arg_split = -len(asp.defaults) if asp.defaults else None
            args = ', '.join(asp.args[:arg_split])

            if arg_split:
                pairs = zip(asp.args[arg_split:], asp.defaults)
                kwargs = ', '.join(['%s=%s' % (n, v) for n, v in pairs])
                args = args + ', ' + kwargs

            env = {'obj': obj}
            s = ("""def f({0}, name=None, value=None):
                    return self.variable(obj, {1}, name=name, value=value)""")
            input_args = ', '.join(asp.args[1:])
            exec(s.format(args, input_args), env)
            f = env['f']
            f.__name__ = f_name
            f.__doc__ = doc
            setattr(Trace, f_name, f)


_autogen_trace_methods()
