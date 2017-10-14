from collections import OrderedDict, MutableMapping
from . import distributions 
import abc

__all__ = ["Stochastic", "Factor", "RandomVariable", "Trace"]

class Stochastic(object):
    """Stochastic variables wrap Pytorch Variables to associate a log probability
    density or mass.

    Attributes:
        value(:class:Variable): The value of the variable.
        log_prob(:class:Variable): The log probability mass or density.
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
    and log probability.

    Parameters:
        dist(:class:`Distribution`): The distribution of the variable.
        value(:class:`Variable`): The value of the variable.
        observed(bool): Indicates whether the value was sampled or observed.
    """
    def __init__(self, dist, value, observed=False):
        self._dist = dist
        self._value = value
        self._log_prob = None
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
        if self._log_prob is None:
            self._log_prob = self._dist.log_prob(self._value)
        return self._log_prob

    def __repr__(self):
        return "%s RandomVariable containing: %s" % (type(self._dist).__name__,
                                                     repr(self._value.data))
class Factor(Stochastic):
    """A Factor wraps a log probability without an associated value. A Factor 
    only provides a log_prob attribute. The value attribute is `None`.

    Parameters:
        log_prob(:class:`Variable`): The log-probability.
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
    """A Loss associates a log probability with a variable of the form 
    `-loss(value, target)`.

    Parameters:
        loss(:class:`Function`): A PyTorch loss function.
        value(:class:`Variable`): The value.
        target(:class:`Variable`): The target value.
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
    """A dictionary-like container for stochastic variables. Entries are 
    ordered and can be retrieved by key, which by convention is a string,
    or by index. A Trace is write-once, in the sense that an entry cannot 
    be removed or reassigned. 
    """
    def __init__(self):
        # TODO Python 3 dicts are ordered as of 3.6,
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

    def factor(self, log_prob, name=None, value=None):
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

    # TODO: we need to automate this, and add docstring magic
    def normal(self, mu, sigma=None, tau=None, name=None, value=None, **kwargs):
        """Creates a new Normal-distributed RandomVariable node."""
        return self.variable(distributions.Normal, mu, sigma=None, tau=None, 
                             name=name, value=value, **kwargs)

    def concrete(self, log_weights, temp, name=None, value=None, **kwargs):
        """Creates a new Concrete-distributed RandomVariable node."""
        return self.variable(distributions.Concrete, log_weights, temp, 
                             name=name, value=value, **kwargs)

    def uniform(self, lower=0.0, upper=1.0, name=None, value=None, **kwargs):
        """Creates a new Uniform-distributed RandomVariable node."""
        return self.variable(distributions.Uniform, lower, upper, 
                             name=name, value=value, **kwargs)
