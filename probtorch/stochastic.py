from collections import OrderedDict, MutableMapping
from . import distributions 
import abc

__all__ = ["Stochastic", "Factor", "RandomVariable", "Trace"]

class Stochastic(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def value(self):
        "Holds the value of a stochastic node"

    @abc.abstractproperty
    def log_prob(self):
        "Holds the log probability of a stochastic node"

class Factor(Stochastic):
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
    def __init__(self, loss, value, target):
        self._value = value
        self._log_prob = -loss(value, target)

    @property
    def value(self):
        return self._value

    @property
    def log_prob(self):
        return self._log_prob

    def __repr__(self):
        return "Loss with log probability: %s" % repr(self._log_prob.data)

class RandomVariable(Stochastic):
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

class Trace(MutableMapping):
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
        return list(self._nodes.values())[pos]

    def append(self, node):
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
        for node in nodes:
            self.append(node)

    def factor(self, log_prob, name=None, value=None):
        node = Factor(log_prob)
        if name is None:
            self.append(node)
        else:
            self[name] = node

    def loss(self, objective, value, target, name=None):
        self[name] = Loss(objective, value, target)

    def variable(self, Dist, *args, **kwargs):
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
    def normal(self, mean, std, name=None, value=None, **kwargs):
        return self.variable(distributions.Normal, mean, std, 
                             name=name, value=value, **kwargs)

    def concrete(self, log_weights, temp, name=None, value=None, **kwargs):
        return self.variable(distributions.Concrete, log_weights, temp, 
                             name=name, value=value, **kwargs)

