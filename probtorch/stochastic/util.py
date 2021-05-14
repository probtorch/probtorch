from enum import Enum, auto, unique
from probtorch.stochastic import Trace, RandomVariable, ImproperRandomVariable


@unique
class WriteMode(Enum):
    LastWriteWins = auto()
    FirstWriteWins = auto()
    NoOverlaps = auto()


def copytraces(*traces, exclude_nodes=None, mode=WriteMode.NoOverlaps):
    """
    merge traces together. domains should be disjoint otherwise last-write-wins.
    """
    newtrace = Trace()
    if exclude_nodes is None:
        exclude_nodes = {}

    for tr in traces:
        for k, rv in tr.items():
            if k in exclude_nodes:
                continue
            elif k in newtrace:
                if mode == WriteMode.LastWriteWins:
                    newtrace._nodes[k] = tr[k]
                elif mode == WriteMode.FirstWriteWins:
                    continue
                elif mode == WriteMode.NoOverlaps:
                    raise RuntimeError("traces should not overlap")
                else:
                    raise TypeError("impossible specification")

            newtrace._inject(rv, name=k, silent=True)
    return newtrace


def rerun_with_detached_values(trace: Trace):
    """
    Rerun a trace with detached values, recomputing the computation graph so that
    value do not cause a gradient leak.
    """
    newtrace = Trace()

    def rerun_rv(rv):
        value = rv.value.detach()
        if isinstance(rv, RandomVariable):
            return RandomVariable(
                value=value,
                dist=rv.dist,
                provenance=rv.provenance,
                reparameterized=rv.reparameterized,
            )
        elif isinstance(rv, ImproperRandomVariable):
            return ImproperRandomVariable(
                value=value, log_density_fn=rv.log_density_fn, provenance=rv.provenance
            )
        else:
            raise NotImplementedError(
                "Only supports RandomVariable and ImproperRandomVariable"
            )

    for k, v in trace.items():
        newtrace._inject(rerun_rv(v), name=k)

    return newtrace
