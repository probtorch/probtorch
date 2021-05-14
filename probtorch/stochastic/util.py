from enum import Enum, auto, unique
from probtorch.stochastic import Trace, Provenance

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
