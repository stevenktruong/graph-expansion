from copy import deepcopy
from typing import TypeVar

from .Coefficient import Coefficient, S, Theta
from .Graph import Graph, Size, Trace
from .MatrixFactor import G, MatrixFactor

T = TypeVar("T", MatrixFactor, list[MatrixFactor])


def adjoint(x: T) -> T:
    if isinstance(x, list):
        return deepcopy([adjoint(x0) for x0 in x])
    else:
        out = deepcopy(x)
        out.__adjoint__()
        return out


def p(x: Graph) -> int:
    return len([c for c in x.coefficients if isinstance(c, S)])


def q(x: Graph) -> int:
    return len(x.light_weights)


def r(x: Graph) -> int:
    return len(x.g_loops)


def k(t: Trace) -> int:
    return len([f for f in t if isinstance(f, G)])


def n(x: Graph) -> int:
    return sum([k(t) for t in x.g_loops])


def size(x: Graph) -> Size:
    return Size(
        -(p(x) + q(x)),
        r(x)
        - q(x)
        - n(x)
        - len(
            [
                c
                for c in x.coefficients
                if isinstance(c, Theta) and c.charges[0] != c.charges[1]
            ]
        ),
    )


def _mul(self: Coefficient | Trace | Graph, other) -> Graph:
    if isinstance(other, Coefficient | Trace | Graph):
        return Graph(self, other)
    else:
        raise TypeError(f"Tried to multiply {self.__class__} by {other}")


Coefficient.__mul__ = Coefficient.__rmul__ = _mul
Trace.__mul__ = Trace.__rmul__ = _mul
Graph.__mul__ = Graph.__rmul__ = _mul
