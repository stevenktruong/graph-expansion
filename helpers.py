from typing import TypeVar
from copy import deepcopy

from Graph import Coefficient, Graph, Trace
from MatrixFactor import G, MatrixFactor, wtG


T = TypeVar(
    "T",
    MatrixFactor,
    list[MatrixFactor],
    Coefficient,
    Trace,
    Graph,
)


def transpose(x: T) -> T:
    if isinstance(x, MatrixFactor):
        out = deepcopy(x)
        out.transposed = not out.transposed
        return out
    elif isinstance(x, list):
        out = deepcopy(x)
        out.reverse()
        return [transpose(f) for f in out]
    else:
        raise TypeError("Tried to transpose a Coefficient, Graph, or Trace")


def conjugate(x: T) -> T:
    if isinstance(x, MatrixFactor) or isinstance(x, Coefficient):
        out = deepcopy(x)
        out.conjugated = not out.conjugated
        return out
    elif isinstance(x, list):
        out = deepcopy(x)
        return [conjugate(f) for f in out]
    elif isinstance(x, Trace):
        return Trace([conjugate(c) for c in x.factors])
    else:
        return Graph(
            x.n.exponent,
            [conjugate(c) for c in x.coefficients],
            [conjugate(t) for t in x.traces],
        )


def adjoint(x: T) -> T:
    return transpose(conjugate(x))
