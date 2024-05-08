from copy import deepcopy
from Graph import N, Graph, Trace
from MatrixFactor import E, G, M, Alpha, MatrixFactor, wtG
from helpers import transpose


def Xf(x: Graph, i: int) -> tuple[list[MatrixFactor], Graph]:
    t = deepcopy(x.traces[i])
    f = Graph(
        x.n.exponent,
        deepcopy(x.traces[:i]) + deepcopy(x.traces[i + 1 :]),
        deepcopy(x.coefficients),
    )
    return (t.factors[1:], f)


def ND(x: Trace) -> list[list[MatrixFactor]]:
    out: list[list[MatrixFactor]] = []

    for i, f in enumerate(x.factors):
        if isinstance(f, G) or isinstance(f, wtG):
            product = (
                [G(like=f)]
                + deepcopy(x.factors[i + 1 :])
                + deepcopy(x.factors[:i])
                + [G(like=f)]
            )
            out.append(transpose(product))
            out.append(product)

    return out


def wtGpM(X: list[MatrixFactor], f: Graph) -> list[Graph]:
    return [Trace(wtG(), X) * deepcopy(f), Trace(M(), X) * deepcopy(f)]


def MGG(X: list[MatrixFactor], f: Graph, alpha: Alpha) -> Graph:
    return N(-1) * Trace(M(), E(alpha), G(), E(alpha), G(), X) * deepcopy(f)


def wtGMG(X: list[MatrixFactor], f: Graph, alpha: Alpha) -> Graph:
    return Trace(wtG(), E(alpha)) * Trace(M(), E(alpha), G(), X) * deepcopy(f)


def DX(X: list[MatrixFactor], f: Graph, alpha: Alpha) -> list[Graph]:
    out: list[Graph] = []
    # X = X[: i]
    # Y = X[i+1: ]
    for i, Ga in enumerate(X):
        if isinstance(Ga, G) or isinstance(Ga, wtG):
            out.append(
                N(-1)
                * Trace(
                    M(),
                    E(alpha),
                    transpose(G(like=Ga)),
                    transpose(X[:i]),
                    transpose(G()),
                    E(alpha),
                    G(like=Ga),
                    deepcopy(X[i + 1 :]),
                )
                * deepcopy(f)
            )
            out.append(
                Trace(M(), E(alpha), G(like=Ga), deepcopy(X[i + 1 :]))
                * Trace(G(), deepcopy(X[:i]), G(like=Ga), E(alpha))
                * deepcopy(f)
            )

    return out


def Df(X: list[MatrixFactor], f: Graph, alpha: Alpha) -> list[Graph]:
    out: list[Graph] = []
    for i, t in enumerate(f.traces):
        remaining = Graph(
            f.n.exponent, deepcopy(f.traces[:i]), deepcopy(f.traces[i + 1 :])
        )
        for Nderivative in ND(t):
            out.append(
                N(-2) * Trace(M(), E(alpha), Nderivative, E(alpha), G(), X) * remaining
            )
    return out
