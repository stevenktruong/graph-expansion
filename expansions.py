from typing import Callable
from Graph import Graph, ImF, Trace, k, last_G_index
from actions import *
from helpers import conjugate, transpose


def expand_light_weight(
    x: Graph, predicate: Callable[[Trace], bool] | None = None, solve: bool = True
) -> list[Graph]:

    i = 0
    if predicate:
        for j, t in enumerate(x.light_weights):
            if predicate(t):
                i = j
                break
        else:
            raise LookupError("Could not find light-weight from predicate")

    target = x.light_weights[i]
    _, f = Xf(x, x.traces.index(target))
    final_deterministic_start = last_G_index(target) + 1
    next_alpha = 1 + max(
        [f.alpha.index for t in x.traces for f in t.factors if isinstance(f, E)],
        default=0,
    )
    alpha = Alpha(next_alpha)
    beta = Alpha(next_alpha + 1)
    gamma = Alpha(next_alpha + 2)

    # Reduce to the case where we expand a \G that's not transposed or conjugated
    if target.factors[0].transposed:
        target = Trace(transpose(target.factors[0]), transpose(target.factors[1:]))

    conjugated = target.factors[0].conjugated
    if conjugated:
        target = conjugate(target)
        f = conjugate(f)

    def h(A: MatrixFactor | list[MatrixFactor]) -> list[Graph]:
        out: list[Graph] = []

        X_A = deepcopy(target.factors[1:final_deterministic_start]) + (
            [A] if isinstance(A, MatrixFactor) else A
        )

        out.append(MGG(X_A, f, alpha))

        # Self-consistent mechanism is from the <\G><MG> term
        out.append(
            Trace(wtG(), E(alpha)) * Trace(M(), E(alpha), wtG(), A) * deepcopy(f)
        )

        if not solve:
            out.append(
                Trace(wtG(), E(alpha)) * Trace(M(), E(alpha), M(), A) * deepcopy(f)
            )

        out.extend(DX(X_A, f, alpha))
        out.extend(Df(X_A, f, alpha))

        return out

    out = h(deepcopy(target.factors[final_deterministic_start:]))
    if solve:
        out += [
            ImF(beta, gamma)
            * Trace(
                M(), E(beta), M(), deepcopy(target.factors[final_deterministic_start:])
            )
            * x1
            for x1 in h(E(gamma))
        ]
    if conjugated:
        return [conjugate(x1) for x1 in out]
    else:
        return out


def expand_G_loop(
    x: Graph,
    predicate: Callable[[Trace], bool] | None = None,
    fallback: int | None = None,
    solve: bool = True,
    self_consistent_term: bool = True,
) -> list[Graph]:
    i = 0
    if predicate:
        for j, t in enumerate(x.g_loops):
            if predicate(t):
                i = j
                break
        else:
            if fallback is None:
                raise LookupError(
                    "Could not find G-loop from predicate with no fallback"
                )
            else:
                i = fallback

    target = x.g_loops[i]
    _, f = Xf(x, x.traces.index(target))

    # Reduce to the case where we expand a \G that's not transposed or conjugated
    transposed = target.factors[0].transposed
    if transposed:
        target = Trace(transpose(target.factors[0]), transpose(target.factors[1:]))

    conjugated = target.factors[0].conjugated
    if conjugated:
        target = conjugate(target)
        f = conjugate(f)

    # These need to be calculated after transposing, since transposes change the order
    final_deterministic_start = last_G_index(target) + 1
    next_alpha = 1 + max(
        [f.alpha.index for t in x.traces for f in t.factors if isinstance(f, E)],
        default=0,
    )
    alpha = Alpha(next_alpha)
    beta = Alpha(next_alpha + 1)
    gamma = Alpha(next_alpha + 2)

    def h(A: MatrixFactor | list[MatrixFactor]) -> list[Graph]:
        out: list[Graph] = []

        X_A = deepcopy(target.factors[1:final_deterministic_start]) + (
            [A] if isinstance(A, MatrixFactor) else A
        )

        i_Ga = last_G_index(target) - 1  # -1 since X_A doesn't include the first G
        Ga = target.factors[i_Ga + 1]

        if k(target) == 2:
            out.append(
                Graph(0, Trace(M(), X_A[:i_Ga], wtG(like=Ga), X_A[i_Ga + 1 :]))
                * deepcopy(f)
            )
            out.append(
                Graph(0, Trace(M(), X_A[:i_Ga], M(like=Ga), X_A[i_Ga + 1 :]))
                * deepcopy(f)
            )
        else:
            out.append(Trace(M(), X_A) * f)

        out.append(MGG(X_A, f, alpha))
        out.append(wtGMG(X_A, f, alpha))

        # Self-consistent mechanism is from the final DX term
        out.extend(DX(X_A, f, alpha)[:-1])
        out.append(
            Trace(M(), E(alpha), wtG(like=Ga), deepcopy(X_A[i_Ga + 1 :]))
            * Trace(G(), deepcopy(X_A[:i_Ga]), G(like=Ga), E(alpha))
            * deepcopy(f)
        )

        if not solve and self_consistent_term:
            out.append(
                Trace(M(), E(alpha), M(like=Ga), deepcopy(X_A[i_Ga + 1 :]))
                * Trace(G(), deepcopy(X_A[:i_Ga]), G(like=Ga), E(alpha))
                * deepcopy(f)
            )

        out.extend(Df(X_A, f, alpha))

        return out

    Ga = target.factors[last_G_index(target)]
    out = h(deepcopy(target.factors[final_deterministic_start:]))
    if solve:
        out += [
            ImF(beta, gamma, 1 if Ga.conjugated else 0)
            * Trace(
                M(),
                E(beta),
                M(like=Ga),
                deepcopy(target.factors[final_deterministic_start:]),
            )
            * deepcopy(x1)
            for x1 in h(E(gamma))
        ]
    if conjugated:
        return [conjugate(x1) for x1 in out]
    else:
        return out
