from typing import Callable, Optional

from .Coefficient import S, STheta, Theta, calM
from .Graph import Graph, Trace, last_G_index
from .helpers import k
from .MatrixFactor import E, G, M, MatrixFactor, wtG
from .Symbol import b


def expand_G_loop(
    x: Graph,
    trace_predicate: Optional[Callable[[Trace], bool]] = None,
    trace_index: Optional[int] = 0,
    G_predicate: Optional[Callable[[G], bool]] = None,
    G_index: int = 0,
    solve: bool = True,
    with_cross_derivative_terms=True,
) -> list[Graph]:
    if trace_predicate:
        for j, t in enumerate(x.g_loops):
            if trace_predicate(t):
                trace_index = j
                break
        else:
            if trace_index is None:
                raise LookupError(
                    "Could not find G-loop from trace_predicate with no fallback"
                )
    assert trace_index is not None
    target_trace = x.g_loops[trace_index]

    if G_predicate:
        for j, f in enumerate(target_trace):
            if isinstance(f, G) and G_predicate(f):
                G_index = j
                break
        else:
            if G_index is None:
                raise LookupError("Could not find G from G_predicate with no fallback")
    assert G_index is not None

    # Rewrite the loop so that the target G is the first matrix in the trace
    for _ in range(G_index):
        target_trace.cycle()

    for j, f in enumerate(target_trace[1:]):
        if isinstance(f, G):
            next_G_index = j + 1
            break
    else:
        raise TypeError("x needs to have more than 1 G to be expanded")

    G_1 = target_trace[0]
    B_1: list[MatrixFactor] = target_trace[1:next_G_index]
    assert isinstance(G_1, G) and all([not isinstance(f, G) for f in B_1])

    last_G_i = last_G_index(target_trace)
    G_n = target_trace[last_G_i]
    B_n: list[MatrixFactor] = target_trace[last_G_i + 1 :]
    assert isinstance(G_n, G) and all([not isinstance(f, G) for f in B_n])

    # Begin main computations
    sigma_1 = G_1.charge
    sigma_n = G_n.charge

    current_max_b_index = largest_b_index(x)
    b_1 = b(current_max_b_index + 1)
    b_2 = b(current_max_b_index + 2)
    b_3 = b(current_max_b_index + 3)
    b_4 = b(current_max_b_index + 4)

    # x = coefficients * target_trace * f
    coefficients = x.coefficients
    f_G = (
        x.deterministics
        + x.light_weights
        + x.g_loops[:trace_index]
        + x.g_loops[trace_index + 1 :]
    )

    def _G(
        A: MatrixFactor | list[MatrixFactor],
        with_self_consistent_term=False,
        with_cross_derivative_terms=True,
    ) -> list[Graph]:
        out: list[Graph] = []

        # From G = \G + M, get the M term
        if k(target_trace) == 2:
            out.append(
                Graph(
                    coefficients,
                    Trace(
                        M(sigma_1),
                        B_1,
                        target_trace[next_G_index:last_G_i],
                        M(sigma_n),
                        A,
                    ),
                    f_G,
                )
            )
            out.append(
                Graph(
                    coefficients,
                    Trace(
                        M(sigma_1),
                        B_1,
                        target_trace[next_G_index:last_G_i],
                        wtG(sigma_n),
                        A,
                    ),
                    f_G,
                )
            )
        else:
            out.append(
                Graph(
                    coefficients,
                    Trace(
                        M(sigma_1),
                        B_1,
                        target_trace[next_G_index:last_G_i],
                        G(sigma_n),
                        A,
                    ),
                    f_G,
                )
            )

        # Light-weight term 1
        out.append(
            Graph(
                coefficients,
                Trace(
                    M(sigma_1),
                    E(b_1),
                    target_trace[:last_G_i],
                    G(sigma_n),
                    A,
                ),
                S(b_1, b_2),
                Trace(wtG(sigma_1), E(b_2)),
                f_G,
            )
        )

        # Self-consistent term
        if with_self_consistent_term:
            out.append(
                Graph(
                    coefficients,
                    Trace(
                        M(sigma_1),
                        E(b_1),
                        M(sigma_n),
                        A,
                    ),
                    S(b_1, b_2),
                    Trace(target_trace[:last_G_i], G(sigma_n), E(b_2)),
                    f_G,
                )
            )

        # Self-derivative terms
        for i, f in enumerate(target_trace[1:last_G_i]):
            if not isinstance(f, G):
                continue
            j = i + 1
            sigma_j = f.charge
            out.append(
                Graph(
                    coefficients,
                    Trace(
                        M(sigma_1),
                        E(b_1),
                        target_trace[j:last_G_i],
                        G(sigma_n),
                        A,
                    ),
                    S(b_1, b_2),
                    Trace(target_trace[:j], G(sigma_j), E(b_2)),
                    f_G,
                )
            )

        # Light-weight term 2
        out.append(
            Graph(
                coefficients,
                Trace(
                    M(sigma_1),
                    E(b_1),
                    wtG(sigma_n),
                    A,
                ),
                S(b_1, b_2),
                Trace(target_trace[:last_G_i], G(sigma_n), E(b_2)),
                f_G,
            )
        )

        if with_cross_derivative_terms:
            other_traces = (
                x.g_loops[:trace_index] + x.g_loops[trace_index + 1 :] + x.light_weights
            )
            for i, other_trace in enumerate(other_traces):
                remaining_traces = other_traces[:i] + other_traces[i + 1 :]
                for j, f in enumerate(other_trace):
                    if not isinstance(f, G | wtG):
                        continue
                    out.append(
                        Graph(
                            coefficients,
                            S(b_1, b_2),
                            Trace(
                                M(sigma_1),
                                E(b_1),
                                G(like=other_trace[j]),
                                other_trace[j + 1 :],
                                other_trace[:j],
                                G(like=other_trace[j]),
                                E(b_2),
                                target_trace[:last_G_i],
                                G(sigma_n),
                                A,
                            ),
                            x.deterministics,
                            remaining_traces,
                        )
                    )

        return out

    if not solve:
        return _G(
            B_n,
            with_self_consistent_term=True,
            with_cross_derivative_terms=with_cross_derivative_terms,
        )

    # Special case where B_n = E_z
    E_z = B_n[0]
    if len(B_n) == 1 and isinstance(E_z, E):
        z = E_z.i
        return [
            Theta(sigma_n, sigma_1, z, b_3) * g
            for g in _G(E(b_3), with_cross_derivative_terms=with_cross_derivative_terms)
        ]
    else:
        return _G(B_n, with_cross_derivative_terms=with_cross_derivative_terms) + [
            Trace(M(sigma_n), B_n, M(sigma_1), E(b_3))
            * STheta(sigma_n, sigma_1, b_3, b_4)
            * g
            for g in _G(E(b_4), with_cross_derivative_terms=with_cross_derivative_terms)
        ]


def expand_light_weight(
    x: Graph,
    trace_predicate: Optional[Callable[[Trace], bool]] = None,
    trace_index: Optional[int] = None,
    solve: bool = True,
) -> list[Graph]:
    if trace_predicate:
        for j, t in enumerate(x.light_weights):
            if trace_predicate(t):
                trace_index = j
                break
        else:
            if trace_index is None:
                raise LookupError(
                    "Could not find light-weight from trace_predicate with no fallback"
                )
    assert trace_index is not None
    target_trace = x.light_weights[trace_index]

    # Rewrite the loop so that the target G is the first matrix in the trace
    while not isinstance(target_trace[0], wtG):
        target_trace.cycle()

    G_1 = target_trace[0]
    B_1: list[MatrixFactor] = target_trace[1:]
    assert isinstance(G_1, wtG) and all([not isinstance(f, wtG) for f in B_1])

    # Begin main computations
    sigma = G_1.charge

    current_max_b_index = largest_b_index(x)
    b_1 = b(current_max_b_index + 1)
    b_2 = b(current_max_b_index + 2)
    b_3 = b(current_max_b_index + 3)
    b_4 = b(current_max_b_index + 4)

    # x = coefficients * target_trace * f
    coefficients = x.coefficients
    f_G = (
        x.deterministics
        + x.light_weights[:trace_index]
        + x.light_weights[trace_index + 1 :]
        + x.g_loops
    )

    def _G(
        A: MatrixFactor | list[MatrixFactor], with_self_consistent_term=False
    ) -> list[Graph]:
        out: list[Graph] = []

        # Self-consistent term
        if with_self_consistent_term:
            out.append(
                Graph(
                    coefficients,
                    Trace(
                        M(sigma),
                        E(b_1),
                        M(sigma),
                        A,
                    ),
                    S(b_1, b_2),
                    Trace(wtG(sigma), E(b_2)),
                    f_G,
                )
            )

        # Light-weight
        out.append(
            Graph(
                coefficients,
                Trace(
                    M(sigma),
                    E(b_1),
                    wtG(sigma),
                    A,
                ),
                S(b_1, b_2),
                Trace(wtG(sigma), E(b_2)),
                f_G,
            )
        )

        other_traces = (
            x.light_weights[:trace_index]
            + x.light_weights[trace_index + 1 :]
            + x.g_loops
        )
        for i, other_trace in enumerate(other_traces):
            remaining_traces = other_traces[:i] + other_traces[i + 1 :]
            for j, f in enumerate(other_trace):
                if not isinstance(f, G | wtG):
                    continue
                out.append(
                    Graph(
                        coefficients,
                        S(b_1, b_2),
                        Trace(
                            M(sigma),
                            E(b_1),
                            G(like=other_trace[j]),
                            other_trace[j + 1 :],
                            other_trace[:j],
                            G(like=other_trace[j]),
                            E(b_2),
                            G(sigma),
                            A,
                        ),
                        x.deterministics,
                        remaining_traces,
                    )
                )

        return out

    if not solve:
        return _G(B_1, with_self_consistent_term=True)

    # Special case where B_1 = E_z
    E_z = B_1[0]
    if len(B_1) == 1 and isinstance(E_z, E):
        z = E_z.i
        return [Theta(sigma, sigma, z, b_3) * g for g in _G(E(b_3))]
    else:
        return _G(B_1) + [
            Trace(M(sigma), B_1, M(sigma), E(b_3)) * STheta(sigma, sigma, b_3, b_4) * g
            for g in _G(E(b_4))
        ]


def largest_b_index(x: Graph) -> int:
    all_b_indices: list[b] = []
    for c in x.coefficients:
        if isinstance(c, S | Theta | STheta | calM):
            for i in c.indices:
                if isinstance(i, b):
                    all_b_indices.append(i)
    for t in x.traces:
        for f in t:
            if isinstance(f, E) and isinstance(f.i, b):
                all_b_indices.append(f.i)
    return max([b.i for b in all_b_indices], default=0)
