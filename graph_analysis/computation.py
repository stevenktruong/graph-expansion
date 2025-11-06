from graph_expansion import *

# Leading term functions


def compute_leading_terms(
    x0: Graph,
    o: int,
    verbose=False,
    count_error_terms=False,
    max_terms: Optional[int] = None,
):
    stack: list[Graph] = [x0]
    leading_terms: list[Graph] = []
    small_terms: list[Graph] = []
    n_expansions = 0
    while stack and (not max_terms or len(leading_terms) <= max_terms):
        x = stack.pop()

        # Stop if the graph is deterministic
        if x.is_deterministic():
            if order(x) == o:
                leading_terms.append(x)
            continue

        # Stop if the graph is small enough
        if order(x) > o:
            if count_error_terms:
                small_terms.append(x)
            else:
                del x
            continue

        n_expansions += 1
        stack.extend(expand(x, with_cross_derivative_terms=True))

    if verbose:
        render(x0)
        print(f"# order {o} deterministics: ", len(leading_terms))
        if count_error_terms:
            print("# smaller graphs:         ", len(small_terms))
        print("# expansions:             ", n_expansions)

    return leading_terms


def order(x: Graph) -> int:
    out: int = 0

    indices: set[Symbol] = set()
    for t in x.g_loops:
        for f in t:
            if isinstance(f, E) and isinstance(f.i, b):
                indices.add(f.i)
    for c in x.coefficients:
        for i in c.indices:
            if isinstance(i, b):
                indices.add(i)

    out += sum([len(t) // 2 for t in x.light_weights])
    out += sum([len(t) // 2 - 1 for t in x.g_loops])
    out += sum([len(t) // 2 - 1 for t in x.deterministics])
    out += len([c for c in x.coefficients])

    return out - len(indices)


def expand(x: Graph, with_cross_derivative_terms=False) -> list[Graph]:
    if x.light_weights:
        return expand_light_weight(x, trace_index=0)
    else:
        return expand_G_loop(
            x,
            trace_index=0,
            with_cross_derivative_terms=with_cross_derivative_terms,
        )


# Loop operations


def cutL(t: Trace, k: int, l: int, i: Symbol) -> Trace:
    assert k <= l
    assert 1 <= k and k <= len(t) // 2
    assert 1 <= l and l <= len(t) // 2
    left_index = (k - 1) * 2
    right_index = (l - 1) * 2
    G_k = t[left_index]
    G_l = t[right_index]

    assert isinstance(G_k, G | wtG) and isinstance(G_l, G | wtG)
    sigma_k = G_k.charge
    sigma_l = G_l.charge

    return Trace(G(sigma_l), t[right_index + 1 :], t[:left_index], G(sigma_k), E(i))


def cutR(t: Trace, k: int, l: int, i: Symbol) -> Trace:
    assert k <= l
    assert 1 <= k and k <= len(t) // 2
    assert 1 <= l and l <= len(t) // 2
    left_index = (k - 1) * 2
    right_index = (l - 1) * 2
    G_k = t[left_index]
    G_l = t[right_index]

    assert isinstance(G_k, G | wtG) and isinstance(G_l, G | wtG)
    sigma_k = G_k.charge
    sigma_l = G_l.charge

    return (
        Trace(t[left_index:right_index], G(sigma_l), E(i))
        if k != l
        else Trace(t[left_index:right_index], wtG(sigma_l), E(i))
    )


def cross(t1: Trace, t2: Trace, k1: int, k2: int, i: Symbol, j: Symbol) -> Trace:
    assert 1 <= k1 and k1 <= len(t1) // 2
    assert 1 <= k2 and k2 <= len(t2) // 2
    index1 = (k1 - 1) * 2
    index2 = (k2 - 1) * 2
    G_k1 = t1[index1]
    G_k2 = t2[index2]

    assert isinstance(G_k1, G | wtG) and isinstance(G_k2, G | wtG)
    sigma_k1 = G_k1.charge
    sigma_k2 = G_k2.charge

    return Trace(
        G(sigma_k1),
        t1[index1 + 1 :],
        t1[:index1],
        G(sigma_k1),
        E(i),
        G(sigma_k2),
        t2[index2 + 1 :],
        t2[:index2],
        G(sigma_k2),
        E(j),
    )


def drift_terms(x: Graph) -> list[Graph]:
    out: list[Graph] = []

    coefficients: list[Coefficient] = list(x.coefficients)
    non_deterministics: list[Trace] = list(x.light_weights + x.g_loops)
    current_max_b_index = largest_b_index(x)
    b_1 = b(current_max_b_index + 1)
    b_2 = b(current_max_b_index + 2)

    # Cut terms
    for i, t in enumerate(non_deterministics):
        remaining_traces = non_deterministics[:i] + non_deterministics[i + 1 :]
        n = len(t) // 2
        for k in range(1, n + 1):
            for l in range(k, n + 1):
                out.append(
                    Graph(
                        S(b_1, b_2),
                        cutL(t, k, l, b_1),
                        cutR(t, k, l, b_2),
                        coefficients,
                        remaining_traces,
                    )
                )

    # Cross terms
    for i, t1 in enumerate(non_deterministics):
        for j, t2 in enumerate(non_deterministics[i + 1 :]):
            j = i + 1 + j
            remaining_traces = [
                t for k, t in enumerate(non_deterministics) if k != i and k != j
            ]
            n1 = len(t1) // 2
            n2 = len(t2) // 2
            for k1 in range(1, n1 + 1):
                for k2 in range(1, n2 + 1):
                    out.append(
                        Graph(
                            S(b_1, b_2),
                            cross(t1, t2, k1, k2, b_1, b_2),
                            coefficients,
                            remaining_traces,
                        )
                    )

    return out
