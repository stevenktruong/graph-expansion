from graph_expansion import *


def matrix_multiplication(pre: list[Graph]) -> list[Graph]:
    # Perform some simple matrix multiplication
    # - Simplify to S\Theta, S\Theta\MS, etc.
    out: list[Graph] = []
    for x in pre:
        old_coefficients = x.coefficients
        old_traces = x.traces

        coefficients: list[Coefficient] = list(old_coefficients)
        removed_indices: set[Symbol] = set()

        finished = False
        while True:
            if finished:
                break
            finished = True

            next_coefficients: list[Coefficient] = []
            coefficients_to_remove: list[int] = []
            for i, c in enumerate(coefficients):
                if i in coefficients_to_remove:
                    continue
                for j, c1 in enumerate(coefficients[i + 1 :]):
                    j = i + 1 + j
                    if j in coefficients_to_remove:
                        continue
                    if not set(c.indices) & set(c1.indices):
                        continue
                    assert len(set(c.indices) & set(c1.indices)) == 1

                    finished = False
                    product = single_matrix_multiplication(c, c1)
                    # except:
                    #     render(x)
                    #     render(c)
                    #     render(c1)

                    if not product:
                        continue

                    next_coefficients.append(product)
                    coefficients_to_remove.extend([i, j])
                    removed_indices.union(set(c.indices) & set(c1.indices))

                    break

            next_coefficients += [
                c for i, c in enumerate(coefficients) if i not in coefficients_to_remove
            ]
            coefficients = next_coefficients

        assert all([isinstance(i, b) for i in removed_indices])

        # Double-check that each b_i only appears at most twice
        for t in x.traces:
            for f in enumerate(t):
                if isinstance(f, E):
                    assert f.i not in removed_indices

        # Replace internal a_i's with delta_{a_ib_i}'s
        traces: list[Trace] = []
        next_b_index = largest_b_index(x) + 1
        for i, t in enumerate(old_traces):
            pre_trace: list[MatrixFactor] = []
            for f in t:
                if isinstance(f, E) and isinstance(f.i, a):
                    coefficients.append(I(f.i, b(next_b_index)))
                    pre_trace.append(E(b(next_b_index)))
                    next_b_index += 1
                else:
                    pre_trace.append(f)
            traces.append(Trace(pre_trace))

        out.append(
            Graph(
                [c if not isinstance(c.j, a) else c.transpose() for c in coefficients],
                traces,
            )
        )
    return out


def vertical_cancel(x: Graph) -> Graph:
    coefficients: list[Coefficient] = []
    for c in x.coefficients:
        if isinstance(c, ThetacalMS):
            coefficients.append(Theta(c.charges[0], c.charges[1], c.i, c.j))
        elif isinstance(c, SThetacalMS):
            coefficients.append(STheta(c.charges[0], c.charges[1], c.i, c.j))
        else:
            coefficients.append(c)
    return Graph(coefficients, x.traces)


def single_matrix_multiplication(
    c1: Coefficient, c2: Coefficient
) -> Optional[Coefficient]:
    if not set(c1.indices) & set(c2.indices):
        return None

    for d1, d2 in [(c1, c2), (c2, c1)]:
        if isinstance(d1, calM) and isinstance(d2, S):
            if d1.i in set(d2.indices):
                d1 = d1.transpose()
            if d1.j != d2.i:
                d2 = d2.transpose()
            return calMS(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, Theta) and isinstance(d2, S):
            assert d1.i in set(d2.indices)
            if d2.j != d1.i:
                d2 = d2.transpose()
            return STheta(d1.charges[0], d1.charges[1], d2.i, d1.j)

        if isinstance(d1, Theta) and isinstance(d2, calM):
            assert d1.j in set(d2.indices)
            if d1.j != d2.i:
                d2 = d2.transpose()
            assert d1.charges == d2.charges
            return ThetacalM(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, Theta) and isinstance(d2, calMS):
            if d1.i == d2.i:
                d2 = d2.transpose()
                assert d1.charges == d2.charges
                return ThetacalMS(d1.charges[0], d1.charges[1], d2.i, d1.j)
            elif d1.j == d2.j:
                d2 = d2.transpose()
                assert d1.charges == d2.charges
                return ThetacalMS(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, STheta) and isinstance(d2, calM):
            if d1.j in set(d2.indices):
                d1 = d1.transpose()
            if d2.j != d1.i:
                d2 = d2.transpose()
            assert d1.charges == d2.charges
            return ThetacalMS(d1.charges[0], d1.charges[1], d2.i, d1.j)

        if isinstance(d1, STheta) and isinstance(d2, Theta):
            assert d2.i in set(d1.indices)
            if d1.i in set(d2.indices):
                d1 = d1.transpose()
            assert d1.charges == d2.charges
            return SThetaTheta(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, ThetacalM) and isinstance(d2, S):
            if d1.i in set(d2.indices):
                d1 = d1.transpose()
            if d1.j != d2.i:
                d2 = d2.transpose()
            return ThetacalMS(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, ThetacalM) and isinstance(d2, STheta):
            if d1.i in set(d2.indices):
                d1 = d1.transpose()
            if d1.j != d2.i:
                d2 = d2.transpose()
            assert d1.charges == d2.charges
            return ThetacalMSTheta(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, ThetacalMS) and isinstance(d2, S):
            assert d1.i in set(d2.indices)
            if d2.i in set(d1.indices):
                d2 = d2.transpose()
            return SThetacalMS(d1.charges[0], d1.charges[1], d2.i, d1.j)

        if isinstance(d1, STheta) and isinstance(d2, calMS):
            assert d2.i in set(d1.indices)
            if d1.j != d2.i:
                d1 = d1.transpose()
            return SThetacalMS(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, SThetacalMS) and isinstance(d2, Theta):
            assert d2.i in set(d1.indices)
            if d1.i in set(d2.indices):
                d1 = d1.transpose()
            assert d1.charges == d2.charges
            return SThetacalMSTheta(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, SThetacalMS) and isinstance(d2, ThetacalMS):
            if d1.i in set(d2.indices):
                d1 = d1.transpose()
            assert d1.j == d2.i and d1.charges == d2.charges
            return SThetacalMSThetacalMS(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, ThetacalM) and isinstance(d2, SThetacalMS):
            if d1.i in set(d2.indices):
                d1 = d1.transpose()
            if d1.j != d2.i:
                d2 = d2.transpose()
            assert d1.charges == d2.charges
            return ThetacalMSThetacalMS(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, ThetacalMSTheta) and isinstance(d2, S):
            assert d1.i in set(d2.indices)
            if d2.j != d1.i:
                d2 = d2.transpose()
            return SThetacalMSTheta(d1.charges[0], d1.charges[1], d2.i, d1.j)

        if isinstance(d1, STheta) and isinstance(d2, ThetacalMS):
            assert d2.i in set(d1.indices)
            if d1.j != d2.i:
                d1 = d1.transpose()
            assert d1.charges == d2.charges
            return SThetaThetaMS(d1.charges[0], d1.charges[1], d1.i, d2.j)

        if isinstance(d1, ThetacalM) and isinstance(d2, ThetacalMS):
            assert d2.j in set(d1.indices)
            if d2.j != d1.i:
                d1 = d1.transpose()
            assert d1.charges == d2.charges
            return ThetacalMSThetacalM(d1.charges[0], d1.charges[1], d2.i, d1.j)

        # Charges don't have to align, but this means that calMS should pair with a Theta somewhere
        # if isinstance(d1, ThetacalMS) and isinstance(d2, calMS):
        #     assert d1.charges == d2.charges
        #     if d1.i == d2.j:
        #         return SThetacalMSTheta(d1.charges[0], d1.charges[1], d2.i, d1.j)
        #     elif d1.j == d2.i:
        #         return SThetacalMSTheta(d1.charges[0], d1.charges[1], d1.i, d2.j)
