from copy import deepcopy
from Term import *
from Factor import *

# TODO: Track number of G, wtG, etc. in Term


def transpose(product: list[MatrixFactor]):
    product.reverse()
    return [f.transpose() for f in product]


def rotate(t: Trace):
    t.product = t.product[1:] + t.product[:1]


def standardize(t: Term):
    for light_weight in t.light_weights:
        while not isinstance(light_weight.product[0], wtG):
            rotate(light_weight)

    for g_loop in t.g_loops:
        if any(
            [
                isinstance(g_loop.product[i], G)
                and isinstance(g_loop.product[i + 2], G)
                for i in range(len(g_loop.product) - 2)
            ]
        ):
            while not (
                isinstance(g_loop.product[0], G) and isinstance(g_loop.product[-2], G)
            ):
                rotate(g_loop)
        else:
            while not isinstance(g_loop.product[-2], G):
                rotate(g_loop)


def ND(t: Trace):
    out: list[list[MatrixFactor]] = []
    for k, f in enumerate(t.product):
        if isinstance(f, G) or isinstance(f, wtG):
            out.extend(
                [
                    transpose(
                        [G()]
                        + deepcopy(t.product[k + 1 :])
                        + deepcopy(t.product[:k])
                        + [G()]
                    ),
                    [G()]
                    + deepcopy(t.product[k + 1 :])
                    + deepcopy(t.product[:k])
                    + [G()],
                ]
            )
    return out


def expand_light_weight(x: Term):
    out: list[Term] = []

    target = x.light_weights[0]
    alpha = (
        max(
            [f.alpha.i for t in x.traces for f in t.product if isinstance(f, E)],
            default=0,
        )
        + 1
    )

    # <MGG f(H)> term
    out.append(
        N(x.n_exponent - 1)
        * Trace(M(), E(alpha), G(), E(alpha), G(), target.product[1:])
        * x.light_weights[1:]
        * x.g_loops
    )

    # <\G> <MG f(H)> term
    out.extend(
        [
            N(x.n_exponent)
            * Trace(wtG(), E(alpha))
            * Trace(M(), E(alpha), wtG(), target.product[1:])
            * x.light_weights[1:]
            * x.g_loops,
            # Self-consistent term
            N(x.n_exponent)
            * Trace(wtG(), E(alpha))
            * Trace(M(), E(alpha), M(), target.product[1:])
            * x.light_weights[1:]
            * x.g_loops,
        ]
    )

    out.extend(
        [
            N(x.n_exponent - 2)
            * Trace(M(), E(alpha), nderivative, E(alpha), G(), target.product[1:])
            * x.light_weights[1:][:k]
            * x.light_weights[1:][k + 1 :]
            * x.g_loops
            for k, light_weight in enumerate(x.light_weights[1:])
            for nderivative in ND(light_weight)
        ]
    )

    out.extend(
        [
            N(x.n_exponent - 2)
            * Trace(M(), E(alpha), nderivative, E(alpha), G(), target.product[1:])
            * x.light_weights[1:]
            * x.g_loops[:k]
            * x.g_loops[k + 1 :]
            for k, g_loop in enumerate(x.g_loops)
            for nderivative in ND(g_loop)
        ]
    )

    return out, alpha


def expand_g_loop(x: Term):
    out: list[Term] = []

    target = x.g_loops[0]
    target_i = -1
    for k, y in enumerate(target.product):
        if isinstance(y, G):
            target_i = k
            break
    alpha = (
        max(
            [f.alpha.i for t in x.traces for f in t.product if isinstance(f, E)],
            default=0,
        )
        + 1
    )

    # G = \G + M
    if len([y for y in target.product if isinstance(y, G)]) != 2:
        out.append(
            N(x.n_exponent)
            * Trace(target.product[:target_i], M(), target.product[target_i + 1 :])
            * x.light_weights
            * x.g_loops[1:]
        )
    else:
        i = -1
        for k, y in enumerate(target.product[target_i + 1 :]):
            if isinstance(y, G):
                i = k + target_i + 1
                break

        out.extend(
            [
                N(x.n_exponent)
                * Trace(
                    target.product[:target_i],
                    M(),
                    target.product[target_i + 1 : i],
                    wtG(),
                    target.product[i + 1 :],
                )
                * x.light_weights
                * x.g_loops[1:],
                N(x.n_exponent)
                * Trace(
                    target.product[:target_i],
                    M(),
                    target.product[target_i + 1 : i],
                    M(),
                    target.product[i + 1 :],
                )
                * x.light_weights
                * x.g_loops[1:],
            ]
        )

    # <MGG f(H)> term
    out.append(
        N(x.n_exponent - 1)
        * Trace(
            target.product[:target_i],
            M(),
            E(alpha),
            G(),
            E(alpha),
            G(),
            target.product[target_i + 1 :],
        )
        * x.light_weights
        * x.g_loops[1:]
    )

    # <\G> <MG f(H)> term
    out.append(
        N(x.n_exponent)
        * Trace(wtG(), E(alpha))
        * Trace(
            target.product[:target_i],
            M(),
            E(alpha),
            G(),
            target.product[target_i + 1 :],
        )
        * x.light_weights
        * x.g_loops[1:]
    )

    # Derivative on self
    out.extend(
        [
            N(x.n_exponent - 1)
            * Trace(
                target.product[:target_i],
                M(),
                E(alpha),
                transpose(deepcopy(target.product[target_i + 1 :][: k + 1])),
                G(),
                E(alpha),
                deepcopy(
                    (target.product[:target_i] + target.product[target_i + 1 :])[k:]
                ),
            )
            * x.light_weights
            * x.g_loops[1:]
            for k, f in enumerate(target.product[target_i + 1 :])
            if isinstance(f, G)
        ]
    )

    # Other derivative terms on self
    out.extend(
        [
            N(x.n_exponent)
            * Trace(
                target.product[:target_i],
                M(),
                E(alpha),
                deepcopy((target.product[target_i + 1 :])[k:]),
            )
            * Trace(
                deepcopy(
                    (target.product[:target_i] + target.product[target_i + 1 :])[
                        : k + 1
                    ]
                ),
                E(alpha),
                G(),
            )
            * x.light_weights
            * x.g_loops[1:]
            for k, f in enumerate(target.product[target_i + 1 :])
            if isinstance(f, G)
        ]
    )

    # Derivative on light weight
    out.extend(
        [
            N(x.n_exponent - 2)
            * Trace(
                M(),
                E(alpha),
                target.product[:target_i],
                nderivative,
                E(alpha),
                G(),
                target.product[target_i + 1 :],
            )
            * x.light_weights[1:][:k]
            * x.light_weights[1:][k + 1 :]
            * x.g_loops[1:]
            for k, light_weight in enumerate(x.light_weights)
            for nderivative in ND(light_weight)
        ]
    )

    # Derivative on G-loop
    out.extend(
        [
            N(x.n_exponent - 2)
            * Trace(
                M(),
                E(alpha),
                target.product[:target_i],
                nderivative,
                E(alpha),
                G(),
                target.product[target_i + 1 :],
            )
            * x.g_loops[1:][:k]
            * x.g_loops[1:][k + 1 :]
            * x.light_weights
            for k, g_loop in enumerate(x.g_loops[1:])
            for nderivative in ND(g_loop)
        ]
    )

    return out, alpha


def expand(t: Term):
    if t.light_weights:
        out = expand_light_weight(t)
    else:
        out = expand_g_loop(t)

    # for x in out[0]:
    #     standardize(x)

    return out
