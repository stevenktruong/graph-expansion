from collections import defaultdict

from graph_expansion import *


def group_by_pattern(pre: list[Graph], signed=True, verbose=False):
    first_pattern_seen: list[tuple[tuple[int, ...], ...]] = []
    pattern_to_graph: dict[tuple[tuple[int, ...], ...], list[Graph]] = defaultdict(list)
    for x in pre:
        representative_pattern = m_loop_pattern(x, signed)
        for pattern in first_pattern_seen:
            if patterns_are_same(representative_pattern, pattern):
                # if patterns_are_same_or_flipped(representative_pattern, pattern):
                representative_pattern = pattern
                break
        else:
            first_pattern_seen.append(representative_pattern)
        pattern_to_graph[representative_pattern].append(x)

    for v in pattern_to_graph.values():
        v.sort(key=lambda x: len(x.deterministics), reverse=True)

    if verbose:
        for i, (k, v) in enumerate(pattern_to_graph.items()):
            print(f"{i:>2} {len(v):>3} {', '.join(str(l) for l in k)}")

    return pattern_to_graph


def to_python(x: Graph) -> str:
    out = ""
    out += "Graph(\n"
    for c in x.coefficients:
        assert isinstance(c.i, NumberedSymbol)
        assert isinstance(c.j, NumberedSymbol)
        if isinstance(c, ChargedCoefficient):
            out += f"  {type(c).__name__}({c.charges[0]}, {c.charges[1]}, {type(c.i).__name__}({c.i.i}), {type(c.j).__name__}({c.j.i})), \n"
        elif isinstance(c, Coefficient):
            out += f"  {type(c).__name__}({type(c.i).__name__}({c.i.i}), {type(c.j).__name__}({c.j.i})), \n"

    for t in x.traces:
        out += "  Trace(\n"
        for f in t:
            if isinstance(f, E):
                assert isinstance(f.i, NumberedSymbol)
                out += f"    {type(f).__name__}({type(f.i).__name__}({f.i.i})), \n"
            else:
                out += f"    {type(f).__name__}({f.charge}), \n"
        out += "  ),\n"

    out += ")"

    return out


# Helpers


def m_loop_pattern(x: Graph, signed=True) -> tuple[tuple[int, ...], ...]:
    if not x.is_deterministic():
        raise ValueError(f"Called {__name__} on a non-deterministic graph")

    # E.g., <MEM^*EME> -> [0. 1, 0]
    pattern: list[tuple[int, ...]] = []
    for c in x.coefficients:
        if isinstance(c, ThetacalM):
            pattern.append(
                tuple(
                    int(charge == Charge.Minus) if signed else 0 for charge in c.charges
                )
            )
    for t in x.deterministics:
        # Ignore F_0 and F_1, i.e., <MEME> graph
        if len(t) == 4:
            continue

        next: list[int] = []
        for f in t:
            if not isinstance(f, M):
                continue
            next.append(int(f.charge == Charge.Minus) if signed else 0)

        # E.g., [0, 0, 1] -> [0, 1, 0]
        if len(next) % 2 == 1:
            while next[0] != next[-1]:
                next = next[1:] + next[:1]
        else:
            # E.g., [1, 0, 1, 0] -> [0, 1, 0, 1]
            if min(next) != max(next):
                while next[0] != 0 or next[-1] != 1:
                    next = next[1:] + next[:1]

        pattern.append(tuple(next))
    pattern.sort(key=lambda f: (len(f), sum(f)))
    return tuple[tuple[int, ...], ...](pattern)


def flip(l: tuple[int, ...]):
    return tuple([1 - x for x in l])


# E.g., (0, 1, 0) is the same as (1, 0, 0)
def loops_are_same(l1: tuple[int, ...], l2: tuple[int, ...]):
    if len(l1) != len(l2):
        return False
    for i in range(len(l1)):
        if l1 == l2[i:] + l2[:i]:
            return True
    return False


def patterns_are_same(p1: tuple[tuple[int, ...], ...], p2: tuple[tuple[int, ...], ...]):
    if len(p1) != len(p2):
        return False
    remaining_loops = list(p2)
    for l1 in p1:
        for l2 in remaining_loops:
            if loops_are_same(l1, l2):
                remaining_loops.remove(l2)
                break
    return len(remaining_loops) == 0


# E.g., ((0, 0), (0, 1, 0)) is the conjugate of ((1, 1), (1, 0, 1))
def patterns_are_same_or_flipped(
    p1: tuple[tuple[int, ...], ...], p2: tuple[tuple[int, ...], ...]
):
    flipped_p2 = tuple(flip(l) for l in p2)
    return patterns_are_same(p1, p2) or patterns_are_same(p1, flipped_p2)
