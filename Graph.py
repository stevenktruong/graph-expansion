from copy import deepcopy
from MatrixFactor import Alpha, MatrixFactor, G, wtG
from Texable import Texable, render, tex


class Trace(Texable):
    factors: list[MatrixFactor]

    def __init__(self, *args):
        self.factors = []
        for input in args:
            if isinstance(input, MatrixFactor):
                self.factors.append(input)
            elif (
                isinstance(input, list) and input and isinstance(input[0], MatrixFactor)
            ):
                self.factors.extend(input)

        if not self.is_deterministic():
            while self.factors[0].is_deterministic():
                self.factors = self.factors[1:] + self.factors[:1]

    def __mul__(self, other):
        if isinstance(other, Trace):
            return Graph(0, deepcopy(self), deepcopy(other))
        elif isinstance(other, Graph):
            return Graph(other.n.exponent, deepcopy(self), deepcopy(other.traces))
        elif isinstance(other, N):
            return Graph(other.exponent, deepcopy(self))
        elif isinstance(other, Coefficient):
            return Graph(0, deepcopy(other), deepcopy(self))
        else:
            raise TypeError("Tried to multiply Trace by an invalid object")

    __rmul__ = __mul__

    def __tex__(self):
        return f"\\avg{{{' '.join([tex(f) for f in self.factors])}}}"

    def __copy__(self):
        return Trace(deepcopy(self.factors))

    def is_deterministic(self):
        return all([f.is_deterministic() for f in self.factors])


class Coefficient(Texable):
    conjugated: bool

    def __init__(self, conjugated: bool = False):
        self.conjugated = conjugated

    pass


class ImF(Coefficient):
    a: int
    alpha: Alpha
    beta: Alpha

    def __init__(self, alpha, beta, a: int = 0, conjugated: bool = False):
        super().__init__(conjugated)
        self.a = a
        self.alpha = Alpha(alpha)
        self.beta = Alpha(beta)

    def __tex__(self):
        unconjugated_string = (
            f"\\p{{I - F_{self.a}}}^{{-1}}_{{{tex(self.alpha)}{tex(self.beta)}}}"
        )
        if self.conjugated:
            return f"\\conj{{{unconjugated_string}}}"
        else:
            return unconjugated_string

    def __copy__(self):
        return ImF(deepcopy(self.alpha), deepcopy(self.beta), self.a, self.conjugated)


class N(Texable):
    exponent: int

    def __init__(self, exponent: int = 0):
        if exponent > 0:
            raise TypeError("exponent must be non-positive")
        self.exponent = exponent

    def __tex__(self):
        if self.exponent == 0:
            return ""
        elif self.exponent == -1:
            return "\\frac{1}{N}"
        elif self.exponent < -9:
            return f"\\frac{{1}}{{N^{{{-self.exponent}}}}}"
        else:
            return f"\\frac{{1}}{{N^{-self.exponent}}}"


class Graph(Texable):
    n: N
    coefficients: list[Coefficient]
    traces: list[Trace]
    deterministics: list[Trace]
    light_weights: list[Trace]
    g_loops: list[Trace]
    others: list[Trace]

    def __init__(self, p: int = 0, *args):
        self.n = N(p)
        self.coefficients = []
        self.traces = []
        self.deterministics = []
        self.light_weights = []
        self.g_loops = []
        self.others = []

        for x in args:
            if isinstance(x, Trace):
                self.traces.append(x)
            elif isinstance(x, list) and x and isinstance(x[0], Trace):
                self.traces.extend(x)
            elif isinstance(x, Coefficient):
                self.coefficients.append(x)
            elif isinstance(x, list) and x and isinstance(x[0], Coefficient):
                self.coefficients.extend(x)

        for t in self.traces:
            n_wtG = len([f for f in t.factors if isinstance(f, wtG)])
            n_G = len([f for f in t.factors if isinstance(f, G)])

            if n_wtG == 0 and n_G == 0:
                self.deterministics.append(t)
            elif n_wtG == 0 and n_G > 1:
                self.g_loops.append(t)
            elif n_wtG == 1 and n_G == 0:
                self.light_weights.append(t)
            else:
                self.others.append(t)
                # render(t)
                # raise TypeError("Trace is not deterministic, light-weight, or G-loop")

    def __mul__(self, other):
        if isinstance(other, Trace):
            return Graph(
                self.n.exponent,
                deepcopy(self.traces),
                deepcopy(self.coefficients),
                deepcopy(other),
            )
        elif isinstance(other, Graph):
            return Graph(
                self.n.exponent,
                deepcopy(self.traces),
                deepcopy(self.coefficients),
                deepcopy(other.traces),
                deepcopy(other.coefficients),
            )
        elif isinstance(other, N):
            return Graph(
                self.n.exponent + other.exponent,
                deepcopy(self.traces),
                deepcopy(self.coefficients),
            )
        elif isinstance(other, Coefficient):
            return Graph(
                self.n.exponent,
                deepcopy(self.traces),
                deepcopy(self.coefficients),
                deepcopy(other),
            )
        else:
            raise TypeError("Tried to multiply Graph by an invalid object")

    __rmul__ = __mul__

    def __tex__(self):
        deterministic_string = f"{tex(self.n)}{''.join([tex(t) for t in self.coefficients])}{''.join([tex(t) for t in self.deterministics])}"
        non_deterministic_string = f"\\E{''.join([tex(t) for t in self.light_weights])}{''.join([tex(t) for t in self.g_loops])}{''.join([tex(t) for t in self.others])}"
        if len(self.deterministics) == len(self.traces):
            return deterministic_string
        else:
            return deterministic_string + non_deterministic_string


def p(x: Graph) -> int:
    return -x.n.exponent


def q(x: Graph) -> int:
    return len(x.light_weights)


def r(x: Graph) -> int:
    return len(x.g_loops)


def k(t: Trace) -> int:
    return len([f for f in t.factors if isinstance(f, G)])
