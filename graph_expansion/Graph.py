from copy import deepcopy

from .Coefficient import Coefficient, STheta, Theta, calM
from .MatrixFactor import E, G, M, MatrixFactor, wtG
from .Symbol import Symbol, a
from .Texable import Texable, render, tex


class Trace(Texable):
    _factors: tuple[MatrixFactor, ...]

    def __init__(self, *args):
        factors: list[MatrixFactor] = []

        def _parse_inputs(args):
            if isinstance(args, MatrixFactor):
                factors.append(args)
            elif isinstance(args, list | tuple):
                for x in args:
                    _parse_inputs(x)

        _parse_inputs(args)

        self._factors = tuple(factors)

    def __eq__(self, other):
        if not isinstance(other, Trace):
            return False
        for i in range(len(self._factors)):
            if self._factors[i:] + self._factors[:i] == other._factors:
                return True
        return False

    def __mul__(self, other):
        raise NotImplementedError("__mul__ not implemented on Trace")

    __rmul__ = __mul__

    def __len__(self):
        return len(self._factors)

    def __getitem__(self, i):
        return self._factors[i]

    def __iter__(self):
        for f in deepcopy(self._factors):
            yield f

    def __tex__(self):
        return f"\\avg{{{' '.join([tex(f) for f in self._factors])}}}"

    def __copy__(self):
        return Trace(deepcopy(self._factors))

    def is_deterministic(self):
        return all([f.is_deterministic() for f in self._factors])

    def cycle(self):
        self._factors = self._factors[1:] + self._factors[:1]


class Graph(Texable):
    _coefficients: tuple[Coefficient, ...]
    _traces: tuple[Trace, ...]
    _deterministics: tuple[Trace, ...]
    _light_weights: tuple[Trace, ...]
    _g_loops: tuple[Trace, ...]

    def __init__(self, *args):
        coefficients: list[Coefficient] = []
        traces: list[Trace] = []
        deterministics: list[Trace] = []
        light_weights: list[Trace] = []
        g_loops: list[Trace] = []

        def _parse_inputs(args):
            if isinstance(args, Trace):
                traces.append(args)
            elif isinstance(args, Coefficient):
                coefficients.append(args)
            elif isinstance(args, Graph):
                _parse_inputs(list(args.coefficients) + list(args.traces))
            elif isinstance(args, list | tuple):
                for x in args:
                    _parse_inputs(x)
            else:
                raise TypeError(f"Tried to initialize a Graph with {args}")

        _parse_inputs(args)

        traces_to_remove: list[Trace] = []
        for t in traces:
            n_wtG = len([f for f in t if isinstance(f, wtG)])
            n_G = len([f for f in t if isinstance(f, G)])

            if n_wtG == 0 and n_G == 0:
                if len(t) == 4:
                    M_1, M_2 = t[0], t[2]
                    E_1, E_2 = t[1], t[3]
                    assert isinstance(M_1, M) and isinstance(M_2, M)
                    assert isinstance(E_1, E) and isinstance(E_2, E)
                    coefficients.append(calM(M_1.charge, M_2.charge, E_1.i, E_2.i))
                    traces_to_remove.append(t)
                else:
                    deterministics.append(t)
            elif n_wtG == 0 and n_G > 1:
                i = best_G_index(t)
                t = Trace(t[i:] + t[:i])
                g_loops.append(t)
            elif n_wtG == 1 and n_G == 0:
                light_weights.append(t)
            else:
                render(t)
                raise TypeError("Trace is not deterministic, light-weight, or G-loop")

        for t in traces_to_remove:
            traces.remove(t)

        deterministics.sort(
            key=lambda t: (
                len(t),
                max([f.i for f in t if isinstance(f, E)]),
            )
        )
        coefficients.sort(
            key=lambda c: (
                min([c.i, c.j]) if isinstance(c, Theta | STheta) else Symbol("")
            )
        )
        g_loops.sort(key=lambda t: len([f for f in t if isinstance(f, G)]))

        self._coefficients = tuple(coefficients)
        self._traces = tuple(traces)
        self._deterministics = tuple(deterministics)
        self._light_weights = tuple(light_weights)
        self._g_loops = tuple(g_loops)

    def __mul__(self, other):
        raise NotImplementedError("__mul__ not implemented on Graph")

    __rmul__ = __mul__

    def __tex__(self):
        deterministic_string = f"{''.join([tex(t) for t in self.coefficients])}{''.join([tex(t) for t in self.deterministics])}"
        non_deterministic_string = Rf"\E{''.join([tex(t) for t in self.light_weights])}{''.join([tex(t) for t in self.g_loops])}"
        if len(self.deterministics) == len(self.traces):
            return deterministic_string
        else:
            return deterministic_string + non_deterministic_string

    def is_deterministic(self):
        return not self.light_weights and not self.g_loops

    @property
    def coefficients(self):
        return deepcopy(self._coefficients)

    @property
    def traces(self):
        return deepcopy(self._traces)

    @property
    def deterministics(self):
        return deepcopy(self._deterministics)

    @property
    def light_weights(self):
        return deepcopy(self._light_weights)

    @property
    def g_loops(self):
        return deepcopy(self._g_loops)


class Size(Texable):
    # TODO: _member syntax and add getters
    n_exponent: float
    eta_exponent: float

    def __init__(self, n_exponent: float, eta_exponent: float):
        self.n_exponent = n_exponent
        self.eta_exponent = eta_exponent

    def __tex__(self):
        if self.n_exponent == 0 and self.eta_exponent == 0:
            return "1"

        if self.n_exponent == 0:
            n_string = ""
        elif self.n_exponent == 1:
            n_string = "N"
        else:
            n_string = f"N^{{{self.n_exponent}}}"

        if self.eta_exponent == 0:
            eta_string = ""
        elif self.eta_exponent == 1:
            eta_string = "\\eta"
        else:
            eta_string = f"\\eta^{{{self.eta_exponent}}}"

        return n_string + eta_string

    def __lt__(self, other):
        if isinstance(other, Size):
            return (self.n_exponent, -self.eta_exponent) < (
                other.n_exponent,
                -other.eta_exponent,
            )
        else:
            raise TypeError("Tried to compare Size with non-Size object")

    def __le__(self, other):
        if isinstance(other, Size):
            return (self.n_exponent, -self.eta_exponent) <= (
                other.n_exponent,
                -other.eta_exponent,
            )
        else:
            raise TypeError("Tried to compare Size with non-Size object")

    def __eq__(self, other):
        if isinstance(other, Size):
            return (self.n_exponent, self.eta_exponent) == (
                other.n_exponent,
                other.eta_exponent,
            )
        else:
            raise TypeError("Tried to compare Size with non-Size object")

    def __truediv__(self, other):
        if isinstance(other, Size):
            return Size(
                self.n_exponent - other.n_exponent,
                self.eta_exponent - other.eta_exponent,
            )
        else:
            raise TypeError("Tried to divide Size by non-Size object")

    def __mul__(self, other):
        if isinstance(other, Size):
            return Size(
                self.n_exponent + other.n_exponent,
                self.eta_exponent + other.eta_exponent,
            )
        else:
            raise TypeError("Tried to multiply Size by non-Size object")

    __rmul__ = __mul__

    def __hash__(self):
        return (self.n_exponent, self.eta_exponent).__hash__()


def last_G_index(t: Trace) -> int:
    i = len(t) - 1
    while i > -1:
        if isinstance(t[i], G | wtG):
            return i
        i -= 1
    return i


def best_G_index(t: Trace) -> int:
    # (1) The best G must result Theta^(+,+) or Theta^(-,-)
    # (2) Among these G's, we prefer a G next to an external index (so all M-loops will not contain any a indices)
    # (3) Among these G's, we prefer a G followed by deterministic matrices (e.g., <GEBEGE> is preferred over <GEGEBE>)
    # First index of the score is 1 if (1) holds and second index of the score is 1 if (2) holds
    n_factors = len(t)
    scores = [[0, 0, 0] for _ in range(n_factors)]
    for i, f in enumerate(t):
        if isinstance(f, G):
            j = i - 1
            while j > -1:
                if isinstance(t[j], G):
                    break
                j -= 1
            else:
                j = last_G_index(t)
            prev_G: G | wtG = t[j]
            prev_E: E = t[i - 1]

            if f.charge == prev_G.charge:
                scores[i][0] = 1
            if isinstance(prev_E.i, a):
                scores[i][1] = 1
            if isinstance(t[(i + 2) % n_factors], M):
                scores[i][2] = 1
    return max(range(len(scores)), key=scores.__getitem__)
