from typing import Literal, Optional

from .Symbol import Charge, Symbol
from .Texable import Texable, tex


class Coefficient(Texable):
    _indices: tuple[Symbol, Symbol]

    def __init__(
        self,
        i: Symbol,
        j: Symbol,
    ):
        self._indices = (i, j)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._indices == other._indices

    def __mul__(self, other):
        raise NotImplementedError("__mul__ not implemented on Coefficient")

    __rmul__ = __mul__

    @property
    def indices(self):
        return self._indices

    @property
    def i(self):
        return self._indices[0]

    @property
    def j(self):
        return self._indices[1]


class ChargedCoefficient(Coefficient):
    _charges: tuple[Charge, Charge]

    def __init__(self, charge_1: Charge, charge_2: Charge, i: Symbol, j: Symbol):
        super().__init__(i, j)
        self._charges = (charge_1, charge_2)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._charges == other._charges and self._indices == other._indices

    @property
    def charges(self):
        return (self._charges[0], self._charges[1])


class S(Coefficient):

    def __tex__(self):
        return Rf"S_{{{tex(self.i)} {tex(self.j)}}}"


class I(Coefficient):
    def __tex__(self):
        return Rf"I_{{{tex(self.i)} {tex(self.j)}}}"


class Theta(ChargedCoefficient):
    def __tex__(self):
        return Rf"\Theta^{{\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}}}_{{{tex(self.i)}{tex(self.j)}}}"


class calM(ChargedCoefficient):
    def __tex__(self):
        return Rf"\M^{{\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}}}_{{{tex(self.i)}{tex(self.j)}}}"


class STheta(ChargedCoefficient):
    _charges: tuple[Charge, Charge]
    _indices: tuple[Symbol, Symbol]

    def __init__(
        self,
        charge_1: Charge,
        charge_2: Charge,
        i: Symbol,
        j: Symbol,
    ):
        self._charges = (charge_1, charge_2)
        self._indices = (i, j)

    def __tex__(self):
        return Rf"\p{{S\Theta^{{\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}}}}}_{{{tex(self.i)}{tex(self.j)}}}"

    def __eq__(self, other):
        if not isinstance(other, STheta):
            return False
        return (
            self._charges == other._charges
            and self._indices == other._indices
            or (
                (self._charges[1], self._charges[0]) == other._charges
                and (self._indices[1], self._indices[0]) == other._indices
            )
        )


class ThetacalMS(ChargedCoefficient):
    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{\Theta^{{{charge_string}}}\M^{{{charge_string}}}S}}_{{{tex(self.i)}{tex(self.j)}}}"


class SThetacalMS(ChargedCoefficient):
    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{S\Theta^{{{charge_string}}}\M^{{{charge_string}}}S}}_{{{tex(self.i)}{tex(self.j)}}}"
