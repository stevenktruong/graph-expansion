from typing import Self
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

    @property
    def n_thetas(self) -> int:
        raise NotImplementedError("n_thetas() not implemented on Coefficient")

    def transpose(self) -> Self:
        raise NotImplementedError("transpose() not implemented on Coefficient")


class ChargedCoefficient(Coefficient):
    _charges: tuple[Charge, Charge]

    def __init__(self, charge1: Charge, charge2: Charge, i: Symbol, j: Symbol):
        super().__init__(i, j)
        self._charges = (charge1, charge2)

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

    @property
    def n_thetas(self) -> int:
        return 0

    def transpose(self) -> Self:
        return type(self)(self.j, self.i)


class I(Coefficient):
    def __tex__(self):
        return Rf"I_{{{tex(self.i)} {tex(self.j)}}}"

    @property
    def n_thetas(self) -> int:
        return 0

    def transpose(self) -> Self:
        return type(self)(self.j, self.i)


class Theta(ChargedCoefficient):
    def __tex__(self):
        return Rf"\Theta^{{\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}}}_{{{tex(self.i)}{tex(self.j)}}}"

    @property
    def n_thetas(self) -> int:
        return 1

    def transpose(self) -> Self:
        raise TypeError("Theta.transpose() called (likely a mistake)")


class calM(ChargedCoefficient):
    def __tex__(self):
        return Rf"\M^{{\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}}}_{{{tex(self.i)}{tex(self.j)}}}"

    @property
    def n_thetas(self) -> int:
        return 0

    def transpose(self) -> Self:
        return type(self)(self.charges[1], self.charges[0], self.j, self.i)


class calMS(ChargedCoefficient):
    def __tex__(self):
        return Rf"\p{{\M^{{\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}}}S}}_{{{tex(self.i)}{tex(self.j)}}}"

    @property
    def n_thetas(self) -> int:
        return 0

    def transpose(self) -> Self:
        raise TypeError("calMS.transpose() called (likely a mistake)")


class ThetacalM(ChargedCoefficient):
    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{\Theta^{{{charge_string}}}\M^{{{charge_string}}}}}_{{{tex(self.i)}{tex(self.j)}}}"

    @property
    def n_thetas(self) -> int:
        return 1

    def transpose(self) -> Self:
        return type(self)(self.charges[1], self.charges[0], self.j, self.i)


class STheta(ChargedCoefficient):
    _charges: tuple[Charge, Charge]
    _indices: tuple[Symbol, Symbol]

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

    @property
    def n_thetas(self) -> int:
        return 1

    def transpose(self) -> Self:
        return type(self)(self.charges[1], self.charges[0], self.j, self.i)


# TODO: Maybe use a more general class then make each product its own class...


class ThetacalMS(ChargedCoefficient):
    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{\Theta^{{{charge_string}}}\M^{{{charge_string}}}S}}_{{{tex(self.i)}{tex(self.j)}}}"

    @property
    def n_thetas(self) -> int:
        return 1

    def transpose(self) -> Self:
        raise TypeError("ThetacalMS.transpose() called (likely a mistake)")


class ThetacalMSTheta(ChargedCoefficient):
    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{\Theta^{{{charge_string}}}\M^{{{charge_string}}}S\Theta^{{{charge_string}}}}}_{{{tex(self.i)}{tex(self.j)}}}"

    @property
    def n_thetas(self) -> int:
        return 2

    def transpose(self) -> Self:
        raise TypeError("ThetacalMSTheta.transpose() called (likely a mistake)")


class SThetacalMS(ChargedCoefficient):
    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{S\Theta^{{{charge_string}}}\M^{{{charge_string}}}S}}_{{{tex(self.i)}{tex(self.j)}}}"

    @property
    def n_thetas(self) -> int:
        return 1

    def transpose(self) -> Self:
        return type(self)(self.charges[1], self.charges[0], self.j, self.i)


class SThetaTheta(ChargedCoefficient):
    _charges: tuple[Charge, Charge]
    _indices: tuple[Symbol, Symbol]

    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{S\Theta^{{{charge_string}}}\Theta^{{{charge_string}}}}}_{{{tex(self.i)}{tex(self.j)}}}"

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

    @property
    def n_thetas(self) -> int:
        return 2

    def transpose(self) -> Self:
        return type(self)(self.charges[1], self.charges[0], self.j, self.i)


class SThetaThetaMS(ChargedCoefficient):
    _charges: tuple[Charge, Charge]
    _indices: tuple[Symbol, Symbol]

    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{S\Theta^{{{charge_string}}}\Theta^{{{charge_string}}}\M^{{{charge_string}}}S}}_{{{tex(self.i)}{tex(self.j)}}}"

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

    @property
    def n_thetas(self) -> int:
        return 2

    def transpose(self) -> Self:
        return type(self)(self.charges[1], self.charges[0], self.j, self.i)


class SThetacalMSTheta(ChargedCoefficient):
    _charges: tuple[Charge, Charge]
    _indices: tuple[Symbol, Symbol]

    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{S\Theta^{{{charge_string}}}\M^{{{charge_string}}}S\Theta^{{{charge_string}}}}}_{{{tex(self.i)}{tex(self.j)}}}"

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

    @property
    def n_thetas(self) -> int:
        return 2

    def transpose(self) -> Self:
        return type(self)(self.charges[1], self.charges[0], self.j, self.i)


class ThetacalMScalMS(ChargedCoefficient):
    _charges: tuple[Charge, Charge]
    _indices: tuple[Symbol, Symbol]

    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{\Theta^{{{charge_string}}}\M^{{{charge_string}}}S\M^{{{charge_string}}}S}}_{{{tex(self.i)}{tex(self.j)}}}"

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

    @property
    def n_thetas(self) -> int:
        return 1

    def transpose(self) -> Self:
        raise TypeError("ThetacalMScalMS.transpose() called (likely a mistake)")


class SThetacalMSThetacalMS(ChargedCoefficient):
    _charges: tuple[Charge, Charge]
    _indices: tuple[Symbol, Symbol]

    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{S\Theta^{{{charge_string}}}\M^{{{charge_string}}}S\Theta^{{{charge_string}}}\M^{{{charge_string}}}S}}_{{{tex(self.i)}{tex(self.j)}}}"

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

    @property
    def n_thetas(self) -> int:
        return 2

    def transpose(self) -> Self:
        return type(self)(self.charges[1], self.charges[0], self.j, self.i)


class ThetacalMSThetacalMS(ChargedCoefficient):
    _charges: tuple[Charge, Charge]
    _indices: tuple[Symbol, Symbol]

    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{\Theta^{{{charge_string}}}\M^{{{charge_string}}}S\Theta^{{{charge_string}}}\M^{{{charge_string}}}S}}_{{{tex(self.i)}{tex(self.j)}}}"

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

    @property
    def n_thetas(self) -> int:
        return 2

    def transpose(self) -> Self:
        raise TypeError("ThetacalMSThetacalMS.transpose() called (likely a mistake)")


class ThetacalMSThetacalM(ChargedCoefficient):
    _charges: tuple[Charge, Charge]
    _indices: tuple[Symbol, Symbol]

    def __tex__(self):
        charge_string = (
            Rf"\p{{{tex(self.charges[0].value)}, {tex(self.charges[1].value)}}}"
        )
        return Rf"\p{{\Theta^{{{charge_string}}}\M^{{{charge_string}}}S\Theta^{{{charge_string}}}\M^{{{charge_string}}}}}_{{{tex(self.i)}{tex(self.j)}}}"

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

    @property
    def n_thetas(self) -> int:
        return 2

    def transpose(self) -> Self:
        return type(self)(self.charges[1], self.charges[0], self.j, self.i)
