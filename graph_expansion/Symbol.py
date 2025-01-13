from enum import Enum

from .Texable import Texable


class Symbol(Texable):
    _value: str

    def __init__(self, value: str):
        self._value = value

    def __tex__(self):
        return self.value

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, Symbol):
            raise TypeError(f"Tried to compare Symbol with {other}")
        return self.value < other.value

    def __hash__(self):
        return self._value.__hash__()

    @property
    def value(self):
        return self._value


class Charge(Enum):
    Plus = Symbol("+")
    Minus = Symbol("-")
    Neutral = Symbol(R"\pm")


class NumberedSymbol(Symbol):
    _label: str
    _i: int

    def __init__(self, label: str, i: int):
        self._label = label
        self._i = i
        if i == 0:
            super().__init__(f"{self._label}")
        elif i < 10:
            super().__init__(f"{self._label}_{i}")
        else:
            super().__init__(Rf"{self._label}_{{{i}}}")

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.i < other.i
        else:
            return Symbol.__lt__(self, other)

    def __repr__(self):
        return self.__tex__()

    @property
    def i(self):
        return self._i


class a(NumberedSymbol):
    def __init__(self, i: int):
        super().__init__("a", i)


class b(NumberedSymbol):
    def __init__(self, i: int):
        super().__init__("b", i)


class d(Symbol):
    def __init__(self, i: int, j: int):
        super().__init__(f"d_{{{i},{j}}}")
