from .Symbol import Charge, Symbol
from .Texable import Texable, tex


class MatrixFactor(Texable):
    _charge: Charge

    def __init__(self, charge: Charge = Charge.Plus, like=None):
        if isinstance(like, MatrixFactor):
            self._charge = like._charge
        else:
            self._charge = charge

    def __adjoint__(self):
        if self.charge == Charge.Plus:
            self._charge = Charge.Minus
        elif self.charge == Charge.Minus:
            self._charge = Charge.Plus

    def is_deterministic(self) -> bool:
        raise NotImplementedError("is_deterministic not implemented")

    @property
    def charge(self):
        return self._charge


class G(MatrixFactor):
    def __tex__(self):
        if self.charge == Charge.Plus:
            return "G"
        else:
            return "G^*"

    def is_deterministic(self):
        return False


class wtG(MatrixFactor):
    def __tex__(self):
        if self.charge == Charge.Plus:
            return R"\G"
        else:
            return R"\G^*"

    def is_deterministic(self):
        return False


class M(MatrixFactor):
    def __tex__(self):
        if self.charge == Charge.Plus:
            return R"M"
        else:
            return R"M^*"

    def __eq__(self, other):
        if not isinstance(other, M):
            return False
        return self.charge == other.charge

    def is_deterministic(self):
        return True


class E(MatrixFactor):
    _i: Symbol

    def __init__(self, i: Symbol):
        super().__init__(Charge.Neutral)
        self._i = i

    def __tex__(self):
        return Rf"E_{{{tex(self.i)}}}"

    def __eq__(self, other):
        if not isinstance(other, E):
            return False
        return self.i == other.i

    def is_deterministic(self):
        return True

    @property
    def i(self):
        return self._i
