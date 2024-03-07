from Texable import Texable, tex


class MatrixFactor(Texable):
    conjugated: bool
    transposed: bool

    def __init__(self, conjugated: bool = False, transposed: bool = False, like=None):
        if isinstance(like, MatrixFactor):
            self.conjugated = like.conjugated
            self.transposed = like.transposed
        else:
            self.conjugated = conjugated
            self.transposed = transposed

    def __copy__(self):
        return MatrixFactor(like=self)

    def is_deterministic(self) -> bool:
        raise NotImplemented("is_deterministic not implemented")


class Alpha(Texable):
    index: int

    def __init__(self, alpha=None):
        if isinstance(alpha, int):
            self.index = alpha
        elif isinstance(alpha, Alpha):
            self.index = alpha.index
        elif not alpha:
            self.index = 0
        else:
            raise TypeError("alpha must be int or Alpha")

    def __tex__(self):
        if self.index > 9:
            return f"\\alpha_{{{self.index}}}"
        else:
            return f"\\alpha_{self.index}"

    def __copy__(self):
        return Alpha(self.index)


class G(MatrixFactor):
    def __tex__(self):
        if self.conjugated and self.transposed:
            return "G^*"
        elif self.conjugated:
            return "\\conj{G}"
        elif self.transposed:
            return "G^\\top"
        else:
            return "G"

    def is_deterministic(self):
        return False


class wtG(MatrixFactor):
    def __tex__(self):
        if self.conjugated and self.transposed:
            return "\\G^*"
        elif self.conjugated:
            return "\\conj{\\G}"
        elif self.transposed:
            return "\\G^\\top"
        else:
            return "\\G"

    def is_deterministic(self):
        return False


class M(MatrixFactor):
    def __tex__(self):
        if self.conjugated and self.transposed:
            return "M^*"
        elif self.conjugated:
            return "\\conj{M}"
        elif self.transposed:
            return "M^\\top"
        else:
            return "M"

    def is_deterministic(self):
        return True


class E(MatrixFactor):
    alpha: Alpha

    def __init__(
        self,
        alpha: int | Alpha | None = None,
        conjugated: bool = False,
        transposed: bool = False,
        like=None,
    ):
        super().__init__(conjugated, transposed, like)
        self.alpha = Alpha(alpha)

    def __tex__(self):
        if self.alpha.index == 0:
            return "E_{\\alpha}"
        else:
            return f"E_{{{tex(self.alpha)}}}"

    def __copy__(self):
        return E(self.alpha.index, like=self)

    def is_deterministic(self):
        return True


class B(MatrixFactor):
    i: int | None

    def __init__(
        self,
        i: int | None = None,
        conjugated: bool = False,
        transposed: bool = False,
        like=None,
    ):
        super().__init__(conjugated, transposed, like)
        self.i = i

    def __tex__(self):
        if not self.i:
            B_string = "B"
        elif self.i < 9:
            B_string = f"B_{self.i}"
        else:
            B_string = f"B_{{{self.i}}}"

        if self.conjugated and self.transposed:
            return f"{B_string}^*"
        elif self.conjugated:
            return f"\\conj{{{B_string}}}"
        elif self.transposed:
            return f"{B_string}^\\top"
        else:
            return f"{B_string}"

    def is_deterministic(self):
        return True


class DeltaKL(MatrixFactor):
    def __tex__(self):
        if self.transposed:
            return "\\Delta^{\\ell k}"
        else:
            return "\\Delta^{k\\ell}"

    def is_deterministic(self):
        return True
