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
    subscript: str | None

    def __init__(
        self,
        subscript: str | None = None,
        conjugated: bool = False,
        transposed: bool = False,
        like=None,
    ):
        super().__init__(conjugated, transposed, like)
        self.subscript = None
        if (
            isinstance(like, G) or isinstance(like, wtG) or isinstance(like, M)
        ) and like.subscript:
            self.subscript = like.subscript
        if subscript:
            self.subscript = subscript

    def __tex__(self):
        if self.subscript and len(self.subscript) == 1:
            subscript_string = f"_{self.subscript}"
        elif self.subscript and len(self.subscript) > 1:
            subscript_string = f"_{{{self.subscript}}}"
        else:
            subscript_string = ""

        if self.conjugated and self.transposed:
            return f"G{subscript_string}^*"
        elif self.conjugated:
            return f"\\conj{{G{subscript_string}}}"
        elif self.transposed:
            return f"G{subscript_string}^\\top"
        else:
            return f"G{subscript_string}"

    def is_deterministic(self):
        return False


class wtG(MatrixFactor):
    subscript: str | None

    def __init__(
        self,
        subscript: str | None = None,
        conjugated: bool = False,
        transposed: bool = False,
        like=None,
    ):
        super().__init__(conjugated, transposed, like)
        self.subscript = None
        if (
            isinstance(like, G) or isinstance(like, wtG) or isinstance(like, M)
        ) and like.subscript:
            self.subscript = like.subscript
        if subscript:
            self.subscript = subscript

    def __tex__(self):
        if self.subscript and len(self.subscript) == 1:
            subscript_string = f"_{self.subscript}"
        elif self.subscript and len(self.subscript) > 1:
            subscript_string = f"_{{{self.subscript}}}"
        else:
            subscript_string = ""

        if self.conjugated and self.transposed:
            return f"\\G{subscript_string}^*"
        elif self.conjugated:
            return f"\\conj{{\\G{subscript_string}}}"
        elif self.transposed:
            return f"\\G{subscript_string}^\\top"
        else:
            return f"\\G{subscript_string}"

    def is_deterministic(self):
        return False


class M(MatrixFactor):
    subscript: str | None

    def __init__(
        self,
        subscript: str | None = None,
        conjugated: bool = False,
        transposed: bool = False,
        like=None,
    ):
        super().__init__(conjugated, transposed, like)
        self.subscript = None
        if (
            isinstance(like, G) or isinstance(like, wtG) or isinstance(like, M)
        ) and like.subscript:
            self.subscript = like.subscript
        if subscript:
            self.subscript = subscript

    def __tex__(self):
        if self.subscript and len(self.subscript) == 1:
            subscript_string = f"_{self.subscript}"
        elif self.subscript and len(self.subscript) > 1:
            subscript_string = f"_{{{self.subscript}}}"
        else:
            subscript_string = ""

        if self.conjugated and self.transposed:
            return f"M{subscript_string}^*"
        elif self.conjugated:
            return f"\\conj{{M{subscript_string}}}"
        elif self.transposed:
            return f"M{subscript_string}^\\top"
        else:
            return f"M{subscript_string}"

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


class Delta(MatrixFactor):
    k: str
    l: str

    def __init__(
        self,
        k: str = "",
        l: str = "",
        conjugated: bool = False,
        transposed: bool = False,
        like=None,
    ):
        super().__init__(conjugated, transposed, like)
        self.k = k
        self.l = l

    def __tex__(self):
        superscript_string = ""
        if self.k and self.l:
            superscript_string = (
                f"^{{{self.k} {self.l}}}"
                if not self.transposed
                else f"^{{{self.l} {self.k}}}"
            )
        elif self.transposed:
            superscript_string = "^{\\top}"
        return f"\\Delta{superscript_string}"

    def is_deterministic(self):
        return True
