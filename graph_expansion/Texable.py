from IPython.display import Latex, display

macros = R"""
\gdef\avg#1{\mathopen{}\left\langle #1 \right\rangle\mathclose{}}
\gdef\p#1{\mathopen{}\left\lparen #1 \right\rparen\mathclose{}}
\gdef\conj#1{\overline{#1}}
\gdef\G{\widetilde{G}}
\gdef\M{\mathcal{M}}
\gdef\E{\mathbb{E}}
"""


class Texable:
    def __tex__(self) -> str:
        raise NotImplementedError("__tex__ not implemented")


def tex(x: Texable) -> str:
    return x.__tex__()


def tex_with_macros(x: Texable) -> str:
    return f"{macros} {tex(x)}"


def _get_latex(args) -> str:
    if isinstance(args, str):
        return f" {args} "
    elif isinstance(args, int | float):
        return f" {args} "
    elif isinstance(args, list | tuple):
        return " ".join([_get_latex(x) for x in args])
    elif isinstance(args, Texable):
        return f" {tex(args)} "
    else:
        raise TypeError(f"Could not generate LaTeX from {args.__class__}")


def render(*args, huge=False):
    latex = Rf"${R'\huge ' if huge else ''}{macros} {_get_latex(args)}$"
    display(Latex(latex))
