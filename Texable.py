from IPython.display import display, Latex


macros = """
\\gdef\\avg#1{\\mathopen{}\\left\\langle #1 \\right\\rangle\\mathclose{}}
\\gdef\\p#1{\\mathopen{}\\left\\lparen #1 \\right\\rparen\\mathclose{}}
\\gdef\\conj#1{\\overline{#1}}
\\gdef\\G{\\widetilde{G}}
\\gdef\\E{\\mathbb{E}}
"""


class Texable:
    def __tex__(self) -> str:
        raise NotImplementedError("__tex__ not implemented")


def tex(x: Texable) -> str:
    return x.__tex__()


def render(x: Texable):
    display(Latex(f"${macros + tex(x)}$"))
