import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from numpy.typing import NDArray

from graph_expansion import *

neutral_color = "#211a1d"
non_conjugated_color = "#8b85c1"
conjugated_color = "#ed254e"

node_color = "#dce0d9"
highlight_color = "#fed766"


def get_edge_color(u: Symbol, v: Symbol, f: Theta | STheta | M | G | ThetacalM) -> str:
    if isinstance(f, Theta | STheta):
        if f.charges[0] == f.charges[1]:
            return get_charge_color(f.charges[0])
        else:
            return "black"
    elif isinstance(f, ThetacalM):
        return "green"
    else:
        return non_conjugated_color if f.charge == Charge.Plus else conjugated_color


def get_edge_style(f: Theta | STheta | M | G) -> str:
    if isinstance(f, STheta):
        return "--"
    elif isinstance(f, G):
        return ":"
    else:
        return "-"


def get_arrow_style(f: Theta | STheta | M | G) -> str:
    if isinstance(f, STheta | ThetacalM):
        return "-"
    # else:
    return "-|>"


def get_charge_color(c: Charge):
    if c == Charge.Plus:
        return non_conjugated_color
    elif c == Charge.Minus:
        return conjugated_color
    else:
        return node_color


def get_charge_text(c: Charge):
    if c == Charge.Plus:
        return "$+$"
    elif c == Charge.Minus:
        return "$-$"
    else:
        return ""


def draw_theta_edges(
    g: nx.Graph,
    edges,
    pos: dict[Symbol, list[float]],
    ax: Axes,
    dperp: float = 1 / 16,
    linestyle="-",
    width=2.0,
    arrowsize=10,
    node_size=200,
):
    for v1, v2 in edges:
        c = g[v1][v2]["matrix"]
        assert isinstance(c, Theta | STheta)

        x1, x2 = pos[c.i], pos[c.j]
        charge1, charge2 = c.charges

        # Edges will be rotated sigmoid functions
        u = np.array([x2[0] - x1[0], x2[1] - x1[1]])
        norm_u = np.sqrt(np.sum(np.abs(u) ** 2))
        angle = np.arctan2(u[1], u[0])
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        )

        x = np.linspace(0, norm_u, 50)
        y = 2 * dperp * (sigma(x * (12 / norm_u) - 6)) - dperp
        p1: NDArray[np.float16] = x1 + (rotation_matrix @ np.vstack([x, y])).T
        p2: NDArray[np.float16] = (x1 + (rotation_matrix @ np.vstack([x, -y])).T)[::-1]

        # Some annoying work to calculate the size of the node markers
        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        xlim = ax.get_xlim()
        units_per_inch = (xlim[1] - xlim[0]) / bbox.width
        node_radius = np.sqrt(node_size / np.pi) * (units_per_inch / ax.figure.dpi)

        # Since the edges are shifted, we can draw them a little closer than just the radius
        inset = 1.4 * np.sqrt(node_radius**2 - dperp**2)
        mask = x <= x[-1] - inset
        for path, color in [
            (p1, get_charge_color(charge2)),
            (p2, get_charge_color(charge1)),
        ]:
            ax.add_patch(
                FancyArrowPatch(
                    path=Path(path[mask]),
                    arrowstyle="-|>",
                    color=color,
                    mutation_scale=arrowsize,
                    linestyle=linestyle,
                    linewidth=width,
                    zorder=1,  # arrows go behind nodes
                )
            )


def sigma(x):
    return np.reciprocal(1 + np.exp(-x))


# Legend styles


class NeutralThetaHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        x = np.linspace(x0, x0 + width, 50)
        y = height * (sigma(x * (12 / width) - 6)) + y0
        return [
            FancyArrowPatch(
                path=Path(np.vstack([x, y]).T),
                arrowstyle="-|>",
                color=non_conjugated_color,
                linestyle="--",
                mutation_scale=5,
            ),
            FancyArrowPatch(
                path=Path(np.vstack([x, height - (y - y0)]).T[::-1]),
                arrowstyle="-|>",
                color=conjugated_color,
                linestyle="--",
                mutation_scale=5,
            ),
        ]


class ChargedThetaHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        l = Line2D([y0 + width * 0.45, y0 + width * 0.55], [0, height], color="k")
        x = np.linspace(x0, x0 + 0.45 * width, 50)
        y = height * (sigma(x * (12 / (0.45 * width)) - 6)) + y0
        return [
            l,
            FancyArrowPatch(
                path=Path(np.vstack([x, y]).T),
                arrowstyle="-|>",
                color=non_conjugated_color,
                linestyle="--",
                mutation_scale=5,
            ),
            FancyArrowPatch(
                path=Path(
                    np.vstack([x, height - (y - y0)]).T[x < x0 + 0.35 * width][::-1]
                ),
                arrowstyle="-|>",
                color=non_conjugated_color,
                linestyle="--",
                mutation_scale=5,
            ),
            FancyArrowPatch(
                path=Path(np.vstack([0.55 * width + x, y]).T),
                arrowstyle="-|>",
                color=conjugated_color,
                linestyle="--",
                mutation_scale=5,
            ),
            FancyArrowPatch(
                path=Path(
                    np.vstack([0.55 * width + x, height - (y - y0)]).T[
                        x > 0.05 * width
                    ][::-1]
                ),
                arrowstyle="-|>",
                color=conjugated_color,
                linestyle="--",
                mutation_scale=5,
            ),
        ]


class MHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        l = Line2D([y0 + width * 0.45, y0 + width * 0.55], [0, height], color="k")
        l11 = Line2D(
            [x0, y0 + width * 0.4],
            [0.5 * height, 0.5 * height],
            color=non_conjugated_color,
        )
        l12 = Line2D(
            [y0 + width * 0.6, y0 + width],
            [0.5 * height, 0.5 * height],
            color=conjugated_color,
        )
        return [l, l11, l12]


class SigmaHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        l = Line2D([y0 + width * 0.45, y0 + width * 0.55], [0, height], color="k")
        l11 = Line2D(
            [x0, y0 + width * 0.4],
            [0.5 * height, 0.5 * height],
            linestyle=":",
            color=non_conjugated_color,
        )
        l12 = Line2D(
            [y0 + width * 0.6, y0 + width],
            [0.5 * height, 0.5 * height],
            linestyle=":",
            color=conjugated_color,
        )
        return [l, l11, l12]
