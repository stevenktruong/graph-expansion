from .visualization import to_nx_graph

from graph_expansion import *


def external_vertex_order(x: Graph):
    g = to_nx_graph(x, with_external_edges=False)[0]

    vertices_connected_to_loop: list[Symbol] = [a(1)]
    external_edges: dict[Symbol, STheta] = {}

    first_b = [
        v for v in g.successors(a(1)) if not isinstance(g[a(1)][v]["matrix"], STheta)
    ][0]
    curr = next(g.successors(first_b))
    while curr != first_b:
        external_vertex = (
            [
                v
                for v in g.successors(curr)
                if isinstance(g[curr][v]["matrix"], STheta | Theta)
            ]
            + [
                v
                for v in g.predecessors(curr)
                if isinstance(g[v][curr]["matrix"], STheta | Theta)
            ]
        )[0]
        vertices_connected_to_loop.append(external_vertex)
        e: Coefficient = (
            g[curr][external_vertex]["matrix"]
            if (curr, external_vertex) in g.edges()
            else g[external_vertex][curr]["matrix"]
        )
        if isinstance(e, STheta):
            for u in [curr, external_vertex]:
                if e.i != u:
                    e = e.transpose()
                external_edges[u] = e

        curr = [
            v
            for v in g.successors(curr)
            if not isinstance(g[curr][v]["matrix"], STheta)
        ][0]

    return vertices_connected_to_loop, external_edges
