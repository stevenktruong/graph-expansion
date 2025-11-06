from collections import defaultdict

import networkx as nx
import numpy as np

from graph_expansion import *


def to_nx_graph(
    x: Graph,
    with_external_edges=True,
    extra_edges: list[Coefficient] = [],
    auto_stheta_edges: list[tuple[Symbol, Symbol, Symbol, Symbol]] = [],
    ignored_edges: list[tuple[Symbol, Symbol]] = [],
):
    g = nx.DiGraph()
    node_charges: dict[Symbol, Charge] = {}
    m_edges: list[tuple[Symbol, Symbol]] = []
    non_m_edges: list[tuple[Symbol, Symbol]] = []

    # non-M edges
    for c in list(x.coefficients) + extra_edges:
        if isinstance(c, ChargedCoefficient):
            for i, charge in zip(c.indices, c.charges):
                node_charges[i] = charge
            if (c.i, c.j) not in ignored_edges and (c.j, c.i) not in ignored_edges:
                non_m_edges.append((c.i, c.j))
                g.add_edge(c.i, c.j, matrix=c)

    # M edges
    for t in x.deterministics:
        for i, f in enumerate(t[:-1]):
            E_left = t[i - 1]
            E_right = t[i + 1]
            if isinstance(f, M) and isinstance(E_left, E) and isinstance(E_right, E):
                m_edges.append((E_left.i, E_right.i))
                g.add_edge(E_left.i, E_right.i, matrix=f)

    # Automatically color extra edges
    for b1, b2, pinf, minf in auto_stheta_edges:
        if g.has_edge(b1, b2):
            non_m_edges.remove((b1, b2))
            g.remove_edge(b1, b2)
        else:
            assert g.has_edge(b2, b1)
            non_m_edges.remove((b2, b1))
            g.remove_edge(b2, b1)
        non_m_edges.append((b1, pinf))
        non_m_edges.append((minf, b2))
        g.add_edge(
            b1, pinf, matrix=STheta(node_charges[b1], node_charges[b2], b1, pinf)
        )
        g.add_edge(
            minf, b2, matrix=STheta(node_charges[b1], node_charges[b2], minf, b2)
        )
        node_charges[pinf] = node_charges[b2]
        node_charges[minf] = node_charges[b1]

    return g, node_charges, m_edges, non_m_edges


def tutte_embedding(
    graph: nx.DiGraph | nx.Graph,
    external_vertices: list[Symbol],
    external_vertex_positions: Optional[dict[Symbol, list[float]]] = None,
):
    pos: dict[Symbol, list[float]] = defaultdict(lambda: [0, 0])
    if not external_vertex_positions:
        theta = (
            np.linspace(0, 2 * np.pi, len(external_vertices), endpoint=False)
            + (np.pi / 2)
            + (2 * np.pi) / len(external_vertices)
        )
        X = np.cos(theta)
        Y = np.sin(theta)
        for i, alpha in enumerate(external_vertices):
            pos[alpha] = [X[i], Y[i]]
    else:
        for vertex, position in external_vertex_positions.items():
            pos[vertex] = position

    internal_vertices = [u for u in graph if u not in external_vertices]
    size = len(internal_vertices)

    # System for x-coordinates
    A = np.eye(size, dtype=float)
    b = np.zeros((size,), dtype=float)

    # System for y-coordinates
    C = np.eye(size, dtype=float)
    d = np.zeros((size,), dtype=float)

    for i, u in enumerate(internal_vertices):
        if isinstance(graph, nx.DiGraph):
            neighbors = set(graph.successors(u)) | set(graph.predecessors(u))
        else:
            neighbors = set(graph.neighbors(u))
        n = len(neighbors)
        for v in neighbors:
            if v in external_vertices:
                b[i] += pos[v][0] / n
                d[i] += pos[v][1] / n
            else:
                j = internal_vertices.index(v)
                A[i][j] = -1 / n
                C[i][j] = -1 / n

    x = np.linalg.solve(A, b)
    y = np.linalg.solve(C, d)
    for i, u in enumerate(internal_vertices):
        pos[u] = [x[i], y[i]]

    return pos
