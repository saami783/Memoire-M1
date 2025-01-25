def primal_dual_vertex_cover(graph):
    """
    Primal-Dual Algorithm for Vertex Cover.
    Uses a dual fitting technique to iteratively build a feasible solution.
    """
    C = set()
    for u, v in graph.edges():
        if u not in C and v not in C:
            C.update([u, v])
    return list(C)