def primal_dual(graph):
    """
    # @todo supprimer cet algo
    Primal-Dual Algorithm for Vertex Cover.
    Uses a dual fitting technique to iteratively build a feasible solution.
    """
    C = set()
    for u, v in graph.edges():
        if u not in C and v not in C:
            # ajouter u et v
            C.update(u)
            C.update(v)
        # C.update([u, v])
    return list(C)