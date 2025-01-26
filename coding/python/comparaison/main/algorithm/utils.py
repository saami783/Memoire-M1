def is_valid_cover(C_set, graph):
    for (u, v) in graph.edges():
        if u not in C_set and v not in C_set:
            return False
    return True

# Recherche locale très simple
def local_search_simple(cover_set, graph):
    C = set(cover_set)
    improved = True
    while improved:
        improved = False
        for node in list(C):
            Ctemp = C - {node}
            if is_valid_cover(Ctemp, graph):
                C = Ctemp
                improved = True
                break
    return C