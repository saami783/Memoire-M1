import networkx as nx

def matching_based(graph):
    """
    Cet algorithme est basé sur la recherche d'un couplage maximal dans le graphe.
    Il retourne un ensemble de sommets qui forment un couplage maximal.

    Approximation 2-approchée.
    """
    matching = nx.maximal_matching(graph)
    C = set()
    for u, v in matching:
        C.update([u, v])
    return list(C)