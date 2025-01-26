import networkx as nx

def matching_based(graph):
    """
    Matching-Based Algorithm.
    Computes a maximal matching and selects both endpoints of each matched edge.
    """
    matching = nx.maximal_matching(graph)
    C = set()
    for u, v in matching:
        C.update([u, v])
    return list(C)