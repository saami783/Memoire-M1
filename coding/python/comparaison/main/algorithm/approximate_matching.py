import random

def approximate_matching_vertex_cover(graph):
    """
    Approximate Matching Algorithm.
    Picks an edge, adds one endpoint, removes covered edges.
    """
    C = set()
    temp_graph = graph.copy()
    while temp_graph.number_of_edges() > 0:
        u, v = random.choice(list(temp_graph.edges()))
        C.add(u)
        temp_graph.remove_node(u)
    return list(C)