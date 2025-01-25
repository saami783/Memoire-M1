import random

def bar_yehuda_even(graph):
    """
    Bar-Yehuda & Even Algorithm for Vertex Cover (2-approximation).
    Selects an arbitrary uncovered edge, adds both endpoints to the cover.
    """
    C = set()
    temp_graph = graph.copy()
    while temp_graph.number_of_edges() > 0:
        u, v = random.choice(list(temp_graph.edges()))
        C.update([u, v])
        temp_graph.remove_node(u)
        temp_graph.remove_node(v)
    return list(C)