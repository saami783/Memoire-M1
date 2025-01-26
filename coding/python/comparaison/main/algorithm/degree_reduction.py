def degree_reduction(graph):
    """
    Degree Reduction Heuristic.
    Iteratively removes high-degree vertices.
    """
    C = set()
    temp_graph = graph.copy()
    while temp_graph.number_of_edges() > 0:
        max_degree_node = max(temp_graph.degree, key=lambda x: x[1])[0]
        C.add(max_degree_node)
        temp_graph.remove_node(max_degree_node)
    return list(C)