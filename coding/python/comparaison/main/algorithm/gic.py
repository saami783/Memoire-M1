import random

def greedy_independent_cover(graph):
    C = set()
    temp_graph = graph.copy()

    while temp_graph.number_of_edges() > 0:
        degrees = dict(temp_graph.degree())
        min_degree = min(degrees.values())
        min_degree_nodes = [node for node, degree in degrees.items() if degree == min_degree]
        # Choix aléatoire parmi les nœuds de degré min
        min_degree_node = random.choice(min_degree_nodes)

        neighbors = set(temp_graph.neighbors(min_degree_node))
        C.update(neighbors)

        temp_graph.remove_nodes_from(neighbors | {min_degree_node})

    return list(C)