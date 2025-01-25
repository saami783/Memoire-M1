import random

def minimum_degree_greedy(graph):
    C = set()
    temp_graph = graph.copy()

    while temp_graph.number_of_edges() > 0:
        degrees = dict(temp_graph.degree())
        min_degree = min(degrees.values())
        min_degree_nodes = [node for node, deg in degrees.items() if deg == min_degree]
        # Choix aléatoire parmi les nœuds de degré min
        min_degree_node = random.choice(min_degree_nodes)

        C.add(min_degree_node)
        temp_graph.remove_node(min_degree_node)

    return list(C)
