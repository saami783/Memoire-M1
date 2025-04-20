import random

def maximum_degree_greedy(graph):
    """
    Algorithme glouton qui sélectionne le sommet de degré maximum, l'ajoute au cover, et supprime ses arêtes.
    Si plusieurs sommets ont le même degré maximum, on en choisit un aléatoirement.
    """
    C = set()
    temp_graph = graph.copy()

    while temp_graph.number_of_edges() > 0:
        degrees = dict(temp_graph.degree())
        max_degree = max(degrees.values())
        max_degree_nodes = [node for node, degree in degrees.items() if degree == max_degree]
        max_degree_node = random.choice(max_degree_nodes)

        C.add(max_degree_node)
        temp_graph.remove_node(max_degree_node)

    return list(C)