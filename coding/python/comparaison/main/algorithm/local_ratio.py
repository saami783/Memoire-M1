import random


def local_ratio_vertex_cover(graph):
    """
    Implémentation simple du Local Ratio, 2-approximation pour Vertex Cover.
    Très proche en pratique de l'algorithme de matching glouton.
    """
    C = set()
    temp_graph = graph.copy()

    edges = list(temp_graph.edges())
    random.shuffle(edges)  # On mélange l'ordre des arêtes

    for (u, v) in edges:
        # Si l'arête (u,v) n'est pas couverte
        if u not in C and v not in C:
            # On ajoute u au cover
            C.add(u)
            # Et on supprime toutes les arêtes incidentes à u
            if u in temp_graph:
                temp_graph.remove_node(u)

    return list(C)
