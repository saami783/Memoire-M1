import random


def edge_deletion_vertex_cover(graph):
    """
    Algorithme "Edge Deletion" (2-approximation) pour Vertex Cover.
    Tant qu'il y a des arêtes dans le graphe, on en choisit une (u,v),
    on ajoute 'u' (ou 'v') au cover, et on supprime les arêtes incidentes à u.
    """
    C = set()
    temp_graph = graph.copy()

    # Tant qu'il reste au moins une arête
    while temp_graph.number_of_edges() > 0:
        # Choisir une arête (u, v) au hasard
        (u, v) = random.choice(list(temp_graph.edges()))

        # Ajouter l'un de ses sommets au cover (ici: u)
        C.add(u)

        # Supprimer u (et donc toutes les arêtes incidentes) du graphe
        temp_graph.remove_node(u)

    return list(C)
