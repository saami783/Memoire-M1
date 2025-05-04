import random


def edge_deletion_smart(graph):
    """
    Variante 'intelligente' : choisir entre u et v le sommet de plus haut degré
    pour maximiser les arêtes supprimées à chaque itération.
    """
    C = set()
    temp_graph = graph.copy()

    while temp_graph.number_of_edges() > 0:
        u, v = random.choice(list(temp_graph.edges()))
        if temp_graph.degree[u] >= temp_graph.degree[v]:
            chosen = u
        else:
            chosen = v
        C.add(chosen)
        temp_graph.remove_node(chosen)

    return list(C)


def edge_deletion(graph):
    """
    - Choisir une arête (u,v)
    - Ajouter u et v au vertex cover
    - Supprimer u et v (et donc toutes leurs arêtes incidentes)
    """
    C = set()
    temp_graph = graph.copy()

    while temp_graph.number_of_edges() > 0:
        u, v = random.choice(list(temp_graph.edges()))
        C.update([u, v])
        temp_graph.remove_nodes_from([u, v])

    return list(C)

def edge_deletion_max_degree_sum(graph):
    """
    Variante qui choisit l'arête dont la somme des degrés des deux nœuds est la plus élevée,
    puis ajoute ces deux nœuds au vertex cover et les supprime du graphe.
    """
    C = set()
    temp_graph = graph.copy()

    while temp_graph.number_of_edges() > 0:
        # Trouver l'arête avec la somme des degrés maximale
        edges = list(temp_graph.edges())
        max_edge = max(edges, key=lambda e: temp_graph.degree[e[0]] + temp_graph.degree[e[1]])
        u, v = max_edge
        # Ajouter les deux nœuds au vertex cover
        C.update([u, v])
        # Supprimer les nœuds et leurs arêtes incidentes
        temp_graph.remove_nodes_from([u, v])

    return list(C)