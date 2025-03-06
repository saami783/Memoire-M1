# https://link.springer.com/article/10.1007/s11227-023-05397-8
def malatya_vertex_cover(graph):
    """
    Implémentation de l'algorithme Malatya pour Vertex Cover.
    - Calcule la centralité "Malatya" pour chaque nœud.
    - Sélectionne itérativement le nœud avec la centralité maximale.
    - Retourne la couverture sous forme de liste.

    Exemple : Si un nœud de degré 3 a deux voisins de degrés 2 et 4, sa centralité = 3/2 + 3/4 = 2.25
    L'algorithme privilégie les nœuds avec un haut degré ET des voisins peu connectés.
    """
    G = graph.copy()
    vertex_cover = []

    while G.number_of_edges() > 0:
        max_centrality = -1
        selected_node = None

        # Calculer la centralité pour chaque nœud
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue  # Ignorer les nœuds isolés

            # Centralité Malatya = sum(deg(node) / deg(voisin)
            node_degree = G.degree(node)
            centrality = sum(node_degree / G.degree(neighbor) for neighbor in neighbors)

            if centrality > max_centrality:
                max_centrality = centrality
                selected_node = node

        # Ajouter le nœud sélectionné et le supprimer du graphe
        if selected_node is not None:
            vertex_cover.append(selected_node)
            G.remove_node(selected_node)

    return vertex_cover