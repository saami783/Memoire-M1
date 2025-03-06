import random

# https://www.sciencedirect.com/science/article/pii/S0304397511000363
def ks_vc(graph):
    """
    Implémentation de l'algorithme KS-VC pour Vertex Cover.
    - Supprime itérativement les feuilles et leurs voisins.
    - Si aucune feuille, supprime un nœud aléatoire.
    """
    G = graph.copy()
    cover = set()

    while G.number_of_edges() > 0:
        # Étape 1 : Trouver toutes les feuilles (degré = 1)
        leaves = [node for node in G.nodes() if G.degree(node) == 1]

        if leaves:
            # Choisir une feuille aléatoire et son voisin
            leaf = random.choice(leaves)
            neighbor = next(iter(G.neighbors(leaf)))
            cover.add(neighbor)
            # Supprimer le voisin et la feuille du graphe
            G.remove_nodes_from([leaf, neighbor])
        else:
            # Étape 2 : Aucune feuille → choisir un nœud aléatoire
            node = random.choice(list(G.nodes()))
            cover.add(node)
            G.remove_node(node)

    return list(cover)