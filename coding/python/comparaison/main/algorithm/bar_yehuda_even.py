import random

def bar_yehuda_even_vertex_cover(vertices, edges, weights):
    """
    Implémente l'algorithme de Bar-Yehuda et Even pour le problème de la couverture de sommets pondéré.

    Paramètres:
      - vertices : ensemble (ou liste) des sommets du graphe.
      - edges : liste d'arêtes, chaque arête est un tuple (u, v).
      - weights : dictionnaire donnant le poids de chaque sommet, par exemple {v: poids}.
                 Pour un graphe non pondéré, utiliser 1 pour tous les sommets.

    Retour:
      - cover : ensemble de sommets formant une couverture (vertex cover) approchée.
    """
    # Initialisation des marges duales : d(v) = poids(v)
    d = {v: weights[v] for v in vertices}
    # Initialisation des variables duales pour chaque arête : y(e) = 0
    y = {e: 0 for e in edges}
    # Ensemble qui contiendra les sommets choisis dans la couverture
    cover = set()
    # Ensemble des arêtes non encore couvertes
    uncovered_edges = set(edges)
    while uncovered_edges:
        # On sélectionne arbitrairement une arête non couverte
        u, v = uncovered_edges.pop()
        # Si l'arête est déjà couverte par un sommet dans cover, on passe à la suivante
        if u in cover or v in cover:
            continue
        # On détermine la quantité maximale que l'on peut augmenter y(e) sans violer les contraintes
        delta = min(d[u], d[v])
        y[(u, v)] = delta
        # Mise à jour des marges duales pour u et v
        d[u] -= delta
        d[v] -= delta
        # Si la marge d'un sommet tombe à 0, ce sommet devient « tendu » et est ajouté à la couverture
        if d[u] == 0:
            cover.add(u)
        if d[v] == 0:
            cover.add(v)
        # Mise à jour de l'ensemble des arêtes non couvertes :
        # on retire toutes les arêtes qui sont désormais couvertes par un sommet de cover.
        uncovered_edges = {edge for edge in uncovered_edges if edge[0] not in cover and edge[1] not in cover}
    return cover


# Exemple d'utilisation
if __name__ == "__main__":
    # Définition d'un graphe simple
    # Sommets
    vertices = {1, 2, 3, 4}
    # Liste d'arêtes (chaque arête est un tuple (u, v))
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)]
    # Pour un graphe non pondéré, on attribue un poids de 1 à chaque sommet
    weights = {v: 1 for v in vertices}
    cover = bar_yehuda_even_vertex_cover(vertices, edges, weights)
    print("Vertex Cover :", cover)