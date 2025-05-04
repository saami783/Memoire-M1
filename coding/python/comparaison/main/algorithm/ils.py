import random

# https://www.sciencedirect.com/science/article/pii/S0304397511000363
def ils(graph, max_iter=10000):
    """
    - Commence avec tous les nœuds dans la couverture.
    - Retire les nœuds redondants ou effectue des échanges d'arêtes.
    """
    cover = set(graph.nodes())

    for _ in range(max_iter):
        # Étape 1 : Suppression des nœuds redondants (tous voisins couverts)
        redundant = [u for u in cover if all(v in cover for v in graph.neighbors(u))]

        if redundant:
            node = random.choice(redundant)
            cover.remove(node)
        else:
            # Étape 2 : Échange d'arêtes
            edges = list(graph.edges())
            if not edges:
                break  # Toutes les arêtes sont couvertes
            u, v = random.choice(edges)

            # Vérifier si u est dans le cover et v non, ou vice versa
            if u in cover and v not in cover:
                # Valider l'échange : tous les voisins de u (sauf v) sont dans le cover
                if all(n in cover for n in graph.neighbors(u) if n != v):
                    cover.remove(u)
                    cover.add(v)
            elif v in cover and u not in cover:
                # Valider l'échange : tous les voisins de v (sauf u) sont dans le cover
                if all(n in cover for n in graph.neighbors(v) if n != u):
                    cover.remove(v)
                    cover.add(u)

    return list(cover)