import random
from collections import deque


def bfs_vertex_cover(graph):
    """
    Heuristique basée sur un parcours en largeur (BFS).
    Pour chaque composante connexe, on lance un BFS ;
    lorsqu'on visite une arête (u,v), on ajoute 'u' (ou 'v') au cover.
    """
    C = set()
    visited = set()

    # On mélange les nœuds pour éviter un ordre déterministe.
    nodes = list(graph.nodes())
    random.shuffle(nodes)

    for start in nodes:
        if start not in visited:
            queue = deque([start])
            visited.add(start)

            while queue:
                u = queue.popleft()
                for v in graph.neighbors(u):
                    if v not in visited:
                        # Ici, on couvre l’arête (u, v) en ajoutant 'u'
                        # (on aurait pu choisir 'v' au lieu de 'u', c’est une variante possible).
                        C.add(u)
                        visited.add(v)
                        queue.append(v)

    return list(C)
