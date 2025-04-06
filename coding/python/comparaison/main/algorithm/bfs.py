from collections import deque


def bfs(graph):
    """
    Trouve une couverture de sommets approximative en utilisant BFS.
    Retourne une liste de nœuds formant la couverture.
    """
    C = []
    uncovered_edges = {frozenset(edge) for edge in graph.edges()}
    visited = set()

    for start_node in graph.nodes():
        if start_node not in visited:
            queue = deque([start_node])

            while queue:
                u = queue.popleft()
                if u in visited:
                    continue
                visited.add(u)

                for v in graph.neighbors(u):
                    edge = frozenset({u, v})
                    if edge in uncovered_edges:
                        C.append(u)
                        C.append(v)

                        # Supprime toutes les arêtes incidentes à u ou v
                        for node in [u, v]:
                            for neighbor in graph.neighbors(node):
                                e = frozenset({node, neighbor})
                                if e in uncovered_edges:
                                    uncovered_edges.remove(e)

                        visited.add(v)
                        break

                # Ajoute les voisins non visités à la file
                for v in graph.neighbors(u):
                    if v not in visited:
                        queue.append(v)

    return C