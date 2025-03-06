import time
import random
from collections import defaultdict
from .utils import is_valid_cover

# à vérifier !!! ne fonctionne pas.
# https://jair.org/index.php/jair/article/view/10812/25808
# ERROR - Erreur lors du traitement du fichier tree_100_42_7.dimacs : 'Graph' object has no attribute 'vertices'
def numvc(graph, cutoff=10, gamma=100, rho=0.3):
    """Implémentation corrigée de NuMVC conforme au pseudo-code"""
    start_time = time.time()
    vertices = list(graph.vertices)
    edges = list(graph.edges.copy())

    # Structures de données initiales
    adj = defaultdict(set)
    edge_weights = defaultdict(int)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_weights[(u, v)] = edge_weights[(v, u)] = 1

    confChange = {v: 1 for v in vertices}
    dscore = {v: 0 for v in vertices}
    last_modified = {v: 0 for v in vertices}

    # Construction gloutonne initiale corrigée
    C = set()
    uncovered_edges = set(edges)
    while uncovered_edges:
        # Sélectionner le sommet avec le meilleur ratio dscore/degre
        candidates = {v: dscore[v] / (len(adj[v]) + 1e-6) for v in vertices if v not in C}
        u = max(candidates, key=candidates.get)
        C.add(u)
        uncovered_edges -= {e for e in uncovered_edges if u in e}

    C_star = set(C)
    iteration = 0

    def update_dscores(nodes):
        for v in nodes:
            dscore[v] = sum(edge_weights[e] for e in adj[v] if e in uncovered_edges)

    update_dscores(vertices)

    while time.time() - start_time < cutoff:
        if not uncovered_edges:
            if len(C) < len(C_star):
                C_star = set(C)

            if C:
                u = max(C, key=lambda x: (-dscore[x], last_modified[x]))
                C.remove(u)
                confChange[u] = 0
                for z in adj[u]:
                    confChange[z] = 1
                update_dscores(adj[u])
                last_modified[u] = iteration
                uncovered_edges.update(e for e in edges if (u in e) and (e[0] not in C and e[1] not in C))
            continue

        # Phase de suppression
        candidates = [v for v in C if confChange[v]] or list(C)
        u = max(candidates, key=lambda x: (-dscore[x], last_modified[x]))
        C.remove(u)
        confChange[u] = 0
        for z in adj[u]:
            confChange[z] = 1
        update_dscores(adj[u])
        last_modified[u] = iteration
        uncovered_edges.update(e for e in edges if (u in e) and (e[0] not in C and e[1] not in C))

        # Phase d'ajout
        e = random.choice(tuple(uncovered_edges))
        candidates = [v for v in e if confChange[v]]
        v = max(candidates, key=lambda x: (-dscore[x], last_modified[x])) if candidates else random.choice(e)

        # Mise à jour des poids AVANT ajout
        for edge in set(uncovered_edges):
            edge_weights[edge] += 1
            edge_weights[(edge[1], edge[0])] += 1

        C.add(v)
        for z in adj[v]:
            confChange[z] = 1
        update_dscores(adj[v])
        last_modified[v] = iteration
        uncovered_edges -= {e for e in uncovered_edges if v in e}

        # Réduction périodique des poids
        avg_weight = sum(edge_weights.values()) / len(edge_weights) if edge_weights else 0
        if avg_weight >= gamma:
            for e in edge_weights:
                edge_weights[e] = int(rho * edge_weights[e])
            update_dscores(vertices)

    assert is_valid_cover(graph, C_star), "Couverture invalide !"
    return C_star