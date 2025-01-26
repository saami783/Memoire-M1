import random
from .utils import is_valid_cover, local_search_simple
from .approximate_matching import approximate_matching

def ils(graph,
                     initial_solution=None,
                     max_iterations=100,
                     perturbation_strength=2):
    """
    Iterated Local Search pour Vertex Cover.

    Paramètres :
    -----------
    graph : graphe NetworkX
    initial_solution : solution de départ
    max_iterations : nb d'itérations globales
    perturbation_strength : nb de sommets (dans C) à enlever et nb (hors C) à ajouter pour perturber
    """

    # -- Init
    if initial_solution is None:
        current_C = approximate_matching(graph)
    else:
        current_C = set(initial_solution)

    # On descend dans le minimum local
    current_C = local_search_simple(graph, current_C, graph)

    best_C = set(current_C)

    for _ in range(max_iterations):
        # 1) Perturbation de la solution courante
        #    On enlève "perturbation_strength" sommets de current_C
        #    et on ajoute "perturbation_strength" sommets hors C
        all_nodes = set(graph.nodes())
        inside = list(current_C)
        outside = list(all_nodes - current_C)

        new_C = set(current_C)

        # enlever perturbation_strength sommets (si possible)
        for _ in range(perturbation_strength):
            if new_C:
                node_in = random.choice(list(new_C))
                new_C.remove(node_in)

        # ajouter perturbation_strength sommets (si possible)
        for _ in range(perturbation_strength):
            candidate_outside = list(all_nodes - new_C)
            if candidate_outside:
                node_out = random.choice(candidate_outside)
                new_C.add(node_out)

        # 2) Recherche locale depuis la solution perturbée
        new_C = local_search_simple(graph, new_C)

        # 3) Si on améliore, on met à jour
        if len(new_C) < len(best_C) and is_valid_cover(new_C, graph):
            best_C = new_C

        # On adopte new_C comme solution courante même si elle n'est pas meilleure
        # (c'est le principe ILS: on bouge de minimum local en minimum local)
        current_C = new_C

    return list(best_C)
