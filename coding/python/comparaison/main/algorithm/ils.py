import random
from .utils import is_valid_cover, local_search_simple
from .approximate_matching import approximate_matching

def ils(graph,
        initial_solution=None,
        max_iterations=100,
        perturbation_strength=2):
    """
    Iterated Local Search pour Vertex Cover.

    Paramètres:
    -----------
    graph : graphe NetworkX
    initial_solution : solution de départ (liste ou set de sommets)
    max_iterations : nb d'itérations globales
    perturbation_strength : nb de sommets (dans C) à enlever et nb (hors C) à ajouter pour perturber
    """

    # -- 1) Solution initiale
    if initial_solution is None:
        current_C = approximate_matching(graph)
    else:
        current_C = set(initial_solution)

    # Convertir la solution en set si ce n'est pas déjà le cas
    current_C = set(current_C)

    # -- 2) Descendre dans le minimum local
    # Important : ordre = (cover_set, graph)
    current_C = local_search_simple(current_C, graph)

    best_C = set(current_C)

    # -- 3) Boucle principale ILS
    for _ in range(max_iterations):
        # 3a) Perturbation de la solution courante
        all_nodes = set(graph.nodes())
        new_C = set(current_C)

        # Enlever 'perturbation_strength' sommets (au hasard) du cover
        for _ in range(perturbation_strength):
            if new_C:
                node_in = random.choice(list(new_C))
                new_C.remove(node_in)

        # Ajouter 'perturbation_strength' sommets (au hasard) hors cover
        for _ in range(perturbation_strength):
            candidate_outside = list(all_nodes - new_C)
            if candidate_outside:
                node_out = random.choice(candidate_outside)
                new_C.add(node_out)

        # 3b) Recherche locale depuis la solution perturbée
        new_C = local_search_simple(new_C, graph)

        # 3c) Si on améliore, on met à jour
        if len(new_C) < len(best_C) and is_valid_cover(new_C, graph):
            best_C = new_C

        # 3d) On adopte new_C comme solution courante (ILS)
        current_C = new_C

    # Retourne la meilleure solution trouvée, sous forme de liste
    return list(best_C)
