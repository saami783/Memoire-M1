import random
from .approximate_matching import approximate_matching
from .utils import is_valid_cover, local_search_simple

def vns(graph,
        initial_solution=None,
        max_iterations=100,
        max_k=3):
    """
    Variable Neighborhood Search pour Vertex Cover.
    """
    # -- Init
    if initial_solution is None:
        # approximate_matching renvoie une liste => on la convertit en set
        current_C = set(approximate_matching(graph))
    else:
        current_C = set(initial_solution)

    best_C = set(current_C)
    best_size = len(best_C)

    iteration = 0
    k = 1

    while iteration < max_iterations:
        iteration += 1
        if k > max_k:
            break

        # 1) "Shake" : on enlève k sommets et on en ajoute k
        all_nodes = set(graph.nodes())
        new_C = set(current_C)

        for _ in range(k):
            if new_C:
                node_in = random.choice(list(new_C))
                new_C.remove(node_in)

        for _ in range(k):
            candidate_outside = list(all_nodes - new_C)
            if candidate_outside:
                node_out = random.choice(candidate_outside)
                new_C.add(node_out)

        # 2) Recherche locale
        new_C = local_search_simple(new_C, graph)

        # 3) Compare
        if is_valid_cover(new_C, graph) and len(new_C) < best_size:
            best_size = len(new_C)
            best_C = new_C
            current_C = new_C
            k = 1
        else:
            k += 1

    return list(best_C)
