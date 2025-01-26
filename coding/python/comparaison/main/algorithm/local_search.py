from .approximate_matching import approximate_matching
from .utils import is_valid_cover

def local_search(graph, initial_solution=None, max_iterations=1000):
    """
    Métaheuristique de recherche locale :
      - On part d'une solution initiale (ex : approximate_matching).
      - On tente de l'améliorer en retirant des sommets inutiles.
      - On tente aussi des swaps sommet_dans_C <-> sommet_hors_C.
    """
    # 1) Solution de départ
    if initial_solution is None:
        initial_solution = approximate_matching(graph)
    best_C = set(initial_solution)
    best_size = len(best_C)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        improved = False

        # 2) Essayer d'enlever des sommets inutiles
        for node in list(best_C):
            new_C = best_C - {node}
            if is_valid_cover(new_C, graph):
                best_C = new_C
                best_size = len(best_C)
                improved = True
                break

        if improved:
            continue  # on repart pour une nouvelle itération

        # 3) Essayer des swaps (un dedans <-> un dehors)
        all_nodes = set(graph.nodes())
        outside = all_nodes - best_C
        for node_out in outside:
            for node_in in list(best_C):
                # Swap : on enlève node_in, on ajoute node_out
                new_C = (best_C - {node_in}) | {node_out}
                if is_valid_cover(new_C, graph) and len(new_C) < best_size:
                    best_C = new_C
                    best_size = len(best_C)
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Pas d'amélioration => on s'arrête
            break

    return list(best_C)
