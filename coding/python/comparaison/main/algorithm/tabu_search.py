import random
import math
from .approximate_matching import approximate_matching
from .utils import is_valid_cover


def tabu_search_(graph,
                 initial_solution=None,
                 max_iterations=1000,
                 tabu_tenure=10,
                 neighborhood_size=50):
    """
    Tabu Search pour le Vertex Cover.
    """
    # -- Initialisation
    if initial_solution is None:
        current_C = set(approximate_matching(graph))  # Convert to set here
    else:
        current_C = set(initial_solution)

    best_C = set(current_C)
    best_size = len(best_C)

    tabulist = dict()

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # 1) Générer un ensemble de voisins
        all_nodes = set(graph.nodes())  # set
        inside = list(current_C)  # list
        outside = list(all_nodes - current_C)  # OK, both sides are sets

        neighbors = []
        for _ in range(neighborhood_size):
            if inside:
                node_in = random.choice(inside)
            else:
                node_in = None
            if outside:
                node_out = random.choice(outside)
            else:
                node_out = None

            new_C = set(current_C)
            if node_in is not None:
                new_C.remove(node_in)
            if node_out is not None:
                new_C.add(node_out)

            neighbors.append((node_in, node_out, new_C))

        # 2) Choisir la "meilleure" voisine non tabou ou autorisée
        chosen_move = None
        chosen_cover = None
        best_cover_size = math.inf

        for (node_in, node_out, new_C) in neighbors:
            # Sauter solution non valide
            if not is_valid_cover(new_C, graph):
                continue

            size_new = len(new_C)
            move = (node_in, node_out)

            # Règle d'aspiration
            if move in tabulist and size_new >= best_size:
                continue

            if size_new < best_cover_size:
                best_cover_size = size_new
                chosen_move = move
                chosen_cover = new_C

        if chosen_cover is None:
            # Aucun voisin "améliorant" ou permis => on continue
            continue

        # 3) Mise à jour de la solution courante
        current_C = chosen_cover

        # 4) Mise à jour de la meilleure solution
        if len(current_C) < best_size:
            best_size = len(current_C)
            best_C = set(current_C)

        # 5) Liste tabou
        inv_move = (chosen_move[1], chosen_move[0])
        tabulist[inv_move] = tabu_tenure

        # 6) Décrémenter le compteur
        moves_to_remove = []
        for m in tabulist:
            tabulist[m] -= 1
            if tabulist[m] <= 0:
                moves_to_remove.append(m)
        for m in moves_to_remove:
            del tabulist[m]

    return list(best_C)
