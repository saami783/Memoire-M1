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

    Paramètres:
    -----------
    graph : un graphe NetworkX
    initial_solution : liste (ou set) de sommets pour démarrer
    max_iterations : nb max d'itérations de la TS
    tabu_tenure : nb d'itérations pendant lesquelles un 'mouvement' reste tabou
    neighborhood_size : nb de voisins que l'on va évaluer à chaque itération
    """

    # -- Initialisation
    if initial_solution is None:
        current_C = approximate_matching(graph) # Pour générer une solution initiale si besoin
    else:
        current_C = set(initial_solution)

    best_C = set(current_C)
    best_size = len(best_C)

    # La liste tabou : on y stocke des "mouvements" (node_in, node_out)
    # avec un compteur de temps restant avant de libérer ce mouvement.
    # On peut stocker ça dans un dict : tabulist[(node_in, node_out)] = nb_iters_restants
    tabulist = dict()

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # 1) Générer un ensemble de voisins
        #    Par exemple: on fait "swap" (enlève 1 sommet de current_C et ajoute 1 sommet hors_C)
        #    On en sélectionne un certain nombre (neighborhood_size) au hasard.
        all_nodes = set(graph.nodes())
        inside = list(current_C)
        outside = list(all_nodes - current_C)

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
            # On saute les solutions non valides
            if not is_valid_cover(new_C, graph):
                continue

            # Calcul du coût
            size_new = len(new_C)

            # Le mouvement (node_in, node_out)
            move = (node_in, node_out)

            # Vérifier si c'est tabou
            # Règle d'aspiration: si c'est mieux que la meilleure solution qu'on ait jamais eue,
            # on l'autorise même si c'est tabou.
            if move in tabulist and size_new >= best_size:
                # Mouvement tabou, et pas d'aspiration possible
                continue

            # Si c'est mieux que tout ce qu'on a vu dans cette itération
            if size_new < best_cover_size:
                best_cover_size = size_new
                chosen_move = move
                chosen_cover = new_C

        if chosen_cover is None:
            # Aucun voisin "améliorant" ou permis => on peut sortir ou ignorer
            # On peut ignorer et continuer
            continue

        # 3) Mettre à jour la solution courante avec la voisine choisie
        current_C = chosen_cover

        # 4) Mettre à jour la meilleure solution si on améliore
        if len(current_C) < best_size:
            best_size = len(current_C)
            best_C = set(current_C)

        # 5) Mettre à jour la liste tabou
        #    On ajoute l'inverse du move choisi => le move inverse est (node_out, node_in)
        inv_move = (chosen_move[1], chosen_move[0])
        tabulist[inv_move] = tabu_tenure

        # 6) Décrémenter le compteur de tabous existants et retirer ceux qui arrivent à 0
        moves_to_remove = []
        for m in tabulist:
            tabulist[m] -= 1
            if tabulist[m] <= 0:
                moves_to_remove.append(m)
        for m in moves_to_remove:
            del tabulist[m]

    return list(best_C)
