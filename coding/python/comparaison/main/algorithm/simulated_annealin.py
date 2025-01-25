import math
import random


def simulated_annealing_vertex_cover(graph,
                                     initial_solution=None,
                                     max_iterations=1000,
                                     initial_temp=10.0,
                                     cooling_rate=0.99):
    """
    Métaheuristique du recuit simulé pour le Vertex Cover.

    - initial_solution : liste de sommets (facultatif)
    - max_iterations : nombre max d'itérations
    - initial_temp : température initiale
    - cooling_rate : facteur de refroidissement (entre 0 et 1)
    """

    def approximate_matching_vertex_cover(g):
        C = set()
        tmp_g = g.copy()
        while tmp_g.number_of_edges() > 0:
            u, v = random.choice(list(tmp_g.edges()))
            C.add(u)
            tmp_g.remove_node(u)
        return list(C)

    def is_valid_cover(C_set):
        for (u, v) in graph.edges():
            if u not in C_set and v not in C_set:
                return False
        return True

    # Solution de départ
    if initial_solution is None:
        current_C = set(approximate_matching_vertex_cover(graph))
    else:
        current_C = set(initial_solution)

    best_C = set(current_C)
    temperature = initial_temp

    for iteration in range(max_iterations):
        # Générer une solution voisine :
        #   * soit enlever un sommet si c'est toujours valide,
        #   * soit swap (un dedans, un dehors),
        #   * ou ajouter un sommet aléatoire pour tenter un autre chemin...

        # Exemple simple : swap
        all_nodes = set(graph.nodes())
        inside = list(current_C)
        outside = list(all_nodes - current_C)

        if outside:  # pour éviter l'erreur si current_C == tous les nœuds
            node_out = random.choice(outside)
            node_in = random.choice(inside) if inside else None

            new_C = set(current_C)
            if node_in is not None:
                new_C.remove(node_in)
            new_C.add(node_out)
        else:
            new_C = set(current_C)  # cas limite

        # Vérifier si la nouvelle solution est valide
        if is_valid_cover(new_C):
            # Calculer la différence de "coût" (taille)
            diff = len(new_C) - len(current_C)

            if diff < 0:
                # Nouvelle solution meilleure => on l'adopte
                current_C = new_C
            else:
                # Moins bonne => on l'accepte avec prob. e^(-diff / T)
                accept_prob = math.exp(-diff / temperature)
                if random.random() < accept_prob:
                    current_C = new_C

            # Mettre à jour la meilleure solution
            if len(current_C) < len(best_C):
                best_C = set(current_C)

        # Diminuer la température
        temperature *= cooling_rate

        # Optionnel : si la température est trop basse, on peut arrêter
        if temperature < 1e-5:
            break

    return list(best_C)
