import random


def vns_vertex_cover(graph,
                     initial_solution=None,
                     max_iterations=100,
                     max_k=3):
    """
    Variable Neighborhood Search pour Vertex Cover.

    Paramètres:
    -----------
    graph : graphe NetworkX
    initial_solution : liste ou set de sommets de départ
    max_iterations : nombre max d'itérations (arrêts globaux)
    max_k : taille maximale du voisinage "k" (nb de sommets que l'on modifie)

    Stratégie simplifiée:
      - On part d'une solution initiale.
      - k = 1
      - Tant qu'on n'a pas atteint max_iterations:
         1) On "secoue" la solution courante en faisant k modifications (add/remove).
         2) On applique une recherche locale (greedy ou local_search).
         3) Si on améliore la meilleure solution, on remet k=1, sinon k=k+1
         4) Si k > max_k, on arrête ou on remet k=1
    """

    def approximate_matching_vertex_cover(g):
        C = set()
        temp_g = g.copy()
        while temp_g.number_of_edges() > 0:
            (u, v) = random.choice(list(temp_g.edges()))
            C.add(u)
            temp_g.remove_node(u)
        return C

    def is_valid_cover(C_set):
        for (u, v) in graph.edges():
            if u not in C_set and v not in C_set:
                return False
        return True

    # Petite recherche locale (on peut réutiliser local_search_vertex_cover si on veut)
    def local_search_simple(g, cover_set):
        # On essaie d'enlever un sommet inutile tant qu'on peut
        improved = True
        C = set(cover_set)
        while improved:
            improved = False
            for node in list(C):
                Ctemp = C - {node}
                if is_valid_cover(Ctemp):
                    C = Ctemp
                    improved = True
                    break
        return C

    # -- Init
    if initial_solution is None:
        current_C = approximate_matching_vertex_cover(graph)
    else:
        current_C = set(initial_solution)

    best_C = set(current_C)
    best_size = len(best_C)

    iteration = 0
    k = 1

    while iteration < max_iterations:
        iteration += 1
        if k > max_k:
            # On peut relancer k = 1 ou bien arrêter
            break

        # 1) "Shake" : on crée une solution voisine en faisant k modifications
        #    Ex: on enlève k sommets de la couverture et on en ajoute k sommets hors couverture
        all_nodes = set(graph.nodes())
        inside = list(current_C)
        outside = list(all_nodes - current_C)

        new_C = set(current_C)

        # enlever k sommets (au hasard) si possible
        for _ in range(k):
            if new_C:
                node_in = random.choice(list(new_C))
                new_C.remove(node_in)

        # ajouter k sommets (au hasard)
        for _ in range(k):
            candidate_outside = list(all_nodes - new_C)
            if candidate_outside:
                node_out = random.choice(candidate_outside)
                new_C.add(node_out)

        # 2) Recherche locale sur la solution "secouée"
        new_C = local_search_simple(graph, new_C)

        # 3) On compare
        if is_valid_cover(new_C) and len(new_C) < best_size:
            best_size = len(new_C)
            best_C = new_C
            current_C = new_C
            # On reset le voisinage
            k = 1
        else:
            # On augmente le voisinage
            k += 1

    return list(best_C)
