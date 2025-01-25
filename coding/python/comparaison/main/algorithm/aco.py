import random


def aco(graph,
                     ant_count=20,
                     iterations=50,
                     alpha=1.0,
                     rho=0.1):
    """
    Algorithme de Colonies de Fourmis (ACO) pour Vertex Cover (simplifié).

    - "ant_count" fourmis construisent chacune une solution.
    - "iterations" itérations globales.
    - alpha : puissance de la phéromone lors de la décision
    - rho : taux d'évaporation (0 < rho < 1)

    Hypothèse: on associe une phéromone par node.
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    # initialisons la phéromone
    # tau[i] = la phéromone associée au node i
    tau = {node: 1.0 for node in nodes}  # on démarre à 1.0

    def is_valid_cover(C):
        for (u, v) in graph.edges():
            if u not in C and v not in C:
                return False
        return True

    def fitness(C):
        if not is_valid_cover(C):
            return 10_000 + len(C)
        return len(C)

    best_cover = set(nodes)  # On part du pire cas : tous les sommets
    best_score = fitness(best_cover)

    for _ in range(iterations):
        ant_solutions = []

        # Chaque fourmi construit une solution
        for _ant in range(ant_count):
            cover = set()
            # On décide pour chaque node de l'inclure ou pas
            # Probabilité ~ (tau[node]^alpha) / somme(tau[node]^alpha)
            # simple version: on va normaliser
            sum_pheromones = sum(tau[v] ** alpha for v in nodes)

            for v in nodes:
                p = (tau[v] ** alpha) / sum_pheromones
                if random.random() < p:
                    cover.add(v)

            ant_solutions.append(cover)

        # Évaporation
        for v in nodes:
            tau[v] = (1 - rho) * tau[v]

        # Mise à jour de la phéromone
        # Ex: on récompense les meilleures solutions
        # On peut choisir de ne renforcer que la meilleure fourmi
        best_ant = min(ant_solutions, key=fitness)
        if fitness(best_ant) < best_score:
            best_cover = best_ant
            best_score = fitness(best_cover)

        # Renforcement de la phéromone pour chaque node dans la meilleure solution
        # par ex: += 1 / size_of_solution
        if is_valid_cover(best_ant):
            deposit = 1.0 / len(best_ant)
            for v in best_ant:
                tau[v] += deposit

    return list(best_cover)
