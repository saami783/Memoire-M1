import random
from .utils import is_valid_cover

def aco(graph, ant_count=20, iterations=50, alpha=1.0, beta=2.0, rho=0.1):
    nodes = list(graph.nodes())
    n = len(nodes)
    degree = dict(graph.degree())
    tau = {node: 1.0 for node in nodes}  # Initialisation uniforme pour encourager l'exploration

    def fitness(C):
        if not is_valid_cover(C, graph):
            return float('inf')
        return len(C)

    best_cover = set(nodes)
    best_score = fitness(best_cover)

    for _ in range(iterations):
        ant_solutions = []

        # Construction des solutions par les fourmis
        sum_pheromones = sum((tau[v] ** alpha) * (degree[v] ** beta) for v in nodes)
        if sum_pheromones == 0:
            sum_pheromones = 1e-10

        for _ in range(ant_count):
            cover = set()
            # Sélection probabiliste avec une probabilité ajustée
            for v in nodes:
                prob = ((tau[v] ** alpha) * (degree[v] ** beta)) / sum_pheromones
                if random.random() < prob:
                    cover.add(v)
            # Ajout d'une étape de réparation pour garantir la validité
            uncovered_edges = [e for e in graph.edges() if e[0] not in cover and e[1] not in cover]
            for u, v in uncovered_edges:
                # Choisir le nœud avec le plus haut degré/phéromone
                if degree[u] + tau[u] > degree[v] + tau[v]:
                    cover.add(u)
                else:
                    cover.add(v)
            ant_solutions.append(cover)

        # Filtrer les solutions valides
        valid_solutions = [sol for sol in ant_solutions if is_valid_cover(sol, graph)]

        # Évaporation des phéromones
        for v in nodes:
            tau[v] *= (1 - rho)

        # Mise à jour des phéromones seulement si solutions valides
        if valid_solutions:
            best_ant = min(valid_solutions, key=fitness)
            deposit = 1.0 / len(best_ant)
            for v in best_ant:
                tau[v] += deposit

            # Mise à jour de la meilleure solution globale
            current_score = len(best_ant)
            if current_score < best_score:
                best_cover = best_ant.copy()
                best_score = current_score

    return list(best_cover)