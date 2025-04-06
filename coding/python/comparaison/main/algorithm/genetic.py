import numpy as np
import numba as nb
from tqdm import tqdm  # Pour la barre de progression


@nb.njit(nogil=True, cache=True)
def compute_fitness(population, edges):
    n_individuals = population.shape[0]
    fitness = np.zeros((n_individuals, 2), dtype=np.int64)

    for i in nb.prange(n_individuals):
        cover = population[i]
        uncovered = 0
        # Vérifier toutes les arêtes
        for j in range(edges.shape[0]):
            u, v = edges[j]
            if not (cover[u] or cover[v]):
                uncovered += 1
        # Pénalité exponentielle pour les solutions invalides
        fitness[i, 0] = uncovered
        fitness[i, 1] = cover.sum() + (uncovered * 1000000 if uncovered > 0 else 0)

    return fitness


@nb.njit(nogil=True)
def two_point_crossover(parent1, parent2):
    n = len(parent1)
    child = np.zeros(n, dtype=np.uint8)
    crossover_points = np.sort(np.random.choice(n, 2, replace=False))
    child[:crossover_points[0]] = parent1[:crossover_points[0]]
    child[crossover_points[0]:crossover_points[1]] = parent2[crossover_points[0]:crossover_points[1]]
    child[crossover_points[1]:] = parent1[crossover_points[1]:]
    return child


def remove_redundant_nodes(cover, edges):
    nodes = list(np.where(cover)[0])
    for node in sorted(nodes, reverse=True):
        temp = [n for n in nodes if n != node]
        if is_valid_cover(temp, edges):
            nodes = temp
    new_cover = np.zeros_like(cover)
    new_cover[nodes] = 1
    return new_cover


def is_valid_cover(cover_nodes, edges):
    for u, v in edges:
        if u not in cover_nodes and v not in cover_nodes:
            return False
    return True


def genetic(graph, pop_size=500, generations=1000, mutation_rate=0.05, elite_ratio=0.1, verbose=False):
    # Mapping des nœuds
    nodes = sorted(graph.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # Conversion des arêtes
    edges = np.array([(node_map[u], node_map[v]) for u, v in graph.edges()], dtype=np.int64)

    # Initialisation de la population
    population = np.random.randint(2, size=(pop_size, n_nodes), dtype=np.uint8)
    elite_size = max(1, int(pop_size * elite_ratio))
    best_solution = None
    best_size = np.inf

    for gen in tqdm(range(generations), disable=not verbose):
        # Évaluation
        fitness = compute_fitness(population, edges)

        # Mise à jour de la meilleure solution
        valid_mask = fitness[:, 0] == 0
        if np.any(valid_mask):
            valid_fitness = fitness[valid_mask]
            best_idx = np.argmin(valid_fitness[:, 1])
            current_size = valid_fitness[best_idx, 1]

            if current_size < best_size:
                best_size = current_size
                best_solution = population[valid_mask][best_idx].copy()

        # Sélection élitiste
        elite_indices = np.lexsort((fitness[:, 1], fitness[:, 0]))[:elite_size]
        elite = population[elite_indices]

        # Sélection par tournoi
        parents = []
        for _ in range(pop_size - elite_size):
            contestants = np.random.choice(pop_size, 5, replace=False)
            winner = contestants[np.argmin(fitness[contestants, 0])]
            parents.append(population[winner])
        parents = np.array(parents)

        # Croisement en deux points
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                offspring.append(parents[i])
                continue
            child1 = two_point_crossover(parents[i], parents[i + 1])
            child2 = two_point_crossover(parents[i + 1], parents[i])
            offspring.extend([child1, child2])
        offspring = np.array(offspring[:pop_size - elite_size])

        # Mutation bit-flip
        mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
        offspring = np.where(mutation_mask, 1 - offspring, offspring)

        population = np.vstack((elite, offspring))

    # Post-optimisation locale
    if best_size != np.inf:
        optimized_cover = remove_redundant_nodes(best_solution, edges)
        if is_valid_cover(np.where(optimized_cover)[0], edges):
            best_solution = optimized_cover
            best_size = optimized_cover.sum()
        else:
            best_size = np.inf

    if best_size != np.inf:
        return [nodes[i] for i in np.where(best_solution)[0]], best_size
    else:
        return [], np.inf


def run_multiple_times(graph, runs=10, **kwargs):
    results = []
    for _ in range(runs):
        _, size = genetic(graph, **kwargs)
        if size != np.inf:
            results.append(size)

    return {
        'average_size': np.mean(results) if results else np.inf,
        'best_size': np.min(results) if results else np.inf,
        'worst_size': np.max(results) if results else np.inf
    }
