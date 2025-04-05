import numpy as np
import numba as nb

def debug_edges(graph, edges_array, node_map):
    print("=== DEBUG EDGES ===")
    print("Node mapping:", node_map)
    print("Nombre de nœuds dans le graphe:", len(graph.nodes()))

    original_edges = list(graph.edges())[:5]
    converted_edges = edges_array[:5]

    print("\nOriginal vs Converted (premières 5 arêtes):")
    for (u_orig, v_orig), (u_conv, v_conv) in zip(original_edges, converted_edges):
        print(f"({u_orig},{v_orig}) -> ({u_conv},{v_conv})")

    for u, v in graph.edges():
        try:
            if node_map[u] not in range(len(graph)) or node_map[v] not in range(len(graph)):
                raise ValueError(f"Indice hors limites: ({u},{v}) -> ({node_map[u]},{node_map[v]})")
        except KeyError as e:
            print(f"ERREUR CRITIQUE: Nœud {e} non trouvé dans le mapping!")
            raise

@nb.njit(nogil=True, cache=True)
def compute_fitness(population, edges):
    n_individuals = population.shape[0]
    fitness = np.zeros((n_individuals, 2), dtype=np.int64)
    for i in nb.prange(n_individuals):
        cover = population[i]
        uncovered = 0
        for j in range(edges.shape[0]):
            u, v = edges[j]
            if not (cover[u] or cover[v]):
                uncovered += 1
        fitness[i, 0] = uncovered
        fitness[i, 1] = cover.sum()
    return fitness

def genetic(graph, pop_size=100, generations=100, mutation_rate=0.01, elite_ratio=0.1, verbose=False):
    nodes = sorted(graph.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    edges = []
    for u, v in graph.edges():
        try:
            edges.append((node_map[u], node_map[v]))
        except KeyError as e:
            raise ValueError(f"Arête ({u},{v}) contient un nœud non répertorié: {e}")
    edges = np.array(edges, dtype=np.int64)

    if verbose:
        debug_edges(graph, edges, node_map)
        print("\nVérification finale:")
        print("Nombre d'arêtes converties:", len(edges))
        print("Indice max:", np.max(edges) if len(edges) > 0 else 0)
        print("Taille de la couverture optimale connue:", graph.graph.get('optimal_size', 'Inconnue'))

    population = np.random.randint(2, size=(pop_size, n_nodes), dtype=np.uint8)
    elite_size = max(1, int(pop_size * elite_ratio))
    best_solution = None
    best_size = np.inf

    for gen in range(generations):
        fitness = compute_fitness(population, edges)

        valid_indices = np.where(fitness[:, 0] == 0)[0]
        if valid_indices.size > 0:
            best_idx = valid_indices[np.argmin(fitness[valid_indices, 1])]
            current_size = fitness[best_idx, 1]

            if current_size < best_size:
                best_size = current_size
                best_solution = population[best_idx].copy()
                if verbose:
                    print(f"Génération {gen}: Nouvelle meilleure taille = {best_size}")

        # Sélection élitiste
        elite_indices = np.lexsort((fitness[:, 1], fitness[:, 0]))[:elite_size]
        elite = population[elite_indices]

        tournament_size = 5
        tournaments = np.random.randint(low=0, high=pop_size, size=(pop_size - elite_size, tournament_size))
        tournament_fitness = fitness[tournaments, 0]
        winners = np.argmin(tournament_fitness, axis=1)
        selected_indices = tournaments[np.arange(tournaments.shape[0]), winners]
        parents = population[selected_indices]

        crossover_mask = np.random.randint(2, size=parents.shape, dtype=np.uint8)
        parents_shuffled = np.random.permutation(parents)
        offspring = (parents * crossover_mask) + (parents_shuffled * (1 - crossover_mask))

        mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
        offspring ^= mutation_mask.astype(np.uint8)

        population = np.vstack((elite, offspring))

    if best_size != np.inf:
        cover = np.where(best_solution)[0]
        edge_coverage = 0
        for u, v in edges:
            if u in cover or v in cover:
                edge_coverage += 1
        coverage_ratio = edge_coverage / len(edges)

        if verbose:
            print(f"\nSolution finale: {len(cover)} nœuds")
            print(f"Couverture réelle: {coverage_ratio * 100:.2f}% des arêtes")

        if coverage_ratio == 1.0:
            return [nodes[i] for i in cover], best_size
        else:
            print("Avertissement: Solution invalide! (couverture incomplète)")
            return [], 0
    else:
        return [], 0
