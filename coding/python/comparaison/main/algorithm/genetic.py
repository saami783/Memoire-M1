import numpy as np
import random

# https://github.com/MKaczkow/genetic_algo/tree/main
def initiate_population(pop_size, num_vertices):
    """Génère une population initiale aléatoire."""
    return np.random.randint(2, size=(pop_size, num_vertices))


def evaluate(population, graph):
    """Évalue la population et retourne le meilleur couvert valide."""
    best_member = np.ones(population.shape[1], dtype=int)
    best_size = population.shape[1]

    for member in population:
        cover = [i for i, val in enumerate(member) if val == 1]
        temp_graph = graph.copy()
        temp_graph.delete_vertices(cover)

        if temp_graph.ecount() == 0 and sum(member) < best_size:
            best_member = member
            best_size = sum(member)

    return best_member


def tournament_selection(population, graph):
    """Sélection par tournoi entre deux individus."""
    new_pop = np.empty_like(population)

    for i in range(len(new_pop)):
        # Sélection de deux candidats aléatoires
        a, b = random.sample(range(len(population)), 2)
        candidate1 = population[a]
        candidate2 = population[b]

        # Évaluation des candidats
        def eval_candidate(cand):
            cover = [i for i, val in enumerate(cand) if val == 1]
            g = graph.copy()
            g.delete_vertices(cover)
            return (g.ecount(), sum(cand))

        score1 = eval_candidate(candidate1)
        score2 = eval_candidate(candidate2)

        # Sélection du meilleur
        if score1 < score2 or (score1 == score2 and sum(candidate1) < sum(candidate2)):
            new_pop[i] = candidate1
        else:
            new_pop[i] = candidate2

    return new_pop


def mutate(population, mutation_prob=0.01):
    """Applique des mutations aléatoires à la population."""
    mutated = population.copy()
    num_genes = mutated.shape[1]

    for i in range(len(mutated)):
        if random.random() < mutation_prob:
            gene = random.randint(0, num_genes - 1)
            mutated[i, gene] = 1 - mutated[i, gene]

            if random.random() < mutation_prob ** 2:
                mutated[i, (gene - 1) % num_genes] = 1 - mutated[i, (gene - 1) % num_genes]

    return mutated


def evolve(graph, population, max_generations=100, mutation_prob=0.01):
    """Exécute l'algorithme génétique et retourne le meilleur couvert."""
    for _ in range(max_generations):
        population = tournament_selection(population, graph)
        population = mutate(population, mutation_prob)

    best_solution = evaluate(population, graph)
    return [i for i, val in enumerate(best_solution) if val == 1]