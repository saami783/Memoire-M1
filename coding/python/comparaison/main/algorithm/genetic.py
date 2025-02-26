# @todo retourne des solutions en dessous de la taille optimale

import random
from .utils import is_valid_cover

def genetic(graph, population_size=30, generations=100, mutation_prob=0.1):
    """
    Chaque individu est représenté par un ensemble de sommets.
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    def fitness(cover_set):
        # On privilégie la taille la plus petite, tout en imposant
        # une "pénalité" énorme si la couverture n'est pas valide
        if not is_valid_cover(cover_set, graph):
            return 10_000 + len(cover_set)
        else:
            return len(cover_set)

    # --- Générer la population initiale ---
    population = []
    for _ in range(population_size):
        # Version simple aléatoire : prend chaque sommet avec prob 0.5
        individual = {node for node in nodes if random.random() < 0.5}
        population.append(individual)

    # Evolution
    best_cover = min(population, key=fitness)

    for gen in range(generations):
        # 1) Évaluation
        sorted_pop = sorted(population, key=fitness)

        # 2) Sélection (garder les meilleurs 50%)
        survivors = sorted_pop[: population_size // 2]

        # 3) Croisement (reproduction)
        new_population = []
        while len(new_population) < population_size:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            # Croisement (union, intersection, mélange aléatoire)

            # Exemple : on fait un "mélange" bit‐à‐bit
            child = set()
            for node in nodes:
                if node in p1 and node in p2:
                    child.add(node)
                elif node in p1 or node in p2:
                    # 50% de chances
                    if random.random() < 0.5:
                        child.add(node)
                else:
                    # ni p1 ni p2
                    pass

            # Mutation
            for node in nodes:
                if random.random() < mutation_prob:
                    if node in child:
                        child.remove(node)
                    else:
                        child.add(node)

            new_population.append(child)

        population = new_population

        # Mettre à jour le meilleur
        current_best = min(population, key=fitness)
        if fitness(current_best) < fitness(best_cover):
            best_cover = current_best

    return list(best_cover)

# solution non realisable on ne la remet pas dans la population
# croisenement en 2 points, uniforme, la coupe, l'echange par bijection