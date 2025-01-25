def memetic(graph,
                         population_size=30,
                         generations=100,
                         mutation_prob=0.1):
    """
    Algorithme Mémétique simple pour Vertex Cover:
      - Comme un GA, mais on applique une Local Search à chaque enfant avant de l'ajouter à la population.
    """
    nodes = list(graph.nodes())

    def is_valid_cover(C):
        for (u, v) in graph.edges():
            if u not in C and v not in C:
                return False
        return True

    def fitness(C):
        if not is_valid_cover(C):
            return 10_000 + len(C)
        else:
            return len(C)

    # Petite recherche locale basique
    def local_search(C):
        improved = True
        C = set(C)
        while improved:
            improved = False
            for node in list(C):
                Ctemp = C - {node}
                if is_valid_cover(Ctemp):
                    C = Ctemp
                    improved = True
                    break
        return C

    # --- Génération population initiale
    population = []
    for _ in range(population_size):
        individual = {node for node in nodes if random.random() < 0.5}
        population.append(individual)

    best_cover = min(population, key=fitness)

    for gen in range(generations):
        # Évaluation
        sorted_pop = sorted(population, key=fitness)

        # Sélection (conserver top 50% par ex.)
        survivors = sorted_pop[: population_size // 2]

        # Reproduction
        new_population = []
        while len(new_population) < population_size:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            # Croisement
            child = set()
            for n in nodes:
                if (n in p1) and (n in p2):
                    child.add(n)
                elif (n in p1) or (n in p2):
                    if random.random() < 0.5:
                        child.add(n)

            # Mutation
            for n in nodes:
                if random.random() < mutation_prob:
                    if n in child:
                        child.remove(n)
                    else:
                        child.add(n)

            # >>> Ici l'étape "mémétique": on applique une Local Search au child
            child = local_search(child)

            new_population.append(child)

        population = new_population

        # Meilleure solution
        current_best = min(population, key=fitness)
        if fitness(current_best) < fitness(best_cover):
            best_cover = current_best

    return list(best_cover)
