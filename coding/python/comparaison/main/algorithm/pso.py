# @todo retourne des solutions en dessous de la taille optimale (sauf pour les graphes regular)

import math, random
from .utils import is_valid_cover

def pso(graph, swarm_size=20, iterations=50, w=0.9, c1=2.0, c2=2.0):
    """
    Binary PSO amélioré pour le Minimum Vertex Cover.
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    def cover_size(bitvec):
        return sum(bitvec)

    def fitness(bitvec):
        if not is_valid_cover(bitvec, graph):
            return float('inf')  # Élimine complètement les solutions non valides
        return cover_size(bitvec)

    # --- Initialisation
    swarm = []
    for _ in range(swarm_size):
        x = [1 if random.random() < 0.5 else 0 for _ in range(n)]
        v = [random.uniform(-4, 4) for _ in range(n)]  # Vitesses élargies
        swarm.append({"x": x, "v": v, "pbest": x[:]})

    gbest = min([p["x"] for p in swarm], key=fitness)

    for iteration in range(iterations):
        w = 0.9 - (0.9 - 0.4) * (iteration / iterations)  # Inertie adaptative

        for p in swarm:
            for i in range(n):
                r1, r2 = random.random(), random.random()
                p["v"][i] = (w * p["v"][i]
                             + c1 * r1 * (p["pbest"][i] - p["x"][i])
                             + c2 * r2 * (gbest[i] - p["x"][i]))

                # Sigmoïde avec tau adaptatif
                tau = 1.0
                prob = 1.0 / (1.0 + math.exp(-p["v"][i] / tau))
                p["x"][i] = 1 if random.random() < prob else 0

            if fitness(p["x"]) < fitness(p["pbest"]):
                p["pbest"] = p["x"][:]

        best_in_swarm = min(swarm, key=lambda part: fitness(part["x"]))
        if fitness(best_in_swarm["x"]) < fitness(gbest):
            gbest = best_in_swarm["x"][:]  # Mise à jour immédiate

        # Mutation aléatoire pour éviter la stagnation
        if iteration % 10 == 0:
            for p in swarm:
                if random.random() < 0.1:
                    rand_idx = random.randint(0, n - 1)
                    p["x"][rand_idx] = 1 - p["x"][rand_idx]

    return [nodes[i] for i, val in enumerate(gbest) if val == 1]