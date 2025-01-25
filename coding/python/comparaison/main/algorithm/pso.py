import math, random


def pso(graph,
                     swarm_size=20,
                     iterations=50,
                     w=0.7,
                     c1=1.4,
                     c2=1.4):
    """
    Binary PSO (très simplifié) pour Vertex Cover.

    Paramètres:
    -----------
    swarm_size : nb de particules
    iterations : nb d'itérations
    w, c1, c2 : coefficients d'inertie et d'accélération
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    def is_valid_cover(bitvec):
        # bitvec[i] = 1 => on prend nodes[i]
        chosen = {nodes[i] for i, val in enumerate(bitvec) if val == 1}
        for (u, v) in graph.edges():
            if (u not in chosen) and (v not in chosen):
                return False
        return True

    def cover_size(bitvec):
        return sum(bitvec)

    def fitness(bitvec):
        # Pénalité si non valide
        if not is_valid_cover(bitvec):
            return 10_000 + cover_size(bitvec)
        return cover_size(bitvec)

    # --- Initialisation
    swarm = []
    for _ in range(swarm_size):
        # Position initiale binaire aléatoire
        x = [1 if random.random() < 0.5 else 0 for _ in range(n)]
        # Vitesse initiale (réelle) aléatoire
        v = [random.uniform(-1, 1) for _ in range(n)]
        swarm.append({
            "x": x,
            "v": v,
            "pbest": x[:],  # la meilleure position locale
        })

    # Meilleure position globale
    gbest = min([p["x"] for p in swarm], key=fitness)

    for _ in range(iterations):
        for p in swarm:
            # Mise à jour de v
            for i in range(n):
                r1 = random.random()
                r2 = random.random()
                p["v"][i] = (w * p["v"][i]
                             + c1 * r1 * (p["pbest"][i] - p["x"][i])
                             + c2 * r2 * (gbest[i] - p["x"][i]))

            # Mise à jour de x (en binaire via sigmoïde)
            for i in range(n):
                # sigmoïde
                prob = 1.0 / (1.0 + math.exp(-p["v"][i]))
                if random.random() < prob:
                    p["x"][i] = 1
                else:
                    p["x"][i] = 0

            # Mise à jour pbest si amélioration
            if fitness(p["x"]) < fitness(p["pbest"]):
                p["pbest"] = p["x"][:]

        # Mise à jour gbest
        best_in_swarm = min(swarm, key=lambda part: fitness(part["x"]))
        if fitness(best_in_swarm["x"]) < fitness(gbest):
            gbest = best_in_swarm["x"][:]

    return [nodes[i] for i, val in enumerate(gbest) if val == 1]
