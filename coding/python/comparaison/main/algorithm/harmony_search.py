import random
from .utils import is_valid_cover

def harmony_search_(graph,
                                hm_size=10,  # harmony memory size
                                iterations=50,  # nb d'itérations
                                consider_rate=0.9,  # prob de prendre un sommet depuis la mémoire
                                adjust_rate=0.3,  # prob d'ajuster une décision
                                pitch_adjust_prob=0.1):
    """
    Harmony Search simplifiée pour le Vertex Cover.

    Paramètres:
    -----------
    hm_size : taille de la "harmony memory"
    iterations : nb d'itérations
    consider_rate : prob de prendre un "gène" (sommet in/out) depuis la mémoire
    adjust_rate : prob de faire un "adjustment"
    pitch_adjust_prob : intensité de l'ajustement

    Chaque "harmony" est représentée par un set de sommets.
    """
    nodes = list(graph.nodes())

    def fitness(C):
        # Pénalité si pas valide
        if not is_valid_cover(C, graph):
            return 1_000_000 + len(C)
        return len(C)

    # -- 1) Génération initiale d'harmonies (aléatoires)
    harmony_memory = []
    for _ in range(hm_size):
        cover = {n for n in nodes if random.random() < 0.5}
        harmony_memory.append(cover)

    # On garde la meilleure en mémoire
    best_cover = min(harmony_memory, key=fitness)

    for _ in range(iterations):
        # -- 2) Construire une nouvelle "harmony"
        new_cover = set()
        for n in nodes:
            if random.random() < consider_rate:
                # On prend la valeur (in or out) d'une harmonie existante
                random_harmony = random.choice(harmony_memory)
                if n in random_harmony:
                    new_cover.add(n)

                # Ajustement éventuel
                if random.random() < adjust_rate:
                    # On fait un "pitch adjustment" (retire ou ajoute n)
                    if n in new_cover and random.random() < pitch_adjust_prob:
                        new_cover.remove(n)
                    elif n not in new_cover and random.random() < pitch_adjust_prob:
                        new_cover.add(n)
            else:
                # On choisit au hasard
                if random.random() < 0.5:
                    new_cover.add(n)

        # -- 3) Évaluer et insérer dans la mémoire si c'est mieux que la pire
        worst_in_memory = max(harmony_memory, key=fitness)
        if fitness(new_cover) < fitness(worst_in_memory):
            harmony_memory.remove(worst_in_memory)
            harmony_memory.append(new_cover)
            # Update best
            if fitness(new_cover) < fitness(best_cover):
                best_cover = new_cover

    return list(best_cover)
