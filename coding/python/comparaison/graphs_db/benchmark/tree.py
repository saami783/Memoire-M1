import os
import networkx as nx
from tqdm import tqdm
from solveur import minimum_vertex_cover
import random

FIXED_SEED = 42
TREE_SIZES = [25, 50, 75, 100, 250, 500, 750, 1000]
GRAPHS_PER_COMBINATION = 1000

"""
Paramètres de génération des arbres :
1. On débute avec un graphe complet à n > 3 sommets
2. Retirer une arête au hasard de sorte que le graphe résultant soit connexe
3. Tant que le nombre d’arêtes est supérieur à n−1 on recommence l’étape 2.

exemple : pour n = 1000, on a 1000 sommets et 999 arêtes.
"""
def generate_random_tree(n, seed=None):
    if seed is not None:
        random.seed(seed)

    tree = nx.Graph()
    tree.add_nodes_from(range(n))
    for i in range(1, n):
        parent = random.randint(0, i - 1)
        tree.add_edge(parent, i)

    return tree

def save_graph_to_g6(graph, filename):
    os.makedirs("g6_files/tree/", exist_ok=True)
    nx.write_graph6(graph, f"g6_files/tree/{filename}")


if __name__ == "__main__":
    target_cover_sizes = {}

    for n in TREE_SIZES:
        base_seed = hash(f"{FIXED_SEED}-{n}") % (2 ** 32)
        tree = generate_random_tree(n, base_seed)
        solveur = minimum_vertex_cover(tree)

        if solveur and solveur[1] == "Optimal":
            target_cover_sizes[n] = solveur[0]

    for n, target_size in target_cover_sizes.items():
        instance_count = 0
        attempt = 0

        with tqdm(total=GRAPHS_PER_COMBINATION, desc=f"Arbres n={n}", leave=False) as pbar:
            while instance_count < GRAPHS_PER_COMBINATION and attempt < 1000:
                seed = hash(f"{FIXED_SEED}-{n}-{attempt}") % (2 ** 32)
                tree = generate_random_tree(n, seed)
                solveur = minimum_vertex_cover(tree)

                attempt = 0

                if solveur and solveur[1] == "Optimal" and solveur[0] == target_size:
                    save_graph_to_g6(
                        tree,
                        f"tree-{target_size}-{n}-{instance_count + 1}.g6"
                    )
                    instance_count += 1
                    pbar.update(1)

                attempt += 1

            if instance_count < GRAPHS_PER_COMBINATION:
                print(f"Avertissement: Taille {target_size} rare pour n={n} ({instance_count}/5)")