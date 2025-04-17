import os
import networkx as nx
from tqdm import tqdm
from solveur import minimum_vertex_cover
import random

FIXED_SEED = 42
TREE_SIZES = [25, 50, 75, 100]
GRAPHS_PER_COMBINATION = 1000
OUTPUT_DIR = "g6_files/tree"

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
    G = nx.complete_graph(n)
    rng = random.Random(seed)
    while G.number_of_edges() > n - 1:
        edge_to_remove = rng.choice(list(G.edges()))
        G.remove_edge(*edge_to_remove)
        if not nx.is_connected(G):
            G.add_edge(*edge_to_remove)
    return G

def save_graph_to_g6(graph, filename):
    os.makedirs("g6_files/tree/", exist_ok=True)
    nx.write_graph6(graph, f"g6_files/tree/{filename}")


def generate_uniform_mvc_trees(n, graphs_needed, seed_base):
    valid_graphs = 0
    tries = 0
    target_cover_size = None
    pbar = tqdm(total=graphs_needed, desc=f"Arbres n={n} (mvc=?)")

    while valid_graphs < graphs_needed:
        seed = seed_base + tries
        tree = generate_random_tree(n, seed)
        result = minimum_vertex_cover(tree)

        if result and result[1] == "Optimal":
            cover_size = result[0]

            # pour la première instance, je stocke la taille du MVC
            if target_cover_size is None:
                target_cover_size = cover_size
                pbar.set_description(f"Arbres n={n} (mvc={target_cover_size})")

            # on garde uniquement les arbres avec le même MVC que la première instance
            if cover_size == target_cover_size:
                filename = f"tree-{cover_size}-{n}-{valid_graphs+1}.g6"
                save_graph_to_g6(tree, filename)
                valid_graphs += 1
                pbar.update(1)

        tries += 1

    pbar.close()

if __name__ == "__main__":
    for n in TREE_SIZES:
        print(f"Génération de {GRAPHS_PER_COMBINATION} arbres de taille n={n} avec mvc uniforme...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        seed_base = FIXED_SEED * 10**6 + n * 10**4
        generate_uniform_mvc_trees(n, GRAPHS_PER_COMBINATION, seed_base)
        print(f"Terminé pour n={n}. Fichiers dans {OUTPUT_DIR}/")
