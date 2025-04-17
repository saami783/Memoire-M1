import os
import networkx as nx
from tqdm import tqdm
from solveur import minimum_vertex_cover
import random

FIXED_SEED = 42
TREE_SIZES = [25, 50, 75, 100, 250]
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


if __name__ == "__main__":
    for n in TREE_SIZES:
        print(f"Génération de {GRAPHS_PER_COMBINATION} arbres de taille n={n}...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with tqdm(total=GRAPHS_PER_COMBINATION, desc=f"Arbres n={n}") as pbar:
            for i in range(GRAPHS_PER_COMBINATION):
                # graine déterministe pour chaque instance
                seed = FIXED_SEED * 10**6 + n * 10**4 + i
                tree = generate_random_tree(n, seed)
                result = minimum_vertex_cover(tree)
                if result and result[1] == "Optimal":
                    cover_size = result[0]
                else:
                    cover_size = "NA"
                filename = f"tree-{cover_size}-{n}-{i+1}.g6"
                save_graph_to_g6(tree, filename)
                pbar.update(1)
        print(f"Terminé pour n={n}. Fichiers dans {OUTPUT_DIR}/")
