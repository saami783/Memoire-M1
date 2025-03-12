import os
import networkx as nx
from tqdm import tqdm
from main.solveur.solveur import minimum_vertex_cover
from networkx.generators import trees

FIXED_SEED = 42
TREE_SIZES = [20, 40, 60, 80, 100]  # Nombre de nœuds
GRAPHS_PER_COMBINATION = 5  # Instances par taille


def generate_tree(num_nodes, seed):
    """Génère un arbre aléatoire avec n nœuds"""
    return trees.random_tree(num_nodes, seed=seed)


def save_graph_to_g6(graph, filename):
    os.makedirs("g6_files/tree/", exist_ok=True)
    nx.write_graph6(graph, f"g6_files/tree/{filename}")


if __name__ == "__main__":
    # Déterminer la taille cible pour chaque n
    target_cover_sizes = {}

    for n in TREE_SIZES:
        # Génération de l'arbre de référence
        base_seed = hash(f"{FIXED_SEED}-{n}") % (2 ** 32)
        tree = generate_tree(n, base_seed)
        solveur = minimum_vertex_cover(tree)

        if solveur and solveur[1] == "Optimal":
            target_cover_sizes[n] = solveur[0]

    # Génération des instances avec taille contrôlée
    for n, target_size in target_cover_sizes.items():
        instance_count = 0
        attempt = 0

        with tqdm(total=GRAPHS_PER_COMBINATION, desc=f"Arbres n={n}", leave=False) as pbar:
            while instance_count < GRAPHS_PER_COMBINATION and attempt < 1000:
                seed = hash(f"{FIXED_SEED}-{n}-{attempt}") % (2 ** 32)
                tree = generate_tree(n, seed)
                solveur = minimum_vertex_cover(tree)

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