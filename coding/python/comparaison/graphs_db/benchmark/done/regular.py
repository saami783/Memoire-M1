import os
import networkx as nx
from tqdm import tqdm
from main.solveur.solveur import minimum_vertex_cover

FIXED_SEED = 42
REGULAR_SIZES = [28, 40, 52, 64, 76, 98]
REGULAR_DEGREES = [3, 4, 5, 6, 7, 8]
GRAPHS_PER_COMBINATION = 5

def generate_regular_graph(degree, num_nodes, seed):
    return nx.random_regular_graph(degree, num_nodes, seed=seed)


def save_graph_to_g6(graph, filename):
    os.makedirs("g6_files/regular/", exist_ok=True)
    nx.write_graph6(graph, f"g6_files/regular/{filename}")


if __name__ == "__main__":
    target_cover_sizes = {}

    for n in REGULAR_SIZES:
        for d in REGULAR_DEGREES:
            if d >= n:  # Éviter les degrés impossibles
                continue

            base_seed = hash(f"{FIXED_SEED}-{n}-{d}") % (2 ** 32)
            try:
                graph = generate_regular_graph(d, n, base_seed)
                solveur = minimum_vertex_cover(graph)

                if solveur and solveur[1] == "Optimal":
                    target_cover_sizes[(n, d)] = solveur[0]
            except nx.NetworkXError:
                continue

    for (n, d), target_size in target_cover_sizes.items():
        instance_count = 0
        attempt = 0

        with tqdm(total=GRAPHS_PER_COMBINATION, desc=f"n={n} d={d}", leave=False) as pbar:
            while instance_count < GRAPHS_PER_COMBINATION and attempt < 1000:
                seed = hash(f"{FIXED_SEED}-{n}-{d}-{attempt}") % (2 ** 32)

                try:
                    graph = generate_regular_graph(d, n, seed)
                    solveur = minimum_vertex_cover(graph)

                    if solveur and solveur[1] == "Optimal" and solveur[0] == target_size:
                        # regular-[vertex-cover-size]-[n]-[d]-[instance].g6
                        save_graph_to_g6(graph,f"regular-{target_size}-{n}-{d}-{instance_count + 1}.g6")
                        instance_count += 1
                        pbar.update(1)

                except nx.NetworkXError:
                    pass

                attempt += 1

            if instance_count < GRAPHS_PER_COMBINATION:
                print(f"\nAvertissement: Seulement {instance_count} instances valides pour n={n} d={d}")