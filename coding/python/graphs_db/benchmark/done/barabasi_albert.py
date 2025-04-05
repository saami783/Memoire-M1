import os
import networkx as nx
from tqdm import tqdm
from main.solveur.solveur import minimum_vertex_cover

FIXED_SEED = 42
BARABASI_ALBERT_SIZES = [100, 200, 300, 400, 500] # 22min à partir de 400 sommets.
EDGES_TO_ATTACH = [2, 3, 4]
GRAPHS_PER_COMBINATION = 5


def generate_ba_graph(n, m, seed):
    initial_graph = nx.complete_graph(m + 1)
    return nx.barabasi_albert_graph(n, m, seed=seed, initial_graph=initial_graph)


def save_graph_to_g6(graph, filename):
    os.makedirs("g6_files/barabasi_albert/", exist_ok=True)
    nx.write_graph6(graph, f"g6_files/barabasi_albert/{filename}")


if __name__ == "__main__":
    target_sizes = {}

    for n in BARABASI_ALBERT_SIZES:
        for m in EDGES_TO_ATTACH:
            if m >= n:
                continue

            # Génération de référence
            seed = hash(f"{FIXED_SEED}-{n}-{m}") % (2 ** 32)
            try:
                G = generate_ba_graph(n, m, seed)
                solveur = minimum_vertex_cover(G)

                if solveur and solveur[1] == "Optimal" and nx.is_connected(G):
                    target_sizes[(n, m)] = solveur[0]
            except nx.NetworkXError:
                continue

    for (n, m), target in target_sizes.items():
        instance_count = 0
        attempt = 0

        with tqdm(total=GRAPHS_PER_COMBINATION, desc=f"BA n={n} m={m}", leave=False) as pbar:
            while instance_count < GRAPHS_PER_COMBINATION and attempt < 1000:
                current_seed = hash(f"{FIXED_SEED}-{n}-{m}-{attempt}") % (2 ** 32)

                try:
                    G = generate_ba_graph(n, m, current_seed)
                    if not nx.is_connected(G):
                        continue

                    solveur = minimum_vertex_cover(G)
                    if solveur and solveur[1] == "Optimal" and solveur[0] == target:
                        save_graph_to_g6(
                            G,
                            f"ba-{target}-n{n}-m{m}-{instance_count + 1}.g6"
                        )
                        instance_count += 1
                        pbar.update(1)

                except nx.NetworkXError:
                    pass

                attempt += 1