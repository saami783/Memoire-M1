import os
import networkx as nx
from tqdm import tqdm
from main.solveur.solveur import minimum_vertex_cover

FIXED_SEED = 42
NODE_SIZES = [20, 40, 60, 80, 100]
EDGE_PROBABILITIES = [0.1, 0.2, 0.3, 0.4]
GRAPHS_PER_COMBINATION = 5


def generate_erdos_renyi_graph(num_nodes, prob, seed):
    return nx.erdos_renyi_graph(num_nodes, prob, seed=seed)


def save_graph_to_g6(graph, filename):
    os.makedirs("g6_files/erdos_renyi/", exist_ok=True)
    nx.write_graph6(graph, f"g6_files/erdos_renyi/{filename}")


if __name__ == "__main__":
    target_cover_sizes = {}

    for n in NODE_SIZES:
        for p in EDGE_PROBABILITIES:
            base_seed = hash(f"{FIXED_SEED}-{n}-{p}") % (2 ** 32)
            base_graph = generate_erdos_renyi_graph(n, p, base_seed)
            solveur = minimum_vertex_cover(base_graph)

            if solveur is None or solveur[1] != "Optimal":
                continue

            target_size = solveur[0]
            target_cover_sizes[(n, p)] = target_size

    for (n, p), target_size in target_cover_sizes.items():
        instance_count = 0
        attempt = 0

        while instance_count < GRAPHS_PER_COMBINATION and attempt < 1000:
            seed = hash(f"{FIXED_SEED}-{n}-{p}-{attempt}") % (2 ** 32)
            graph = generate_erdos_renyi_graph(n, p, seed)
            solveur = minimum_vertex_cover(graph)

            if solveur and solveur[1] == "Optimal" and solveur[0] == target_size:
                # erdos_renyi-[vertex-cover-size]-[n]-[p]-[instance + 1].g6
                save_graph_to_g6(graph, f"erdos_renyi-{target_size}-{n}-{p}-{instance_count + 1}.g6")
                instance_count += 1

            attempt += 1