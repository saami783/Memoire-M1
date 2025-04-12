import itertools
import os
import networkx as nx
from tqdm import tqdm
from solveur import minimum_vertex_cover

FIXED_SEED = 42
NODE_SIZES = [20, 40, 60, 80, 100]
EDGE_PROBABILITIES = [0.1, 0.2, 0.3, 0.4]
GRAPHS_PER_COMBINATION = 100


def generate_erdos_renyi_graph(num_nodes, prob, seed):
    return nx.erdos_renyi_graph(num_nodes, prob, seed=seed)


def save_graph_to_g6(graph, filename):
    os.makedirs("g6_files/erdos_renyi/", exist_ok=True)
    nx.write_graph6(graph, f"g6_files/erdos_renyi/{filename}")


if __name__ == "__main__":
    target_cover_sizes = {}
    combinations = list(itertools.product(NODE_SIZES, EDGE_PROBABILITIES))

    for n, p in tqdm(combinations, desc="Détermination des couvertures optimales"):
        target_size = None
        attempt_base = 0

        while target_size is None and attempt_base < 1000:
            seed = hash(f"{FIXED_SEED}-{n}-{p}-{attempt_base}") % (2 ** 32)
            base_graph = generate_erdos_renyi_graph(n, p, seed)

            if nx.is_connected(base_graph):
                solveur = minimum_vertex_cover(base_graph)
                if solveur and solveur[1] == "Optimal":
                    target_size = solveur[0]

            attempt_base += 1

        if target_size is not None:
            target_cover_sizes[(n, p)] = target_size
        else:
            print(f"ERREUR : n={n}, p={p} après 1000 tentatives")

    for (n, p), target_size in tqdm(target_cover_sizes.items(), desc="Génération globale"):
        instance_count = 0
        combo_progress = tqdm(total=GRAPHS_PER_COMBINATION, desc=f"n={n}, p={p}", leave=False)

        while instance_count < GRAPHS_PER_COMBINATION:
            attempt = 0
            found = False

            while not found and attempt < 1000:
                seed = hash(f"{FIXED_SEED}-{n}-{p}-{instance_count}-{attempt}") % (2 ** 32)
                graph = generate_erdos_renyi_graph(n, p, seed)

                if nx.is_connected(graph):
                    solveur = minimum_vertex_cover(graph)
                    if solveur and solveur[1] == "Optimal" and solveur[0] == target_size:
                        save_graph_to_g6(graph, f"erdos_renyi-{target_size}-{n}-{p}-{instance_count + 1}.g6")
                        instance_count += 1
                        combo_progress.update(1)
                        found = True  # Passe à l'instance suivante

                attempt += 1
                combo_progress.set_postfix({"Tentatives": attempt})

            if not found:
                print(f"Échec pour n={n}, p={p}, instance {instance_count + 1}")
                break

        combo_progress.close()