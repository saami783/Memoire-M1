import os
import networkx as nx
from tqdm import tqdm
from solveur import minimum_vertex_cover

FIXED_SEED = 42
REGULAR_SIZES = [28, 40, 52, 64, 76, 98]
REGULAR_DEGREES = [3, 4, 5, 6, 7, 8]
GRAPHS_PER_COMBINATION = 100

def generate_regular_graph(degree, num_nodes, seed):
    return nx.random_regular_graph(degree, num_nodes, seed=seed)


def save_graph_to_g6(graph, filename):
    os.makedirs("g6_files/regular/", exist_ok=True)
    nx.write_graph6(graph, f"g6_files/regular/{filename}")


def generate_uniform_mvc_regular_graphs(n, d, graphs_needed, seed_base):
    valid_graphs = 0
    tries = 0
    target_cover_size = None
    pbar = tqdm(total=graphs_needed, desc=f"n={n} d={d} (mvc=?)")

    while valid_graphs < graphs_needed and tries < 1000:
        seed = seed_base + tries
        try:
            G = generate_regular_graph(d, n, seed)
            result = minimum_vertex_cover(G)

            if result and result[1] == "Optimal":
                cover_size = result[0]

                if target_cover_size is None:
                    target_cover_size = cover_size
                    pbar.set_description(f"n={n} d={d} (mvc={target_cover_size})")

                if cover_size == target_cover_size:
                    filename = f"regular-{cover_size}-{n}-{d}-{valid_graphs + 1}.g6"
                    save_graph_to_g6(G, filename)
                    valid_graphs += 1
                    pbar.update(1)
        except nx.NetworkXError:
            pass
        tries += 1

    pbar.close()
    if valid_graphs < graphs_needed:
        print(f"Seulement {valid_graphs} graphes valides pour n={n}, d={d}.")


if __name__ == "__main__":
    for n in REGULAR_SIZES:
        for d in REGULAR_DEGREES:
            if d >= n:
                continue
            print(f"Generation de {GRAPHS_PER_COMBINATION} graphes reguliers n={n}, d={d} avec mvc uniforme...")
            seed_base = FIXED_SEED * 10**6 + n * 10**4 + d * 1000
            generate_uniform_mvc_regular_graphs(n, d, GRAPHS_PER_COMBINATION, seed_base)
