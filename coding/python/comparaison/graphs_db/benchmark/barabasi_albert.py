import os
import networkx as nx
from tqdm import tqdm
from solveur import minimum_vertex_cover

FIXED_SEED = 42
BARABASI_ALBERT_SIZES = [100, 200, 300, 400]
EDGES_TO_ATTACH = [2, 3, 4]
GRAPHS_PER_COMBINATION = 100

def generate_ba_graph(n, m, seed):
    initial_graph = nx.complete_graph(m + 1)
    return nx.barabasi_albert_graph(n, m, seed=seed, initial_graph=initial_graph)

def save_graph_to_g6(graph, filename):
    os.makedirs("g6_files/barabasi_albert/", exist_ok=True)
    nx.write_graph6(graph, f"g6_files/barabasi_albert/{filename}")

def generate_uniform_mvc_ba_graphs(n, m, graphs_needed, seed_base):
    valid_graphs = 0
    tries = 0
    target_cover_size = None
    pbar = tqdm(total=graphs_needed, desc=f"BA n={n} m={m} (mvc=?)")

    while valid_graphs < graphs_needed:
        seed = seed_base + tries
        G = generate_ba_graph(n, m, seed)
        result = minimum_vertex_cover(G)

        if result and result[1] == "Optimal":
            cover_size = result[0]

            if target_cover_size is None:
                target_cover_size = cover_size
                pbar.set_description(f"BA n={n} m={m} (mvc={target_cover_size})")

            if cover_size == target_cover_size:
                filename = f"ba-{cover_size}-{n}-{m}-{valid_graphs+1}.g6"
                save_graph_to_g6(G, filename)
                valid_graphs += 1
                pbar.update(1)

        tries += 1

    pbar.close()

if __name__ == "__main__":
    for n in BARABASI_ALBERT_SIZES:
        for m in EDGES_TO_ATTACH:
            if m >= n:
                continue
            print(f"Generation de {GRAPHS_PER_COMBINATION} graphes BA n={n}, m={m} avec mvc uniforme...")
            seed_base = FIXED_SEED * 10**6 + n * 10**4 + m * 1000
            generate_uniform_mvc_ba_graphs(n, m, GRAPHS_PER_COMBINATION, seed_base)
            print(f"Termine pour n={n}, m={m}.")