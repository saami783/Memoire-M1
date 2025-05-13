import os
import torch
import networkx as nx
import random
from tqdm import tqdm
from fastvc import fastvc as fastvc_approximation
import numpy as np

# On parcourt les arêtes dans un ordre déterminé (ex : lexicographique),
# et pour chaque arête non encore couverte, on ajoute les deux extrémités à la couverture,
# puis on supprime toutes les arêtes incidentes à ces deux sommets.
def greedy_vertex_cover_exact(A):
    G = nx.from_numpy_array(A.numpy())
    edges = sorted(G.edges())
    covered_edges = set()
    cover = set()
    for u, v in edges:
        if (u, v) not in covered_edges:
            cover.update([u, v])
            for w in G.neighbors(u):
                covered_edges.add(tuple(sorted((u, w))))
            for w in G.neighbors(v):
                covered_edges.add(tuple(sorted((v, w))))
    y = torch.zeros(A.shape[0])
    y[list(cover)] = 1.0
    return y

# Générateur de graphe Erdős–Rényi G(n, p), connecté
def generate_connected_erdos_renyi(n_nodes, density_type, max_tries=100, base_seed=42):
    if density_type == "sparse":
        e_min = 2 * n_nodes
        e_max = int(2.5 * n_nodes)
    elif density_type == "dense":
        e_min = 5 * n_nodes
        e_max = min(n_nodes * (n_nodes - 1) // 2, 500 * n_nodes)
    else:
        raise ValueError("density_type must be 'sparse' or 'dense'")

    max_possible_edges = n_nodes * (n_nodes - 1) // 2
    n_edges = min(np.random.randint(e_min, e_max + 1), max_possible_edges)
    p = 2 * n_edges / (n_nodes * (n_nodes - 1))

    for attempt in range(max_tries):
        seed = base_seed + attempt
        G = nx.gnp_random_graph(n_nodes, p, seed=seed)
        if nx.is_connected(G):
            A = nx.to_numpy_array(G)
            return torch.tensor(A, dtype=torch.float32)

    raise RuntimeError(f"[!] Failed to generate connected Erdős–Rényi graph after {max_tries} tries.")

# Génération du dataset
def generate_dataset(save_dir, n_graphs, density_type, prefix, use_fastvc=False, save_labels=True):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    pbar = tqdm(total=n_graphs, desc=f"Generating {prefix}-{density_type}")
    while count < n_graphs:
        # Taille raisonnable selon la densité
        if density_type == "sparse":
            n_nodes = random.randint(20, 300)
        else:
            n_nodes = random.randint(50, 200)

        try:
            A = generate_connected_erdos_renyi(n_nodes, density_type, base_seed=42 + count)
        except RuntimeError as e:
            print(e)
            continue

        torch.save(A, os.path.join(save_dir, f"A_{prefix}_{count:04d}.pt"))

        if save_labels:
            if use_fastvc:
                y = fastvc_approximation(A)
            else:
                y = greedy_vertex_cover_exact(A)
            torch.save(y, os.path.join(save_dir, f"y_{prefix}_{count:04d}.pt"))

        count += 1
        pbar.update(1)
    pbar.close()

# Exécution unique
if __name__ == "__main__":
    # Pour test rapide
    # generate_dataset("test", 100, "sparse", "test_sparse", save_labels=False)
    # generate_dataset("test", 100, "dense", "test_dense", save_labels=False)

    # Pour production complète
    generate_dataset("train", 1500, "sparse", "train_sparse", use_fastvc=False)
    generate_dataset("train", 1500, "dense", "train_dense", use_fastvc=True)
    generate_dataset("val", 300, "sparse", "val_sparse", use_fastvc=False)
    generate_dataset("val", 300, "dense", "val_dense", use_fastvc=True)