import os
import random
import networkx as nx


# Fonction pour écrire un graphe au format DIMACS
def write_graph_to_dimacs(graph, filepath):
    """
    Écrit un graphe au format DIMACS.
    Args:
        graph (networkx.Graph): Le graphe à écrire.
        filepath (str): Chemin du fichier de sortie.
    """
    with open(filepath, "w") as f:
        # Écrire le nombre de sommets et d'arêtes
        f.write(f"p edge {graph.number_of_nodes()} {graph.number_of_edges()}\n")
        # Écrire chaque arête
        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")  # Les indices DIMACS commencent à 1


# Configuration des benchmarks
base_folder = "benchmarks/erdos_renyi"
os.makedirs(base_folder, exist_ok=True)

# Paramètres pour les différentes classes de benchmarks Erdős-Rényi
benchmark_classes = [
    {"n": 1000, "p": 0.01},
    {"n": 5000, "p": 0.005},
    {"n": 4000, "p": 0.01},
    {"n": 5000, "p": 0.02},
    {"n": 10000, "p": 0.01},
]

# Seed fixe pour reproductibilité
SEED = 42
random.seed(SEED)

# Générer les benchmarks
for benchmark in benchmark_classes:
    n = benchmark["n"]
    p = benchmark["p"]

    # Dossier pour la classe de graphes
    class_folder = f"{base_folder}/erdos_renyi_n{n}_p{p:.4f}".replace(".", "_")
    os.makedirs(class_folder, exist_ok=True)

    for instance_id in range(1, 6):  # 5 instances par classe
        print(f"Generating graph for n={n}, p={p:.4f}, instance {instance_id}...")

        # Générer le graphe avec une graine fixe pour cohérence
        graph = nx.erdos_renyi_graph(n, p, seed=SEED)

        # Nommer et écrire le graphe
        filepath = os.path.join(class_folder, f"erdos_renyi_n{n}_p{p:.4f}_instance{instance_id}.dimacs".replace(".", "_"))
        write_graph_to_dimacs(graph, filepath)

print(f"Benchmarks générés dans le dossier {base_folder}.")
