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
base_folder = "benchmarks/barabasi_albert"
os.makedirs(base_folder, exist_ok=True)

# Paramètres pour les différentes classes de benchmarks
benchmark_classes = [
    {"optimal_size": 90, "n": 1000, "m": 2},
    {"optimal_size": 150, "n": 5000, "m": 3},
    {"optimal_size": 300, "n": 4000, "m": 4},
    {"optimal_size": 500, "n": 5000, "m": 5},
    {"optimal_size": 1000, "n": 100000, "m": 6},
]

# Seed fixe pour reproductibilité
SEED = 42
random.seed(SEED)

# Générer les benchmarks
for benchmark in benchmark_classes:
    optimal_size = benchmark["optimal_size"]
    n = benchmark["n"]
    m = benchmark["m"]

    class_folder = f"{base_folder}/barabasi_albert_{optimal_size}-{n}-{m}"
    os.makedirs(class_folder, exist_ok=True)

    for instance_id in range(1, 6):  # 5 instances par classe
        print(f"Generating graph for {optimal_size=}, {n=}, {m=}, instance {instance_id}...")

        # Générer le graphe avec une graine fixe pour cohérence
        graph = nx.barabasi_albert_graph(n, m, seed=SEED)

        # Nommer et écrire le graphe
        filepath = os.path.join(class_folder, f"barabasi_albert_{optimal_size}-{n}-{m}-{instance_id}.dimacs")
        write_graph_to_dimacs(graph, filepath)

print(f"Benchmarks générés dans le dossier {base_folder}.")
