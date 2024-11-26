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
base_folder = "benchmarks/regular"
os.makedirs(base_folder, exist_ok=True)

# Paramètres pour les différentes classes de benchmarks Regular
benchmark_classes = [
    {"n": 10000, "d": 10},  # Graphe moyen avec degré modéré
    {"n": 50000, "d": 20},  # Grand graphe avec un degré élevé
    {"n": 100000, "d": 10}, # Très grand graphe avec un degré faible
    {"n": 200000, "d": 15}, # Très grand graphe avec un degré modéré
    {"n": 500000, "d": 5},  # Immense graphe très clairsemé
]

# Seed fixe pour reproductibilité
SEED = 42
random.seed(SEED)

# Générer les benchmarks
for benchmark in benchmark_classes:
    n = benchmark["n"]
    d = benchmark["d"]

    # Vérifier que les paramètres sont valides
    if d >= n:
        raise ValueError(f"Invalid parameters: d ({d}) must be less than n ({n}).")
    if n % 2 != 0:
        raise ValueError(f"Invalid parameters: n ({n}) must be even for d-regular graphs.")

    # Dossier pour la classe de graphes
    class_folder = f"{base_folder}/regular_n{n}_d{d}"
    os.makedirs(class_folder, exist_ok=True)

    for instance_id in range(1, 6):  # 5 instances par classe
        print(f"Generating graph for n={n}, d={d}, instance {instance_id}...")

        # Générer le graphe avec une graine fixe pour cohérence
        graph = nx.random_regular_graph(d, n, seed=SEED + instance_id)

        # Nommer et écrire le graphe
        filepath = os.path.join(class_folder, f"regular_n{n}_d{d}_instance{instance_id}.dimacs")
        write_graph_to_dimacs(graph, filepath)

print(f"Benchmarks générés dans le dossier {base_folder}.")
