import os
import networkx as nx


# Fonction pour écrire un graphe au format DIMACS
def write_graph_to_dimacs(graph, filepath):
    """
    Écrit un graphe au format DIMACS.
    Args:
        graph (networkx.Graph): Le graphe à écrire.
        filepath (str): Chemin du fichier de sortie.
    """
    # Créer un mapping des sommets vers des indices numériques
    mapping = {node: idx + 1 for idx, node in enumerate(graph.nodes())}

    with open(filepath, "w") as f:
        # Écrire le nombre de sommets et d'arêtes
        f.write(f"p edge {graph.number_of_nodes()} {graph.number_of_edges()}\n")
        # Écrire chaque arête en utilisant le mapping
        for u, v in graph.edges():
            f.write(f"e {mapping[u]} {mapping[v]}\n")  # Les indices DIMACS commencent à 1


# Fonction pour générer un graphe Anti-MDG
def generate_anti_mdg_graph(k):
    """
    Génère un graphe Anti-MDG de dimension k.
    Args:
        k (int): La dimension du graphe Anti-MDG.

    Returns:
        networkx.Graph: Le graphe Anti-MDG généré.
    """
    G = nx.Graph()

    # Étape 1 : Ajouter les ensembles A et B
    A = [f"A{i + 1}" for i in range(k)]
    B = [f"B{i + 1}" for i in range(k)]
    G.add_nodes_from(A)
    G.add_nodes_from(B)

    # Coupler les sommets de A et B
    for i in range(k):
        G.add_edge(A[i], B[i])

    # Étape 2 et suivantes : Ajouter les sommets de l'ensemble C
    C = []
    for i in range(1, k + 1):
        # Ajouter un sommet de degré i dans C
        Ci = f"C{i}"
        C.append(Ci)
        G.add_node(Ci)
        # Connecter Ci aux i premiers sommets de B
        for j in range(i):
            G.add_edge(Ci, B[j])

    return G


# Configuration des benchmarks
base_folder = "benchmarks/anti_mdg"
os.makedirs(base_folder, exist_ok=True)

# Paramètres pour les différentes classes de benchmarks Anti-MDG
benchmark_classes = [
    {"k": 5},  # Petit graphe
    {"k": 10},  # Graphe moyen
    {"k": 20},  # Grand graphe
    {"k": 50},  # Très grand graphe
]

# Générer les benchmarks
for benchmark in benchmark_classes:
    k = benchmark["k"]
    class_folder = f"{base_folder}/anti_mdg_k{k}"
    os.makedirs(class_folder, exist_ok=True)

    for instance_id in range(1, 6):  # 5 instances par classe
        print(f"Generating Anti-MDG graph for k={k}, instance {instance_id}...")

        # Générer le graphe
        graph = generate_anti_mdg_graph(k)

        # Nommer et écrire le graphe
        filepath = os.path.join(class_folder, f"anti_mdg_k{k}_instance{instance_id}.dimacs")
        write_graph_to_dimacs(graph, filepath)

print(f"Benchmarks générés dans le dossier {base_folder}.")
