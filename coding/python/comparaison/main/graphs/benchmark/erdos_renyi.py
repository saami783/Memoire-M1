import os
import networkx as nx
from main.solveur.solveur import minimum_vertex_cover

# Liste des graines fixes pour garantir la reproductibilité
FIXED_SEEDS = [42, 43, 44, 45, 46]

# Paramètres spécifiés dans le texte
NODE_SIZES = [20, 40, 60, 80, 100]  # Tailles des graphes
EDGE_PROBABILITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Probabilités d'arête
GRAPHS_PER_COMBINATION = 100  # Nombre de graphes pour chaque combinaison

def generate_erdos_renyi_graph(num_nodes, prob_connection, seed):
    """
    Génère un graphe Erdős-Rényi avec une graine fixe.
    """
    return nx.erdos_renyi_graph(num_nodes, prob_connection, seed=seed)

def save_graph_to_dimacs(graph, cover_size, filename):
    """
    Sauvegarde le graphe au format DIMACS.
    """
    with open(filename, "w") as f:
        f.write(f"p edge {graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")  # DIMACS utilise des index 1-based

if __name__ == "__main__":
    # Création du répertoire de sortie
    output_dir = "dimacs_files/erdos_renyi"
    os.makedirs(output_dir, exist_ok=True)

    total_graphs = 0

    for seed in FIXED_SEEDS:
        for num_nodes in NODE_SIZES:
            for prob_connection in EDGE_PROBABILITIES:
                for _ in range(GRAPHS_PER_COMBINATION):
                    try:
                        print(f"Génération de G({num_nodes}, {prob_connection}) avec seed={seed}")

                        # Génération du graphe Erdős-Rényi
                        graph = generate_erdos_renyi_graph(num_nodes, prob_connection, seed)

                        # Calcul du minimum vertex cover
                        solution, cover_size, status = minimum_vertex_cover(graph)

                        if status != "Optimal":
                            print("Arrêt : Le solveur n'a pas trouvé une solution optimale.")
                            continue

                        # Création du nom de fichier DIMACS
                        filename = f"erdos_renyi-{num_nodes}-{prob_connection}-{cover_size}-{seed}.dimacs"
                        filepath = os.path.join(output_dir, filename)

                        # Sauvegarde du graphe au format DIMACS
                        save_graph_to_dimacs(graph, cover_size, filepath)
                        print(f"Graphe sauvegardé : {filepath}, couverture minimale : {cover_size}")

                        print("Graphe numéro total_graphs {total_graphs} généré")
                        total_graphs += 1

                    except Exception as e:
                        print(f"Erreur : {e}")
                        continue

    print(f"Nombre total de graphes générés : {total_graphs}")
