import os
import csv
import networkx as nx
from tqdm import tqdm

# On importe la fonction de résolution du Minimum Vertex Cover
from main.solveur.solveur import minimum_vertex_cover

# Constantes et paramètres
NODE_SIZES = [20, 40, 60, 80, 100]  # Tailles des graphes
EDGE_PROBABILITIES = [0.1, 0.2, 0.3, 0.4]  # Probabilités d'arête
GRAPHS_PER_COMBINATION = 30  # Nombre de graphes par (n, p)

def generate_erdos_renyi_graph(num_nodes, prob_connection, seed):
    """
    Génère un graphe G(n, p) selon le modèle d’Erdős-Rényi.
    """
    return nx.erdos_renyi_graph(num_nodes, prob_connection, seed=seed)

def save_graph_to_dimacs(graph: nx.Graph, filename: str):
    """
    Sauvegarde le graphe au format DIMACS avec indexation 1-based.
    """
    with open(filename, "w") as f:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        f.write(f"p edge {n} {m}\n")
        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")

def main():
    # Répertoire de sortie
    output_dir = "dimacs_files/erdos_renyi"
    os.makedirs(output_dir, exist_ok=True)

    # Fichier CSV pour stocker les résultats
    csv_filename = os.path.join(output_dir, "erdos_renyi_properties.csv")

    # Nombre total de graphes à générer
    total_combinations = len(NODE_SIZES) * len(EDGE_PROBABILITIES) * GRAPHS_PER_COMBINATION

    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow([
            "instance_file", "n_sommets", "nb_aretes", "cover_size",
            "deg_min", "deg_max", "prob_connection", "seed", "status"
        ])

        with tqdm(total=total_combinations, desc="Génération des graphes", unit="graphe") as pbar:
            for num_nodes in NODE_SIZES:
                for prob_connection in EDGE_PROBABILITIES:
                    generated_count = 0  # On s'assure d'avoir 30 graphes valides

                    while generated_count < GRAPHS_PER_COMBINATION:
                        seed = generated_count + 1  # On utilise un entier traçable comme seed
                        graph = generate_erdos_renyi_graph(num_nodes, prob_connection, seed)

                        # Résolution du Minimum Vertex Cover
                        solution, cover_size, status = minimum_vertex_cover(graph)

                        if status == "Optimal":
                            nb_edges = graph.number_of_edges()
                            degrees = [deg for _, deg in graph.degree()]
                            deg_min = min(degrees)
                            deg_max = max(degrees)

                            # Fichier DIMACS avec bonne numérotation
                            filename = f"erdosRenyi_{num_nodes}_{cover_size}_{generated_count + 1}.dimacs"

                            filepath = os.path.join(output_dir, filename)

                            # Sauvegarde
                            save_graph_to_dimacs(graph, filepath)

                            # Écriture dans le CSV
                            writer.writerow([
                                filename, num_nodes, nb_edges, int(cover_size),
                                deg_min, deg_max, prob_connection, seed, status
                            ])

                            generated_count += 1  # On incrémente uniquement si le graphe est valide
                            pbar.update(1)

    print(f"\nTerminé. Les fichiers sont disponibles dans : {output_dir}")

if __name__ == "__main__":
    main()
