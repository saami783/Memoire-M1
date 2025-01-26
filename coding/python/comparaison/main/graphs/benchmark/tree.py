import os
import csv
import random
import networkx as nx
import logging
import time
import psutil
from tqdm import tqdm
from main.solveur.solveur import minimum_vertex_cover

logging.basicConfig(level=logging.WARNING)

def generate_random_tree(n, seed=None):
    if seed is not None:
        random.seed(seed)
    G = nx.complete_graph(n)
    while G.number_of_edges() > n - 1:
        edge_to_remove = random.choice(list(G.edges()))
        G.remove_edge(*edge_to_remove)
        if not nx.is_connected(G):
            G.add_edge(*edge_to_remove)
    return G


def save_to_dimacs(graph: nx.Graph, filename: str):
    with open(filename, 'w') as f:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        f.write(f"p edge {n} {m}\n")
        mapping = {node: idx + 1 for idx, node in enumerate(graph.nodes())}
        for u, v in graph.edges():
            f.write(f"e {mapping[u]} {mapping[v]}\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    start_time = time.time()

    sizes = [50, 100, 200, 350]
    nb_instances_par_taille = 10

    output_dir = "dimacs_files/trees"
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = os.path.join(output_dir, "trees_properties.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["instance_file", "n_sommets", "nb_aretes", "cover_size", "deg_min", "deg_max", "status"])

        total_graphs = len(sizes) * nb_instances_par_taille
        with tqdm(total=total_graphs, desc="Génération des graphes", unit="graphe") as pbar:
            for n in sizes:
                logging.info(f"Début de la génération des graphes de taille {n}.")
                for i in range(1, nb_instances_par_taille + 1):
                    G = generate_random_tree(n, seed=42)
                    solution, cover_size, status = minimum_vertex_cover(G)

                    if status == "Optimal":
                        nb_edges = G.number_of_edges()
                        degrees = [deg for _, deg in G.degree()]
                        deg_min = min(degrees)
                        deg_max = max(degrees)

                        filename = f"tree_{n}_{int(cover_size)}_{i}.dimacs"
                        filepath = os.path.join(output_dir, filename)
                        save_to_dimacs(G, filepath)

                        writer.writerow([filename, n, nb_edges, int(cover_size), deg_min, deg_max, status])
                        logging.info(f"Graphe {filename} sauvegardé avec succès.")
                    else:
                        logging.warning(f"Instance (n={n}, i={i}) non sauvegardée (statut = {status})")

                    pbar.update(1)  # Met à jour la barre de progression

        # Affichage des statistiques finales
        end_time = time.time()
        elapsed_time = end_time - start_time
        cpu_usage = psutil.cpu_percent()

        logging.info(f"Nombre total de graphes générés : {total_graphs}")
        logging.info(f"Temps d'exécution total : {elapsed_time:.2f} secondes")
        logging.info(f"Utilisation du CPU : {cpu_usage}%")

    logging.info(f"Terminé. Les graphes et le fichier CSV sont disponibles dans '{output_dir}'.")


if __name__ == "__main__":
    main()
