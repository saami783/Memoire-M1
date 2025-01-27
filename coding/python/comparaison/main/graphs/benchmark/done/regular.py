import os
import csv
import networkx as nx
from tqdm import tqdm

from main.solveur.solveur import minimum_vertex_cover

# Paramètres
FIXED_SEED = 42

# On va générer 36 graphes réguliers = 6 tailles × 6 degrés
# (vous pouvez ajuster ces listes selon vos besoins).
REGULAR_SIZES = [28, 40, 52, 64, 76, 98]  # 6 tailles
REGULAR_DEGREES = [3, 4, 5, 6, 7, 8]  # 6 degrés


def generate_regular_graph(num_nodes, degree, seed=FIXED_SEED):
    """
    Génère un graphe d-régulier sur num_nodes sommets,
    avec la graine 'seed' pour la reproductibilité.
    """
    return nx.random_regular_graph(degree, num_nodes, seed=seed)


def save_graph_to_dimacs(graph: nx.Graph, cover_size: int, filename: str):
    """
    Sauvegarde un graphe non orienté au format DIMACS.
    Les sommets sont numérotés de 1 à n.
    """
    with open(filename, "w") as f:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        f.write(f"p edge {n} {m}\n")
        # Écriture des arêtes (1-based)
        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")


def main():
    output_dir = "dimacs_files/regular"
    os.makedirs(output_dir, exist_ok=True)

    # Fichier CSV de propriétés
    csv_filename = os.path.join(output_dir, "regular_properties.csv")

    # On prévoit 6×6 = 36 graphes
    total_graphs = len(REGULAR_SIZES) * len(REGULAR_DEGREES)

    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        # En-tête CSV
        writer.writerow([
            "instance_file",
            "n_sommets",
            "nb_aretes",
            "cover_size",
            "deg_min",
            "deg_max",
            "degree",
            "seed",
            "status"
        ])

        # Barre de progression
        with tqdm(total=total_graphs, desc="Génération des graphes réguliers", unit="graphe") as pbar:
            instance_index = 1  # Identifiant pour différencier les graphes

            for n in REGULAR_SIZES:
                for d in REGULAR_DEGREES:
                    # Génération d'un graphe d-régulier
                    graph = generate_regular_graph(n, d, seed=FIXED_SEED)

                    # Résolution du Minimum Vertex Cover
                    solution, cover_size, status = minimum_vertex_cover(graph)

                    if status == "Optimal":
                        nb_edges = graph.number_of_edges()
                        degrees = [deg for _, deg in graph.degree()]
                        deg_min = min(degrees)
                        deg_max = max(degrees)

                        # Nom du fichier : regular_<n>_<cover>_<i>.dimacs
                        filename = f"regular_{n}_{cover_size}_{instance_index}.dimacs"
                        filepath = os.path.join(output_dir, filename)

                        # Sauvegarde
                        save_graph_to_dimacs(graph, cover_size, filepath)

                        # Écriture dans le CSV
                        writer.writerow([
                            filename,
                            n,
                            nb_edges,
                            cover_size,
                            deg_min,
                            deg_max,
                            d,
                            FIXED_SEED,
                            status
                        ])
                    else:
                        # Si le solveur n'est pas Optimal (cas rare), on ne sauvegarde pas.
                        pass

                    instance_index += 1
                    pbar.update(1)

    print("Terminé. Les graphes réguliers et le CSV sont disponibles dans :", output_dir)


if __name__ == "__main__":
    main()
