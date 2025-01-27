import os
import csv
import networkx as nx
from tqdm import tqdm

# On importe la fonction de résolution du Minimum Vertex Cover
# depuis votre module solveur.py
from main.solveur.solveur import minimum_vertex_cover

# Constantes et paramètres
FIXED_SEED = 42
NODE_SIZES = [20, 40, 60, 80, 100]  # Tailles des graphes
EDGE_PROBABILITIES = [0.1, 0.2, 0.3,
                      0.4]  # Probabilités d'arête
GRAPHS_PER_COMBINATION = 30  # Nombre de graphes à générer par couple (n, p)
# # node_size * edge_probabilites * graphs_per_combination = nombre de graphes générés
# 450 graphes en tout

def generate_erdos_renyi_graph(num_nodes, prob_connection, seed=FIXED_SEED):
    """
    Génère un graphe G(n, p) selon le modèle d’Erdős-Rényi.
    On fixe la seed pour la reproductibilité.
    """
    return nx.erdos_renyi_graph(num_nodes, prob_connection, seed=seed)


def save_graph_to_dimacs(graph: nx.Graph, filename: str):
    """
    Sauvegarde le graphe 'graph' au format DIMACS,
    numérotation des sommets de 1 à n.
    Format .dimacs non orienté :
      p edge <nb_sommets> <nb_arêtes>
      e u v
    """
    with open(filename, "w") as f:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        f.write(f"p edge {n} {m}\n")
        # DIMACS utilise un index 1-based, d'où le "+ 1" :
        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")


def main():
    # Répertoire de sortie
    output_dir = "dimacs_files/erdos_renyi"
    os.makedirs(output_dir, exist_ok=True)

    # Fichier CSV pour stocker les propriétés de construction et résultats
    csv_filename = os.path.join(output_dir, "erdos_renyi_properties.csv")

    # Calcul du nombre total de graphes à générer
    total_combinations = len(NODE_SIZES) * len(EDGE_PROBABILITIES) * GRAPHS_PER_COMBINATION

    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        # En-tête : ajustez selon les propriétés que vous souhaitez
        writer.writerow([
            "instance_file",
            "n_sommets",
            "nb_aretes",
            "cover_size",
            "deg_min",
            "deg_max",
            "prob_connection",
            "seed",
            "status"
        ])

        # Barre de progression
        with tqdm(total=total_combinations, desc="Génération des graphes", unit="graphe") as pbar:
            instance_count = 0

            for num_nodes in NODE_SIZES:
                for prob_connection in EDGE_PROBABILITIES:
                    for i in range(1, GRAPHS_PER_COMBINATION + 1):
                        # Génération d'un graphe G(n, p)
                        graph = generate_erdos_renyi_graph(num_nodes, prob_connection, seed=FIXED_SEED + i)

                        # Résolution du Minimum Vertex Cover
                        solution, cover_size, status = minimum_vertex_cover(graph)

                        if status == "Optimal":
                            # Calcul des propriétés
                            nb_edges = graph.number_of_edges()
                            degrees = [deg for _, deg in graph.degree()]
                            deg_min = min(degrees)
                            deg_max = max(degrees)

                            # Nom du fichier DIMACS, par exemple
                            filename = f"erdosReniy_{num_nodes}_{int(cover_size)}_{i}.dimacs"
                            filepath = os.path.join(output_dir, filename)

                            # Sauvegarde au format DIMACS
                            save_graph_to_dimacs(graph, filepath)

                            # Écriture dans le CSV
                            writer.writerow([
                                filename,
                                num_nodes,
                                nb_edges,
                                int(cover_size),
                                deg_min,
                                deg_max,
                                prob_connection,
                                FIXED_SEED,
                                status
                            ])
                        else:
                            # Si pas Optimal, on passe simplement
                            pass

                        instance_count += 1
                        pbar.update(1)

    print(f"\nTerminé. Un total de {instance_count} instances ont été tentées.")
    print(f"Les fichiers DIMACS et le CSV sont disponibles dans : {output_dir}")


if __name__ == "__main__":
    main()
