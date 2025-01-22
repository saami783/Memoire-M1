import os
import networkx as nx
from main.solveur.solveur import minimum_vertex_cover

def generate_barabasi_albert_graph(num_nodes, num_edges):
    """
    Génère un graphe Barabási-Albert.

    Args:
        num_nodes (int): Nombre total de sommets dans le graphe.
        num_edges (int): Nombre d'arêtes à attacher à chaque nouveau sommet.

    Returns:
        nx.Graph: Le graphe généré.
    """
    if num_edges < 1 or num_edges >= num_nodes:
        raise ValueError("Le nombre d'arêtes doit être >= 1 et < nombre de sommets.")
    return nx.barabasi_albert_graph(num_nodes, num_edges, seed=42)

def save_graph_to_dimacs(graph, cover_size, filename):
    """
    Sauvegarde un graphe au format DIMACS.

    Args:
        graph (nx.Graph): Le graphe à sauvegarder.
        cover_size (int): Taille de la couverture minimale.
        filename (str): Nom du fichier pour sauvegarder le graphe.
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    with open(filename, "w") as f:
        # Entête DIMACS
        f.write(f"p edge {num_nodes} {num_edges}\n")
        # Liste des arêtes
        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")  # DIMACS utilise des index 1-based

if __name__ == "__main__":
    # Dossier pour les fichiers DIMACS
    output_dir = "dimacs_files/barabasi_albert"
    os.makedirs(output_dir, exist_ok=True)

    num_nodes = 10  # Nombre initial de sommets
    num_edges = 2   # Nombre d'arêtes à attacher à chaque nouveau sommet
    graph_name_base = "barabasi_albert"  # Nom de base pour les fichiers DIMACS

    while True:
        try:
            # Génération du graphe Barabási-Albert
            graph = generate_barabasi_albert_graph(num_nodes, num_edges)

            # Calcul du minimum vertex cover
            solution, cover_size, status = minimum_vertex_cover(graph)

            # Vérification du statut
            if status != "Optimal":
                print("Arrêt : Le solveur n'a pas trouvé une solution optimale.")
                break

            # Nom du fichier avec taille et solution optimale
            filename = f"{graph_name_base}-{num_nodes}-{cover_size}.dimacs"
            filepath = os.path.join(output_dir, filename)

            # Sauvegarde au format DIMACS
            save_graph_to_dimacs(graph, cover_size, filepath)
            print(f"Fichier sauvegardé : {filepath}")

            # Affichage des résultats
            print(f"Graphe avec {num_nodes} sommets, {graph.number_of_edges()} arêtes, taille de couverture minimale : {cover_size}")

            # Augmentation du nombre de sommets pour générer un graphe plus grand
            num_nodes += 5

        except ValueError as e:
            print(f"Erreur : {e}")
            break

        except Exception as e:
            print(f"Erreur inattendue : {e}")
            break
