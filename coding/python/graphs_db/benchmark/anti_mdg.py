import os
import networkx as nx
import random
from main.solveur.solveur import minimum_vertex_cover

# Fonction pour générer un graphe Anti-MDG
def generate_anti_mdg_graph(k, p):
    """
    Génère un graphe Anti-MDG de taille k avec une probabilité de connexion p.

    Args:
        k (int): Taille des ensembles A et B.
        p (float): Probabilité de connexion pour ajouter de la variabilité.

    Returns:
        nx.Graph: Le graphe Anti-MDG généré.
    """
    G = nx.Graph()

    # Ajouter les ensembles A, B et C
    A = list(range(1, k + 1))  # Sommets A numérotés de 1 à k
    B = list(range(k + 1, 2 * k + 1))  # Sommets B numérotés de k+1 à 2k
    C = []

    G.add_nodes_from(A + B)

    # Couplage initial entre A et B
    for i in range(k):
        G.add_edge(A[i], B[i])

    # Ajouter les sommets de C avec des degrés croissants
    current_node = 2 * k + 1
    for degree in range(2, k + 1):
        num_nodes = max(1, k // (2 * degree))  # Limiter le nombre de sommets ajoutés
        for _ in range(num_nodes):
            G.add_node(current_node)
            C.append(current_node)
            # Connecter aléatoirement le sommet C aux sommets de B avec probabilité p
            connected_nodes = random.sample(B, min(len(B), max(2, int(len(B) * p))))
            for b in connected_nodes:
                G.add_edge(current_node, b)
            current_node += 1

    return G

# Fonction pour sauvegarder un graphe au format DIMACS
def save_graph_as_dimacs(graph, filename):
    """
    Sauvegarde un graphe NetworkX au format DIMACS.

    Args:
        graph (nx.Graph): Le graphe à sauvegarder.
        filename (str): Nom du fichier DIMACS.
    """
    with open(filename, "w") as f:
        f.write(f"p edge {len(graph.nodes())} {len(graph.edges())}\n")
        for u, v in graph.edges():
            f.write(f"e {u} {v}\n")

# Fonction principale
if __name__ == "__main__":
    k = 6  # Taille initiale
    p = 0.5  # Probabilité initiale

    while True:  # Boucle infinie pour générer des graphes
        # Générer le graphe Anti-MDG
        graph = generate_anti_mdg_graph(k, p)

        # Résoudre le problème du minimum vertex cover
        solution, cover_size, status = minimum_vertex_cover(graph)

        if status != "Optimal":
            print(f"Arrêt de la génération : Problème non résolu de manière optimale pour k={k}")
            break

        # Sauvegarder le graphe au format DIMACS
        os.makedirs("dimacs_files/anti-mdg", exist_ok=True)
        filename = f"dimacs_files/anti_mdg/anti-mdg_{len(graph.nodes())}_{cover_size}.dimacs"
        save_graph_as_dimacs(graph, filename)
        print(f"Graphe sauvegardé dans {filename} avec |V|={len(graph.nodes())} et solution optimale={cover_size}.")

        # Augmenter la taille et réduire la probabilité légèrement
        k += 5
        p = max(0.3, p - 0.05)
