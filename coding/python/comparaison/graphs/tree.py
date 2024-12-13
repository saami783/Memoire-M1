import os
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus


def minimum_vertex_cover(graph: nx.Graph):
    """
    Calcule la solution optimale du minimum vertex cover pour un graphe donné en utilisant PuLP.

    Args:
        graph (nx.Graph): Un graphe networkx non orienté.

    Returns:
        dict: Contient les sommets dans la couverture minimale (clé: sommet, valeur: 1 si inclus, 0 sinon).
        float: La taille de la couverture minimale.
        str: Le statut de la résolution (par exemple, 'Optimal').
    """
    if graph.is_directed():
        raise ValueError("Le graphe doit être non orienté.")

    prob = LpProblem("MinimumVertexCover", LpMinimize)

    vertex_vars = {v: LpVariable(f"x_{v}", cat="Binary") for v in graph.nodes()}
    prob += lpSum(vertex_vars[v] for v in graph.nodes()), "MinimizeCoverSize"

    for u, v in graph.edges():
        prob += vertex_vars[u] + vertex_vars[v] >= 1, f"Edge_{u}_{v}_Covered"

    prob.solve()

    status = LpStatus[prob.status]
    solution = {v: int(vertex_vars[v].value()) for v in graph.nodes()}
    cover_size = sum(solution[v] for v in graph.nodes())

    return solution, cover_size, status


def generate_random_tree(num_nodes):
    """
    Génère un arbre aléatoire.

    Args:
        num_nodes (int): Nombre total de sommets dans l'arbre.

    Returns:
        nx.Graph: L'arbre généré.
    """
    if num_nodes < 2:
        raise ValueError("Un arbre doit avoir au moins 2 sommets.")
    return nx.random_tree(num_nodes)


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
        f.write(f"p edge {num_nodes} {num_edges} {cover_size}\n")
        # Liste des arêtes
        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")  # DIMACS utilise des index 1-based


if __name__ == "__main__":
    # Dossier pour les fichiers DIMACS
    output_dir = "dimacs_files/trees"
    os.makedirs(output_dir, exist_ok=True)

    num_nodes = 10  # Nombre initial de sommets
    graph_name_base = "tree"  # Nom de base pour les fichiers DIMACS

    while True:
        try:
            # Génération de l'arbre aléatoire
            graph = generate_random_tree(num_nodes)

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

            # Augmentation du nombre de sommets pour générer un arbre plus grand
            num_nodes += 5

        except ValueError as e:
            print(f"Erreur : {e}")
            break