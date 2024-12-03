import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

# Ce script permet de trouver la solution exacte d'un graphe de taille raisonnable au format dimacs

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


def load_graph_from_dimacs(filename):
    """
    Charge un graphe depuis un fichier DIMACS.

    Args:
        filename (str): Chemin du fichier DIMACS.

    Returns:
        nx.Graph: Le graphe chargé.
    """
    graph = nx.Graph()
    with open(filename, "r") as f:
        for line in f:
            # Ignorer les commentaires et les lignes vides
            line = line.strip()
            if line.startswith("c") or line == "":
                continue
            # Lire la ligne d'entête
            if line.startswith("p"):
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError("Ligne d'entête mal formatée dans le fichier DIMACS.")
                _, _, num_nodes, num_edges, _ = parts[:5]
                num_nodes = int(num_nodes)
                num_edges = int(num_edges)
                # Ajouter les sommets au graphe (inutile pour networkx mais explicite)
                graph.add_nodes_from(range(1, num_nodes + 1))
            # Lire les arêtes
            elif line.startswith("e"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError("Ligne d'arête mal formatée dans le fichier DIMACS.")
                _, u, v = parts[:3]
                graph.add_edge(int(u), int(v))
    return graph


if __name__ == "__main__":
    # Spécifiez le fichier DIMACS à charger
    dimacs_file = "dimacs_files/anti_mdg/anti_mdg_k5/anti_mdg_k5_instance1.dimacs"

    try:
        # Charger le graphe
        graph = load_graph_from_dimacs(dimacs_file)

        # Calculer la couverture minimale
        solution, cover_size, status = minimum_vertex_cover(graph)

        # Afficher la solution
        print(f"Statut de la résolution : {status}")
        print(f"Taille de la couverture minimale : {cover_size}")
        print("Sommets dans la couverture minimale :")
        print([v for v, included in solution.items() if included])

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{dimacs_file}' n'existe pas.")
    except ValueError as e:
        print(f"Erreur : {e}")
