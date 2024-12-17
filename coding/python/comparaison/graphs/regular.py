import os
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

# Liste des graines fixes pour garantir la reproductibilité
FIXED_SEEDS = [42, 43, 44, 45, 46]

def minimum_vertex_cover(graph: nx.Graph):
    """
    Calcule la solution optimale du minimum vertex cover pour un graphe donné.
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

def generate_regular_graph(num_nodes, degree, seed):
    """
    Génère un graphe régulier avec une graine fixe.
    """
    if degree >= num_nodes or degree < 0:
        raise ValueError("Le degré doit être >= 0 et < nombre de sommets.")
    return nx.random_regular_graph(degree, num_nodes, seed=seed)

def is_valid_graph(graph):
    """
    Vérifie que le graphe est connexe.
    """
    return nx.is_connected(graph)

def save_graph_to_dimacs(graph, cover_size, filename):
    """
    Sauvegarde le graphe au format DIMACS.
    """
    with open(filename, "w") as f:
        f.write(f"p edge {graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")  # DIMACS utilise des index 1-based

if __name__ == "__main__":
    # Configuration des paramètres
    output_dir = "dimacs_files/regular"
    os.makedirs(output_dir, exist_ok=True)

    num_nodes = 10  # Taille initiale des graphes
    degree = 3      # Degré initial des sommets
    graph_name_base = "regular"

    for seed in FIXED_SEEDS:  # Pour chaque graine fixe
        while num_nodes <= 200:  # Limite de la taille des graphes
            try:
                # Ajuster légèrement le degré pour diversifier
                varied_degree = degree + (num_nodes % 2)  # Alterner entre d et d+1
                print(f"Essai avec num_nodes={num_nodes}, degree={varied_degree}, seed={seed}")

                # Générer le graphe
                graph = generate_regular_graph(num_nodes, varied_degree, seed)

                # Vérifier que le graphe est connexe
                if not is_valid_graph(graph):
                    print(f"Graphe ignoré : non connexe pour num_nodes={num_nodes}, degree={varied_degree}")
                    num_nodes += 5
                    continue

                # Calcul du minimum vertex cover
                solution, cover_size, status = minimum_vertex_cover(graph)

                if status != "Optimal":
                    print("Arrêt : Le solveur n'a pas trouvé une solution optimale.")
                    break

                # Nom du fichier DIMACS
                filename = f"{graph_name_base}-{num_nodes}-{cover_size}.dimacs"
                filepath = os.path.join(output_dir, filename)

                # Sauvegarde au format DIMACS
                save_graph_to_dimacs(graph, cover_size, filepath)
                print(f"Fichier sauvegardé : {filepath}, couverture minimale : {cover_size}")

                # Incrémenter la taille des graphes
                num_nodes += 5

            except Exception as e:
                print(f"Erreur : {e}")
                break
