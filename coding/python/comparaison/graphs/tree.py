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

def generate_random_tree(num_nodes, seed):
    """
    Génère un arbre aléatoire avec une graine fixe.
    """
    if num_nodes < 2:
        raise ValueError("Un arbre doit avoir au moins 2 sommets.")
    return nx.random_tree(num_nodes, seed=seed)

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
    output_dir = "dimacs_files/trees"
    os.makedirs(output_dir, exist_ok=True)

    num_nodes = 10  # Taille initiale des arbres
    graph_name_base = "tree"

    for seed in FIXED_SEEDS:  # Pour chaque graine fixe
        while num_nodes <= 200:  # Taille maximale des arbres
            try:
                print(f"Génération d'un arbre avec num_nodes={num_nodes}, seed={seed}")

                # Génération de l'arbre aléatoire
                graph = generate_random_tree(num_nodes, seed)

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

                # Augmentation progressive de la taille des arbres
                num_nodes += 5

            except Exception as e:
                print(f"Erreur : {e}")
                break
