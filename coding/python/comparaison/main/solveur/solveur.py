from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
import networkx as nx
import pulp
pulp.LpSolverDefault.msg = 0

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

    # Variables binaires pour chaque sommet
    vertex_vars = {v: LpVariable(f"x_{v}", cat="Binary") for v in graph.nodes()}

    # Fonction objectif : minimiser la somme des variables binaires
    prob += lpSum(vertex_vars[v] for v in graph.nodes()), "MinimizeCoverSize"

    # Contraintes : chaque arête doit être couverte
    for u, v in graph.edges():
        prob += vertex_vars[u] + vertex_vars[v] >= 1, f"Edge_{u}_{v}_Covered"

    prob.solve()

    status = LpStatus[prob.status]
    if status != "Optimal":
        print(f"Le solveur n'a pas trouvé la solution exacte. Statut = {status}")
        return None

    solution = {v: int(vertex_vars[v].value()) for v in graph.nodes()}
    cover_size = sum(solution[v] for v in graph.nodes())

    return cover_size, status