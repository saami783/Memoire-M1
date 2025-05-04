from .dfs import dfs
from .edge_deletion import *
from .gic import greedy_independent_cover
from .malatya import malatya_vertex_cover
from .matching_based import matching_based
from .mdg import maximum_degree_greedy
from .slf import sorted_list_left
from .slr import sorted_list_right
from .ks_vc import ks_vc

import networkx as nx

def calculate_statistics(solution_sizes):
    """Calcule les statistiques (moyenne, min, max) pour une liste de tailles de solutions."""
    return {
        "avg_size": sum(solution_sizes) / len(solution_sizes),
        "min_size": min(solution_sizes),
        "max_size": max(solution_sizes),
    }

def evaluate_algorithm(graph, num_runs=300, verbose=True):
    if not isinstance(graph, nx.Graph):
        print(f"Erreur : Le graphe n'est pas de type nx.Graph")
        return []

    results = {
        # "DFS": [],
        # "Edge deletion": [],
        # "Edge deletion smart": [],
        "ED Sum Max Deg": [],
        # "Greedy Independent Cover": [],
        # "Malatya": [],
        # "Matching Based": [],
        # "Maximum Degree Greedy": [],
        # "KS VC": [],
        # "Sorted ListLeft": [],
        #  "Sorted ListRight": [],

        # "Genetic": [],
        # "max_A": [],
        # "max_AR": [],
    }

    for run in range(1, num_runs + 1):
        # Approximations, heuristiques et métaheuristiques
        # results["ILS"].append(len(ils(graph)))

        # results["DFS"].append(len(dfs(graph)))
        # results["Edge deletion"].append(len(edge_deletion(graph)))
        # results["Edge deletion smart"].append(len(edge_deletion_smart(graph)))
        results["ED Sum Max Deg"].append(len(edge_deletion_max_degree_sum(graph)))
        # results["Greedy Independent Cover"].append(len(greedy_independent_cover(graph)))
        # results["Malatya"].append(len(malatya_vertex_cover(graph)))
        # results["Matching Based"].append(len(matching_based(graph)))
        # results["Maximum Degree Greedy"].append(len(maximum_degree_greedy(graph)))
        # results["KS VC"].append(len(ks_vc(graph)))
        # results["Sorted ListLeft"].append(len(sorted_list_left(graph)))
        # results["Sorted ListRight"].append(len(sorted_list_right(graph)))

        # solution, size = genetic(graph)
        # results["Genetic"].append(size)

        # results["max_A"].append(len(maxA(graph)))
        # results["max_AR"].append(len(maxAR(graph)))

        # results["KS VC"].append(len(ks_vc(graph)))
        # results["PSO"].append(len(pso(graph)))
        # results["Harmony Search"].append(len(harmony_search_(graph)))
        # results["Memetic"].append(len(memetic(graph)))
    return {
        heuristic: calculate_statistics(sizes)
        for heuristic, sizes in results.items()
    }

def process_graph(id, instance_number, graph_class, filename, graph, opt_size,  num_nodes, num_edges, verbose=True):
    """Évalue les heuristiques sur un graphe donné et retourne les résultats."""
    heuristic_results = evaluate_algorithm(graph, verbose=verbose)
    results = []
    for heuristic, res in heuristic_results.items():
        results.append({
            "Id": id,
            "Instance": instance_number,
            "Class": graph_class,
            "Graph": filename,
            "Nodes": num_nodes,
            "Edges": num_edges,
            "Optimal_Size": opt_size,
            "Heuristic": heuristic,
            "Average_Size": res["avg_size"],
            "Best_Size": res["min_size"],
            "Worst_Size": res["max_size"],
        })
    return results
