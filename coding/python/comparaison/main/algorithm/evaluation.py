from .dfs import dfs
from .edge_deletion import *
from .gic import greedy_independent_cover
from .malatya import malatya_vertex_cover
from .matching_based import matching_based
from .mdg import maximum_degree_greedy
from .slf import sorted_list_left
from .slr import sorted_list_right
from .ks_vc import ks_vc
from .aci.inference_aci import aci

import time
import networkx as nx

def calculate_statistics(solution_sizes, times):
    """Calcule les statistiques (moyenne, min, max) pour une liste de tailles de solutions."""
    return {
        "avg_size": sum(solution_sizes) / len(solution_sizes),
        "min_size": min(solution_sizes),
        "max_size": max(solution_sizes),
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "total_time": sum(times),
    }

def evaluate_algorithm(graph, num_runs=300, verbose=True):
    if not isinstance(graph, nx.Graph):
        print(f"Erreur : Le graphe n'est pas de type nx.Graph")
        return {}

    heuristics = {
        # "DFS": dfs,
        # "Edge deletion": edge_deletion,
        # "Edge deletion smart": edge_deletion_smart,
        # "ED Sum Max Deg": edge_deletion_max_degree_sum,
        # "Greedy Independent Cover": greedy_independent_cover,
        # "Malatya": malatya_vertex_cover,
        # "Matching Based": matching_based,
        # "Maximum Degree Greedy": maximum_degree_greedy,
        # "KS VC": ks_vc,
        # "Sorted ListLeft": sorted_list_left,
        # "Sorted ListRight": sorted_list_right,
        "ACI": aci,
    }

    results = {}

    for name, func in heuristics.items():
        sizes = []
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            size = len(func(graph))
            end = time.perf_counter()

            sizes.append(size)
            times.append(end - start)

        results[name] = calculate_statistics(sizes, times)

    return results

def process_graph_from_db(id, instance_number, graph_class, filename, graph, opt_size,  num_nodes, num_edges, verbose=True):
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
            "Average_Time": res["avg_time"],
            "Best_Time": res["min_time"],
            "Worst_Time": res["max_time"],
            "Total_Time": res["total_time"],
        })
    return results

def process_graph_from_hog(id, graph_class, filename, graph, opt_size,  num_nodes, num_edges, verbose=True):

    heuristic_results = evaluate_algorithm(graph, verbose=verbose)
    results = []
    for heuristic, res in heuristic_results.items():
        results.append({
            "Id": id,
            "Class": graph_class,
            "Graph": filename,
            "Nodes": num_nodes,
            "Edges": num_edges,
            "Optimal_Size": opt_size,
            "Heuristic": heuristic,
            "Average_Size": res["avg_size"],
            "Best_Size": res["min_size"],
            "Worst_Size": res["max_size"],
            "Average_Time": res["avg_time"],
            "Best_Time": res["min_time"],
            "Worst_Time": res["max_time"],
            "Total_Time": res["total_time"],
        })
    return results