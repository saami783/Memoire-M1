from .dfs import dfs
from .gic import greedy_independent_cover
from .mdg import maximum_degree_greedy
from .slf import sorted_list_left
from .slr import sorted_list_right
from .edge_deletion import edge_deletion
from .ils import ils
from .memetic import memetic
from .pso import pso
from .matching_based import matching_based

def calculate_statistics(solution_sizes):
    """Calcule les statistiques (moyenne, min, max) pour une liste de tailles de solutions."""
    return {
        "avg_size": sum(solution_sizes) / len(solution_sizes),
        "min_size": min(solution_sizes),
        "max_size": max(solution_sizes),
    }

def evaluate_algorithm(graph, num_runs=300, verbose=True):
    results = {
        "matching_based": [],
    }

    for run in range(1, num_runs + 1):
        # Approximations, heuristiques et métaheuristiques
        results["matching_based"].append(len(greedy_independent_cover(graph)))
        # results["ILS"].append(len(ils(graph)))

        # results["Maximum Degree Greedy"].append(len(maximum_degree_greedy(graph)))
        # results["Greedy Independent Cover"].append(len(greedy_independent_cover(graph)))
        # results["Sorted ListLeft"].append(len(sorted_list_left(graph)))
        # results["Sorted ListRight"].append(len(sorted_list_right(graph)))
        # results["DFS"].append(len(dfs(graph)))
        # results["Matching Based"].append(len(matching_based(graph)))
        # results["PSO"].append(len(pso(graph)))
        # results["Edge Deletion"].append(len(edge_deletion(graph)))
        # results["Harmony Search"].append(len(harmony_search_(graph)))
        # results["Genetic"].append(len(genetic(graph)))
        # results["Memetic"].append(len(memetic(graph)))

    return {
        heuristic: calculate_statistics(sizes)
        for heuristic, sizes in results.items()
    }

def process_graph(filename, graph, opt_size,  num_nodes, num_edges, verbose=True):
    """Évalue les heuristiques sur un graphe donné et retourne les résultats."""
    heuristic_results = evaluate_algorithm(graph, verbose=verbose)
    results = []
    for heuristic, res in heuristic_results.items():
        results.append({
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
