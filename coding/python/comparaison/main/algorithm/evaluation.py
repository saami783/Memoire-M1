from .dfs import dfs_heuristic
from .gic import greedy_independent_cover
from .mdg import maximum_degree_greedy
from .slf import sorted_list_left
from .slr import sorted_list_right


def calculate_statistics(solution_sizes):
    """Calcule les statistiques pour une liste de tailles de solution."""
    return {
        "avg_size": sum(solution_sizes) / len(solution_sizes),
        "min_size": min(solution_sizes),
        "max_size": max(solution_sizes),
    }

def evaluate_algorithm(graph, num_runs=300, verbose=True):
    results = {
        "Maximum Degree Greedy": [],
        "Greedy Independent Cover": [],
        "Sorted ListLeft": [],
        "Sorted ListRight": [],
        "DFS": []
    }

    for run in range(1, num_runs + 1):
        if verbose:
            print(f"  Run {run}/{num_runs}", end="\r", flush=True)
        results["Maximum Degree Greedy"].append(len(maximum_degree_greedy(graph)))
        results["Greedy Independent Cover"].append(len(greedy_independent_cover(graph)))
        results["Sorted ListLeft"].append(len(sorted_list_left(graph)))
        results["Sorted ListRight"].append(len(sorted_list_right(graph)))
        results["DFS"].append(len(dfs_heuristic(graph)))

    return {heuristic: calculate_statistics(sizes) for heuristic, sizes in results.items()}

def process_graph(filename, graph, opt_size, verbose=True):
    """Évalue les heuristiques sur un graphe donné et retourne les résultats."""
    heuristic_results = evaluate_algorithm(graph, verbose=verbose)

    results = []
    for heuristic, res in heuristic_results.items():
        results.append({
            "Graph": filename,
            "Nodes": graph.number_of_nodes(),
            "Optimal_Size": opt_size,
            "Heuristic": heuristic,
            "Average_Size": res["avg_size"],
            "Best_Size": res["min_size"],
            "Worst_Size": res["max_size"],
        })
    return results