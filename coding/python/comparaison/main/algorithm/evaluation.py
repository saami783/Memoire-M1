# from .fpt import fpt  # <--- je retire l'algorithme FPT, car il peut fournir une solution exacte
# from .tabu_search import tabu_search_

from .dfs import dfs
from .gic import greedy_independent_cover
from .mdg import maximum_degree_greedy
from .slf import sorted_list_left
from .slr import sorted_list_right
from .aco import aco
from .approximate_matching import approximate_matching
from .bar_yehuda_even import bar_yehuda_even
from .degree_reduction import degree_reduction
from .edge_deletion import edge_deletion
from .genetic import genetic
from .harmony_search import harmony_search_
from .ils import ils
from .local_ratio import local_ratio
from .local_search import local_search
from .lp_rounding import lp_rounding
from .matching_based import matching_based
from .memetic import memetic
from .minimum_degree_greedy import minimum_degree_greedy
from .primal_dual import primal_dual
from .pso import pso
from .vns import vns

def calculate_statistics(solution_sizes):
    """Calcule les statistiques (moyenne, min, max) pour une liste de tailles de solutions."""
    return {
        "avg_size": sum(solution_sizes) / len(solution_sizes),
        "min_size": min(solution_sizes),
        "max_size": max(solution_sizes),
    }

def evaluate_algorithm(graph, num_runs=300, verbose=True):
    results = {
        # "FPT": [],  # <--- Retiré
        # "Tabu": [],  # <--- Retiré très très long
        # "LP Rounding": [],

        # "Maximum Degree Greedy": [],
        # "Greedy Independent Cover": [],
        # "Sorted ListLeft": [],
        # "Sorted ListRight": [],
        # "DFS": [],
        # "Matching Based": [],
        # "Minimum Degree Greedy": [],
        # "Primal Dual": [],
        # "PSO": [],
        # "Degree Reduction": [],
        # "Bar Yehuda Even": [],
        # "Approximate matching": [],
        # "Edge Deletion": [],
        # "Harmony Search": [],
        # "Local Search Ratio": [],
        # "ACO": [],

        "Local Search": [], # en cours depuis 19h15
        "ILS": [],  # lancé à 21H25
        "VNS": [],  # lancé à 21H25
        "Memetic": [], # lancé à 21H36
        "Genetic": [],  # lancé à 21H36
    }

    for run in range(1, num_runs + 1):
        # Approximations, heuristiques et métaheuristiques
        # results["FPT"].append(len(fpt(graph))) # <--- Commenté
        # results["Tabu"].append(len(tabu_search_(graph)))
        # results["LP Rounding"].append(len(lp_rounding(graph)))

        # results["Maximum Degree Greedy"].append(len(maximum_degree_greedy(graph)))
        # results["Greedy Independent Cover"].append(len(greedy_independent_cover(graph)))
        # results["Sorted ListLeft"].append(len(sorted_list_left(graph)))
        # results["Sorted ListRight"].append(len(sorted_list_right(graph)))
        # results["DFS"].append(len(dfs(graph)))
        # results["Matching Based"].append(len(matching_based(graph)))
        # results["Minimum Degree Greedy"].append(len(minimum_degree_greedy(graph)))
        # results["Primal Dual"].append(len(primal_dual(graph)))
        # results["PSO"].append(len(pso(graph)))
        # results["Degree Reduction"].append(len(degree_reduction(graph)))
        # results["Bar Yehuda Even"].append(len(bar_yehuda_even(graph)))
        # results["Approximate matching"].append(len(approximate_matching(graph)))
        # results["Edge Deletion"].append(len(edge_deletion(graph)))
        # results["Harmony Search"].append(len(harmony_search_(graph)))
        # results["Local Search Ratio"].append(len(local_ratio(graph)))
        # results["ACO"].append(len(aco(graph)))

        results["Local Search"].append(len(local_search(graph)))
        results["ILS"].append(len(ils(graph)))
        results["VNS"].append(len(vns(graph)))
        results["Memetic"].append(len(memetic(graph)))
        results["Genetic"].append(len(genetic(graph)))
    return {
        heuristic: calculate_statistics(sizes)
        for heuristic, sizes in results.items()
    }

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
