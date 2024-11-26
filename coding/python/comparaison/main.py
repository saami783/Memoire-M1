import random
import pandas as pd

SEED = 42
random.seed(SEED)

def maximum_degree_greedy(graph):
    C = set()
    temp_graph = graph.copy()

    while temp_graph.number_of_edges() > 0:
        degrees = dict(temp_graph.degree())
        max_degree = max(degrees.values())
        max_degree_nodes = [node for node, degree in degrees.items() if degree == max_degree]
        max_degree_node = random.choice(max_degree_nodes)

        C.add(max_degree_node)
        temp_graph.remove_node(max_degree_node)

    return list(C)

def greedy_independent_cover(graph):
    C = set()
    temp_graph = graph.copy()

    while temp_graph.number_of_edges() > 0:
        degrees = dict(temp_graph.degree())
        min_degree = min(degrees.values())
        min_degree_nodes = [node for node, degree in degrees.items() if degree == min_degree]
        min_degree_node = random.choice(min_degree_nodes)

        neighbors = set(temp_graph.neighbors(min_degree_node))
        C.update(neighbors)

        temp_graph.remove_nodes_from(neighbors | {min_degree_node})

    return list(C)

# Fonction pour évaluer et comparer les algorithmes sur un graphe donné
def evaluate_algorithms(graph, num_runs=100, optimal_size=None):
    mdg_data = []
    gic_data = []

    for i in range(1, num_runs + 1):
        random.seed(random.randint(1, 10000))

        mdg_solution = maximum_degree_greedy(graph)
        gic_solution = greedy_independent_cover(graph)

        # Collecte des données
        mdg_data.append({
            "Run Number": i,
            "Solution Size": len(mdg_solution),
            "Approximation Ratio": '',
        })

        gic_data.append({
            "Run Number": i,
            "Solution Size": len(gic_solution),
            "Approximation Ratio": '',
        })

    mdg_df = pd.DataFrame(mdg_data)
    gic_df = pd.DataFrame(gic_data)

    return mdg_df, gic_df
