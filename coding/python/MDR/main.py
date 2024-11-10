import random
import networkx as nx
import pandas as pd
import time
import os

# Initialiser la graine pour assurer la reproductibilité des tests
SEED = 42
random.seed(SEED)

# Fonction de l'algorithme Maximum Degree Greedy
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

# Fonction de l'algorithme Greedy Independent Cover
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

# Fonction pour associer l'optimal MVC en fonction du fichier
def get_optimal_size(file_name):
    if 'frb30-15' in file_name:
        return 420
    elif 'frb40-19' in file_name:
        return 720
    elif 'frb50-23' in file_name:
        return 1100
    elif 'frb59-26' in file_name:
        return 1475
    else:
        return None  # Si le fichier ne correspond pas aux catégories connues

# Fonction pour évaluer et comparer les algorithmes sur un graphe donné
def evaluate_algorithms(graph, num_runs=100, optimal_size=None):
    mdg_data = []
    gic_data = []

    for i in range(1, num_runs + 1):
        random.seed(random.randint(1, 10000))

        # Exécution et mesure du temps pour MDG
        start_time = time.time()
        mdg_solution = maximum_degree_greedy(graph)
        mdg_time = time.time() - start_time

        # Exécution et mesure du temps pour GIC
        start_time = time.time()
        gic_solution = greedy_independent_cover(graph)
        gic_time = time.time() - start_time

        # Collecte des données
        mdg_data.append({
            "Run Number": i,
            "Solution Size": len(mdg_solution),
            "Approximation Ratio": len(mdg_solution) / optimal_size if optimal_size else None,
            "Execution Time (s)": mdg_time
        })

        gic_data.append({
            "Run Number": i,
            "Solution Size": len(gic_solution),
            "Approximation Ratio": len(gic_solution) / optimal_size if optimal_size else None,
            "Execution Time (s)": gic_time
        })

    mdg_df = pd.DataFrame(mdg_data)
    gic_df = pd.DataFrame(gic_data)

    return mdg_df, gic_df

# Fonction principale pour parcourir tous les benchmarks nettoyés
def main():
    root_dir = 'benchmarks'

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_cleaned.dimacs'):
                file_path = os.path.join(subdir, file)
                optimal_size = get_optimal_size(file)

                if optimal_size is None:
                    print(f"Optimal size not found for {file}. Skipping.")
                    continue

                print(f"Processing {file_path} with optimal MVC size: {optimal_size}...")

                # Charger le graphe
                graph = nx.read_edgelist(file_path, nodetype=int)

                # Évaluer sur un échantillon de 100 exécutions
                mdg_df, gic_df = evaluate_algorithms(graph, num_runs=1000, optimal_size=optimal_size)

                # Enregistrer les résultats dans des fichiers CSV
                mdg_output_path = f'mdg_results_{file.replace(".dimacs", ".csv")}'
                gic_output_path = f'gic_results_{file.replace(".dimacs", ".csv")}'

                mdg_df.to_csv(mdg_output_path, index=False)
                gic_df.to_csv(gic_output_path, index=False)

                print(f"Résultats enregistrés pour {file} :")
                print(f"  - MDG: {mdg_output_path}")
                print(f"  - GIC: {gic_output_path}")

if __name__ == "__main__":
    main()