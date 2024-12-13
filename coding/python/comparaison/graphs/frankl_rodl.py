import os
import random
import pandas as pd
import networkx as nx

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

def sorted_list_left(graph):
    C = set()
    degrees = dict(graph.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: -degrees[x])  # Trie décroissant par degré

    for u in sorted_nodes:  # Parcourir la liste de gauche à droite
        if any(neighbor not in C for neighbor in graph.neighbors(u)):
            C.add(u)

    return list(C)

def sorted_list_right(graph):
    C = set()
    degrees = dict(graph.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: -degrees[x])  # Trie décroissant par degré

    for u in reversed(sorted_nodes):  # Parcourir la liste de droite à gauche
        if any(neighbor not in C for neighbor in graph.neighbors(u)):
            C.add(u)

    return list(C)

def dfs_heuristic(graph):
    dfs_tree = nx.dfs_tree(graph)  # Crée un arbre DFS
    internal_nodes = [node for node in dfs_tree.nodes if dfs_tree.degree[node] > 1]  # Nœuds internes
    return internal_nodes

def load_graph_from_dimacs(filename):
    """
    Charge un graphe depuis un fichier DIMACS.
    """
    graph = nx.Graph()
    node_mapping = {}  # Pour mapper les noms non numérotés à des entiers
    next_id = 1

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("c") or line == "":
                continue
            if line.startswith("p"):
                parts = line.split()
                num_nodes = int(parts[2])  # Nombre de sommets
                graph.add_nodes_from(range(1, num_nodes + 1))  # Ajout des sommets
            elif line.startswith("e"):
                parts = line.split()
                u, v = parts[1], parts[2]
                # Mapper les sommets non numérotés en entiers
                if u not in node_mapping:
                    node_mapping[u] = next_id
                    next_id += 1
                if v not in node_mapping:
                    node_mapping[v] = next_id
                    next_id += 1
                graph.add_edge(node_mapping[u], node_mapping[v])
    return graph

def evaluate_heuristics(graph, num_runs=300):
    mdg_sizes = []
    gic_sizes = []
    left_sizes = []
    right_sizes = []
    dfs_sizes = []

    for run in range(1, num_runs + 1):
        print(f"  Run {run}/{num_runs}", end="\r", flush=True)  # Indique la run en cours
        mdg_solution = maximum_degree_greedy(graph)
        gic_solution = greedy_independent_cover(graph)
        left_solution = sorted_list_left(graph)
        right_solution = sorted_list_right(graph)
        dfs_solution = dfs_heuristic(graph)

        mdg_sizes.append(len(mdg_solution))
        gic_sizes.append(len(gic_solution))
        left_sizes.append(len(left_solution))
        right_sizes.append(len(right_solution))
        dfs_sizes.append(len(dfs_solution))

    # Résultats pour chaque heuristique
    mdg_results = {
        "avg_size": sum(mdg_sizes) / num_runs,
        "min_size": min(mdg_sizes),
        "max_size": max(mdg_sizes),
    }

    gic_results = {
        "avg_size": sum(gic_sizes) / num_runs,
        "min_size": min(gic_sizes),
        "max_size": max(gic_sizes),
    }

    left_results = {
        "avg_size": sum(left_sizes) / num_runs,
        "min_size": min(left_sizes),
        "max_size": max(left_sizes),
    }

    right_results = {
        "avg_size": sum(right_sizes) / num_runs,
        "min_size": min(right_sizes),
        "max_size": max(right_sizes),
    }

    dfs_results = {
        "avg_size": sum(dfs_sizes) / num_runs,
        "min_size": min(dfs_sizes),
        "max_size": max(dfs_sizes),
    }

    return mdg_results, gic_results, left_results, right_results, dfs_results

if __name__ == "__main__":
    # Dossier contenant les graphes DIMACS
    input_dir = "../dimacs_files/frankl_rodl"
    output_file = "../dump/out/result_frankl_rodl.csv"

    # Vérifier si le dossier existe
    if not os.path.exists(input_dir):
        print(f"Erreur : Le dossier '{input_dir}' n'existe pas.")
        exit(1)

    results = []

    # Parcourir tous les fichiers DIMACS en ordre croissant de taille de graphe
    file_list = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".dimacs")],
        key=lambda x: int(x.split("-")[1][1:])  # Extraire le numéro après "n" dans la partie "nXX"
    )

    for idx, filename in enumerate(file_list, start=1):
        filepath = os.path.join(input_dir, filename)
        print(f"\nTraitement du fichier ({idx}/{len(file_list)}) : {filename}")

        try:
            # Charger le graphe
            graph = load_graph_from_dimacs(filepath)

            # Évaluer les heuristiques
            mdg_results, gic_results, left_results, right_results, dfs_results = evaluate_heuristics(graph)

            # Extraire les informations du fichier
            graph_name, n_part, k_part, opt_size = filename.replace(".dimacs", "").split("-")
            num_nodes = int(n_part[1:])  # Extraire la valeur après "n"
            opt_size = int(opt_size)

            # Ajouter les résultats au tableau
            results.append({
                "Graph": graph_name,
                "Nodes": num_nodes,
                "Optimal_Size": opt_size,
                "Heuristic": "Maximum Degree Greedy",
                "Average_Size": mdg_results["avg_size"],
                "Best_Size": mdg_results["min_size"],
                "Worst_Size": mdg_results["max_size"]
            })

            results.append({
                "Graph": graph_name,
                "Nodes": num_nodes,
                "Optimal_Size": opt_size,
                "Heuristic": "Greedy Independent Cover",
                "Average_Size": gic_results["avg_size"],
                "Best_Size": gic_results["min_size"],
                "Worst_Size": gic_results["max_size"]
            })

            results.append({
                "Graph": graph_name,
                "Nodes": num_nodes,
                "Optimal_Size": opt_size,
                "Heuristic": "Sorted ListLeft",
                "Average_Size": left_results["avg_size"],
                "Best_Size": left_results["min_size"],
                "Worst_Size": left_results["max_size"]
            })

            results.append({
                "Graph": graph_name,
                "Nodes": num_nodes,
                "Optimal_Size": opt_size,
                "Heuristic": "Sorted ListRight",
                "Average_Size": right_results["avg_size"],
                "Best_Size": right_results["min_size"],
                "Worst_Size": right_results["max_size"]
            })

            results.append({
                "Graph": graph_name,
                "Nodes": num_nodes,
                "Optimal_Size": opt_size,
                "Heuristic": "DFS",
                "Average_Size": dfs_results["avg_size"],
                "Best_Size": dfs_results["min_size"],
                "Worst_Size": dfs_results["max_size"]
            })

        except Exception as e:
            print(f"Erreur lors du traitement du fichier {filename} : {e}")
            continue

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)

    # Sauvegarder les résultats dans un fichier CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nRésultats sauvegardés dans le fichier : {output_file}")
