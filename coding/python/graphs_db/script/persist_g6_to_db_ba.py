import os
import sqlite3
import networkx as nx
from pprint import pprint
import time
import numpy as np
import re

def create_database(db_name="graphes.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
       CREATE TABLE IF NOT EXISTS graphes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            graph_name TEXT,
            canonical_form TEXT,
            class TEXT,
            nb_sommets INTEGER,
            nb_aretes INTEGER,
            densite REAL,
            cover_size INTEGER,
            instance_number INTEGER,
            max_degree INTEGER,
            min_degree INTEGER,
            avg_degree REAL,
            num_deg_one INTEGER,
            count_triangles INTEGER,
            total_triangles INTEGER,
            fraction_closed_triangles REAL,
            avg_count_triangles REAL,
            max_count_triangles INTEGER,
            min_count_triangles INTEGER,
            avg_neighbor_deg REAL,
            sum_neighbor_deg REAL,
            degree_assortativity_coefficient REAL,
            diameter INTEGER,
            local_efficiency REAL,
            avg_neighbor_clustering REAL,
            avg_clustering_coefficient REAL,
            is_acyclic INTEGER NOT NULL,
            is_bipartite INTEGER NOT NULL,
            is_connected INTEGER NOT NULL,
            radius INTEGER,
            matching_number INTEGER,
            largest_eigenvalue REAL,
            num_components INTEGER,
            is_eularian INTEGER NOT NULL,
            is_semi_eularian INTEGER NOT NULL,
            g_is_planar INTEGER NOT NULL,
            twin_free INTEGER NOT NULL
        );

    """)

    conn.commit() # 22H03
    conn.close()

def parse_graph_filename(filename):
    """
    Extrait vertex_cover, nb_nodes, nb_aretes, instance_number depuis un nom de fichier .g6
    Format attendu : ba-{vertex_cover}-n{node}-m{aretes}-{instance}.g6
    """
    pattern = r"ba-(\d+)-n(\d+)-m(\d+)-(\d+)\.g6"
    match = re.match(pattern, filename)
    if match:
        vertex_cover = int(match.group(1))
        nb_nodes = int(match.group(2))
        nb_aretes = int(match.group(3))
        instance_number = int(match.group(4))
        return vertex_cover, nb_nodes, nb_aretes, instance_number
    return None, None, None, None

def load_g6_graph(file_path):
    graph = nx.read_graph6(file_path)
    nb_sommets = graph.number_of_nodes()
    return graph, nb_sommets


def find_g6_files(root_dir):
    g6_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".g6"):
                g6_files.append(os.path.join(dirpath, file))

    return g6_files


def extract_graph_class(file_path, root_dir):
    relative_path = os.path.relpath(file_path, root_dir)
    class_name = relative_path.split(os.sep)[0]
    return class_name


def insert_graph(db_name, file_path, root_dir):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    start_time = time.time()
    graph, nb_sommets = load_g6_graph(file_path)
    graph_name_with_extension = os.path.basename(file_path)

    graph_name = os.path.splitext(graph_name_with_extension)[0]
    canonical_form = nx.to_graph6_bytes(graph, header=False).decode('ascii').strip()
    graph_class = extract_graph_class(file_path, root_dir)
    nb_aretes = graph.number_of_edges()
    densite = (2 * nb_aretes) / (nb_sommets * (nb_sommets - 1)) if nb_sommets > 1 else 0

    cover_size, _, _, instance_number = parse_graph_filename(graph_name_with_extension)

    max_degree = max(deg for node, deg in graph.degree)
    min_degree = min(deg for node, deg in graph.degree)
    avg_degree = sum(deg for node, deg in graph.degree) / nb_sommets if nb_sommets > 0 else 0
    num_deg_one = sum(1 for node, deg in graph.degree if deg == 1) # à surveiller, prends un tout petit peu de temps
    count_triangles = sum(nx.triangles(graph).values()) // 3  # Calcul le nombre de triangles fermés. 17 secondes
    total_triangles = sum(1 for node, deg in graph.degree if deg >= 2) # Calcul le nombre de triangles totaux (fermé ou non).  à surveiller, retourne 0sec
    fraction_closed_triangles = count_triangles / total_triangles if total_triangles > 0 else 0 # à surveiller, retourne 0sec
    avg_count_triangles = count_triangles / nb_aretes if nb_aretes > 0 else 0 # à surveiller, retourne 0sec
    max_count_triangles = max(nx.triangles(graph).values()) # 17 sec
    min_count_triangles = min(nx.triangles(graph).values()) # 17 sec
    avg_neighbor_deg = sum(nx.average_neighbor_degree(graph).values()) / nb_sommets if nb_sommets > 0 else 0
    sum_neighbor_deg = sum(nx.average_degree_connectivity(graph).values())
    degree_assortativity_coefficient = nx.degree_assortativity_coefficient(graph) # rapide
    diameter = nx.diameter(graph) # rapide
    local_efficiency = nx.local_efficiency(graph) # 38min
    avg_neighbor_clustering = nx.average_clustering(graph) # 1min
    avg_clustering_coefficient = nx.average_clustering(graph) # 1 min

    # Nouvelles propriétés
    is_acyclic = nx.is_forest(graph) # un graphe acyclique non orienté est un arbre
    is_bipartite = nx.is_bipartite(graph)
    is_connected = nx.is_connected(graph)
    radius = nx.radius(graph)

    matching = nx.max_weight_matching(graph, maxcardinality=True, weight=None) # Trouver l'appariement de taille maximale
    matching_number = len(matching) # Nombre d'arêtes dans le matching maximal - 5 secondes

    # laplacian_largest_eigenvalue
    L = nx.laplacian_matrix(graph).todense() # Calcul de la matrice du laplacien
    eigenvalues = np.linalg.eigvals(L) # Calcul des valeurs propres du laplacien
    largest_eigenvalue = np.max(eigenvalues) # Plus grande valeur propre

    # algebraic_connectivity
    num_components = nx.number_connected_components(graph) # calcule le nombre de composants connexes
    is_eularian = nx.is_eulerian(graph)
    is_semi_eularian = nx.is_semieulerian(graph)
    g_is_planar = is_planar(graph)
    # tw = nx.algorithms.approximation.treewidth_min_degree(graph)
    twin_free = is_twin_free(graph)  # 40 sec

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Temps d execution : {execution_time}"+"secondes")

    print(f"Nom du graphe : {graph_name}")
    print(f"Propriétés : num_nodes = {nb_sommets}, cover_size = {cover_size}, instance_number = {instance_number}")


    # propriétés problèmatiques :
    # # avg_node_connectivity = nx.average_node_connectivity( graph) / nb_sommets if nb_sommets > 0 else 0  # testé sur 1h et continue de tourner.
    #
    # cliques = list(nx.find_cliques(graph))  # problème np-difficile basé sur l'énumération donc impraticable
    # nb_cliques = len(cliques) # Nombre de cliques maximales

    largest_eigenvalue = np.max(np.linalg.eigvals(L))  # Directement récupérer la plus grande valeur propre

    cursor.execute("""
        INSERT INTO graphes (
            graph_name,
            canonical_form,
            class,
            nb_sommets,
            nb_aretes,
            densite,
            cover_size,
            instance_number,
            max_degree,
            min_degree,
            avg_degree,
            num_deg_one,
            count_triangles,
            total_triangles,
            fraction_closed_triangles,
            avg_count_triangles,
            max_count_triangles,
            min_count_triangles,
            avg_neighbor_deg,
            sum_neighbor_deg,
            degree_assortativity_coefficient,
            diameter,
            local_efficiency,
            avg_neighbor_clustering,
            avg_clustering_coefficient,
            is_acyclic,
            is_bipartite,
            is_connected,
            radius,
            matching_number,
            largest_eigenvalue,
            num_components,
            is_eularian,
            is_semi_eularian,
            g_is_planar,
            twin_free
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        graph_name,
        canonical_form,
        graph_class,
        nb_sommets,
        nb_aretes,
        densite,
        cover_size,
        instance_number,
        max_degree,
        min_degree,
        avg_degree,
        num_deg_one,
        count_triangles,
        total_triangles,
        fraction_closed_triangles,
        avg_count_triangles,
        max_count_triangles,
        min_count_triangles,
        avg_neighbor_deg,
        sum_neighbor_deg,
        degree_assortativity_coefficient,
        diameter,
        local_efficiency,
        avg_neighbor_clustering,
        avg_clustering_coefficient,
        is_acyclic,
        is_bipartite,
        is_connected,
        radius,
        matching_number,
        largest_eigenvalue,
        num_components,
        is_eularian,
        is_semi_eularian,
        g_is_planar,
        twin_free
    ))

    conn.commit()
    conn.close()

def is_planar(G):
    is_planar, _ = nx.check_planarity(G)
    return is_planar

def is_twin_free(G):
    for u in G.nodes():
        for v in G.nodes():
            if u != v and set(G.neighbors(u)) == set(G.neighbors(v)):
                return False  # une paire de sommets jumeaux a été trouvée
    return True  # aucune paire de sommets jumeaux

if __name__ == "__main__":
    root_directory = "graph6_files/barabasi_albert"
    db_name = "graphes.db"

    create_database(db_name)
    g6_files = find_g6_files(root_directory)

    for file in g6_files:
        print(f"Ajout du graphe : {file}")
        insert_graph(db_name, file, root_directory)

