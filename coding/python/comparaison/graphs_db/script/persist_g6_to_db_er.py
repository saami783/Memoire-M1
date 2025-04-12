import os
import sqlite3
import networkx as nx
from pprint import pprint
import time
import numpy as np
import re
from typing import Optional, Tuple, Union

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
            is_eularian INTEGER,
            is_semi_eularian INTEGER,
            g_is_planar INTEGER,
            twin_free INTEGER
        );

    """)

    conn.commit() # 22H03
    conn.close()


def parse_erdos_renyi_filename(filename: str) -> Optional[Tuple[int, int, Union[int, float], int]]:
    """
    Parse un nom de fichier de graphe de type:
    - ba-{cover_size}-{n}-{m}-{instance}.g6
    - erdos_renyi-{cover_size}-{n}-{p}-{instance}.g6

    Retourne: (cover_size, n, m or p, instance)
    """
    # Sans l'extension
    name = filename.replace(".g6", "")

    # Match BA
    ba_match = re.match(r"ba-(\d+)-(\d+)-(\d+)-(\d+)", name)
    if ba_match:
        return tuple(map(int, ba_match.groups()))

    # Match Erdős–Rényi
    er_match = re.match(r"erdos_renyi-(\d+)-(\d+)-([\d.]+)-(\d+)", name)
    if er_match:
        cover_size, n, p_str, instance = er_match.groups()
        return int(cover_size), int(n), float(p_str), int(instance)

    return None

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
    graph_name = "erdos_renyi"

    canonical_form = nx.to_graph6_bytes(graph, header=False).decode('ascii').strip()
    graph_class = extract_graph_class(file_path, root_dir)
    nb_aretes = graph.number_of_edges()
    densite = (2 * nb_aretes) / (nb_sommets * (nb_sommets - 1)) if nb_sommets > 1 else 0

    cover_size, _, _, instance_number = parse_erdos_renyi_filename(graph_name_with_extension)

    # Degrés
    max_degree = max(deg for node, deg in graph.degree)
    min_degree = min(deg for node, deg in graph.degree)
    avg_degree = sum(deg for node, deg in graph.degree) / nb_sommets if nb_sommets > 0 else 0
    num_deg_one = sum(1 for node, deg in graph.degree if deg == 1)

    # Triangles
    triangle_counts = nx.triangles(graph)
    count_triangles = sum(triangle_counts.values()) // 3
    total_triangles = sum(1 for node, deg in graph.degree if deg >= 2)
    fraction_closed_triangles = count_triangles / total_triangles if total_triangles > 0 else 0
    avg_count_triangles = count_triangles / nb_aretes if nb_aretes > 0 else 0
    max_count_triangles = max(triangle_counts.values())
    min_count_triangles = min(triangle_counts.values())

    # Clustering & degré voisin
    avg_neighbor_deg = sum(nx.average_neighbor_degree(graph).values()) / nb_sommets if nb_sommets > 0 else 0
    sum_neighbor_deg = sum(nx.average_degree_connectivity(graph).values())
    degree_assortativity_coefficient = nx.degree_assortativity_coefficient(graph)

    # Propriétés générales
    is_acyclic = nx.is_forest(graph)
    is_bipartite = nx.is_bipartite(graph)
    is_connected = nx.is_connected(graph)

    # Valeurs par défaut si le graphe n'est pas connexe
    diameter = None
    radius = None
    local_efficiency = None
    avg_neighbor_clustering = None
    avg_clustering_coefficient = None
    num_components = None
    is_eularian = 0
    is_semi_eularian = 0
    g_is_planar = 0

    if is_connected:
        diameter = nx.diameter(graph)
        radius = nx.radius(graph)
        local_efficiency = nx.local_efficiency(graph)
        avg_neighbor_clustering = nx.average_clustering(graph)
        avg_clustering_coefficient = nx.average_clustering(graph)
        num_components = nx.number_connected_components(graph)
        is_eularian = nx.is_eulerian(graph)
        is_semi_eularian = nx.is_semieulerian(graph)
        g_is_planar = is_planar(graph)

    matching = nx.max_weight_matching(graph, maxcardinality=True, weight=None)
    matching_number = len(matching)

    L = nx.laplacian_matrix(graph).todense()
    eigenvalues = np.linalg.eigvals(L)
    largest_eigenvalue = np.max(eigenvalues)

    # Connectivité algébrique (valeur propre > 0 sauf pour connexité)
    algebraic_connectivity = None
    if is_connected:
        eigenvalues_sorted = sorted(np.real(eigenvalues))
        algebraic_connectivity = eigenvalues_sorted[1]  # La deuxième plus petite valeur propre

    # Propriétés spécifiques
    twin_free = is_twin_free(graph)

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
    root_directory = "g6_files/erdos_renyi"
    db_name = "graphes.db"

    create_database(db_name)
    g6_files = find_g6_files(root_directory)

    for file in g6_files:
        print(f"Ajout du graphe : {file}")
        insert_graph(db_name, file, root_directory)

