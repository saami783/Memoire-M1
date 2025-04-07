import os
import sqlite3
import networkx as nx
from pprint import pprint
import time
import numpy as np
import random
from solveur import minimum_vertex_cover

def generate_and_prepare_graph(n=15, p=0.2, seed=None):
    if seed is not None:
        random.seed(seed)
    g = nx.erdos_renyi_graph(n, p)
    if g.number_of_edges() == 0 or not nx.is_connected(g):
        return generate_and_prepare_graph(n, p, seed+1 if seed is not None else None)
    cover_size = minimum_vertex_cover(g)
    return g, g.number_of_nodes(), cover_size


def create_database(db_name="entrainement.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
       CREATE TABLE IF NOT EXISTS graphes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            graph_name TEXT,
            canonical_form TEXT,
            class TEXT,
            label TEXT,
            nb_sommets INTEGER,
            nb_aretes INTEGER,
            densite REAL,
            cover_size INTEGER,
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


def extract_graph_class(file_path, root_dir):
    relative_path = os.path.relpath(file_path, root_dir)
    class_name = relative_path.split(os.sep)[0]
    return class_name


def insert_graph(db_name, label="test"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    start_time = time.time()
    graph, nb_sommets, cover_size = generate_and_prepare_graph(n=15, p=0.2)

    graph_name = "erdos_renyi"
    canonical_form = nx.to_graph6_bytes(graph, header=False).decode('ascii').strip()

    graph_class = "erdos renyi"
    nb_aretes = graph.number_of_edges()
    densite = (2 * nb_aretes) / (nb_sommets * (nb_sommets - 1)) if nb_sommets > 1 else 0
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
    print(f"Propriétés : num_nodes = {nb_sommets}, cover_size = {cover_size}")


    # propriétés problèmatiques :
    # # avg_node_connectivity = nx.average_node_connectivity( graph) / nb_sommets if nb_sommets > 0 else 0  # testé sur 1h et continue de tourner.
    #
    # cliques = list(nx.find_cliques(graph))  # problème np-difficile basé sur l'énumération donc impraticable
    # nb_cliques = len(cliques) # Nombre de cliques maximales

    cursor.execute("""
        INSERT INTO graphes (
            graph_name,
            canonical_form,
            class,
            label,
            nb_sommets,
            nb_aretes,
            densite,
            cover_size,
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
        label,
        nb_sommets,
        nb_aretes,
        densite,
        cover_size[0],
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
    db_name = "entrainement.db"

    create_database(db_name)

    for i in range(1,1000):
        print(f"Ajout du graphe : {i}")
        insert_graph(db_name, "train")


    for i in range(1,7):
        print(f"Ajout du graphe : {i}")
        insert_graph(db_name, "test")