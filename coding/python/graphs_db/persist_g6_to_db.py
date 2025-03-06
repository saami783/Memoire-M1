import os
import sqlite3
import networkx as nx


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
            num_nodes INTEGER,
            cover_size INTEGER,
            instance_number INTEGER
        )
    """)

    conn.commit()
    conn.close()


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

    graph, nb_sommets = load_g6_graph(file_path)
    count_edges = graph.number_of_edges()
    densite = (2 * count_edges) / (nb_sommets * (nb_sommets - 1)) if nb_sommets > 1 else 0
    canonical_form = nx.to_graph6_bytes(graph, header=False).decode('ascii').strip()

    # max_degree = max(deg for node, deg in graph.degree)
    # min_degree = min(deg for node, deg in graph.degree)
    # avg_degree = sum(deg for node, deg in graph.degree) / nb_sommets if nb_sommets > 0 else 0
    # num_deg_one = sum(1 for node, deg in graph.degree if deg == 1)
    # count_triangles = sum(nx.triangles(graph).values()) // 3  # Division entière. Calcul le nombre de triangles fermés
    # total_triangles = sum(1 for node, deg in graph.degree if deg >= 2) # Calcul le nombre de triangles totaux (fermé ou non)
    # fraction_closed_triangles = count_triangles / total_triangles if total_triangles > 0 else 0
    # avg_count_triangles = count_triangles / count_edges if count_edges > 0 else 0
    # max_count_triangles = max(nx.triangles(graph).values())
    # min_count_triangles = min(nx.triangles(graph).values())
    # avg_neighbor_deg = sum(nx.average_neighbor_degree(graph).values()) / nb_sommets if nb_sommets > 0 else 0
    # sum_neighbor_deg = sum(nx.average_degree_connectivity(graph).values())
    # avg_node_connectivity = sum(nx.average_node_connectivity(graph).values()) / nb_sommets if nb_sommets > 0 else 0
    # degree_assortativity_coefficient = nx.degree_assortativity_coefficient(graph)
    # degrees = dict(graph.degree)
    # betweenness_centrality = nx.betweenness_centrality(graph)
    # closeness_centrality = nx.closeness_centrality(graph)
    # eigenvector_centrality = nx.eigenvector_centrality(graph)
    # pagerank = nx.pagerank(graph)
    # core_number = nx.core_number(graph)
    # harmonic_centrality = nx.harmonic_centrality(graph)
    # load_centrality = nx.load_centrality(graph)
    # eccentricity = nx.eccentricity(graph)
    # diameter = nx.diameter(graph)
    # local_efficiency = nx.local_efficiency(graph)
    # avg_neighbor_clustering = nx.average_clustering(graph)
    # avg_clustering_coefficient = nx.average_clustering(graph)

    graph_name_with_extension = os.path.basename(file_path)  # Le nom complet du fichier avec l'extension
    graph_name = os.path.splitext(graph_name_with_extension)[0]  # Retirer l'extension ".g6"

    if "bhoslib" not in file_path:
        parts = graph_name.split('_')
        if len(parts) >= 3:
            try:
                num_nodes = int(parts[-3])  # Avant-dernière partie pour le nombre de noeuds
                cover_size = int(parts[-2])  # Avant-dernière partie pour la taille du couvert
                instance_number = int(parts[-1])  # Dernière partie pour le numéro de l'instance

                graph_name = '_'.join(parts[:-3])  # Conserver uniquement la première partie du nom, avant les numéros
            except ValueError:
                num_nodes = cover_size = instance_number = None
                graph_name = graph_name
        else:
            num_nodes = cover_size = instance_number = None
            graph_name = graph_name
    else:
        num_nodes = cover_size = instance_number = None

    graph_class = extract_graph_class(file_path, root_dir)

    print(f"Nom du graphe : {graph_name}")
    print(f"Propriétés : num_nodes = {num_nodes}, cover_size = {cover_size}, instance_number = {instance_number}")

    cursor.execute("""
        INSERT INTO graphes (graph_name, canonical_form, class, nb_sommets, nb_aretes, densite, num_nodes, cover_size, instance_number) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (graph_name, canonical_form, graph_class, nb_sommets, count_edges, densite, num_nodes, cover_size, instance_number))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    root_directory = "graph6_files"
    db_name = "graphes.db"

    create_database(db_name)
    g6_files = find_g6_files(root_directory)

    for file in g6_files:
        print(f"Ajout du graphe : {file}")
        insert_graph(db_name, file, root_directory)
