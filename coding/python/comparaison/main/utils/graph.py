import sqlite3
import networkx as nx

def load_graph_from_db(canonical_form):
    """
    Charge un graphe depuis la base de données SQLite en utilisant sa forme canonique (graph6).
    """
    graph = nx.from_graph6_bytes(canonical_form.encode())
    return graph


def get_graphs_from_db(graph_names=None):
    """
    Récupère les graphes dont le nom n'est pas dans la liste d'exclusion
    (graphes impraticables) et, si spécifié, dont le nom figure dans 'graph_names'.

    Parameters:
        graph_names (list ou None): Liste optionnelle de noms de graphes à inclure.

    Returns:
        list: Les graphes récupérés avec leurs propriétés.
    """
    exclusion_list = [
        # "frb40-19-1","frb40-19-2", "frb40-19-3",
        # "frb50-23-1", "frb50-23-2",
        # "frb50-23-3", "frb50-23-4", "frb50-23-5", "frb53-24-1", "frb53-24-2",
        # "frb53-24-3", "frb53-24-4", "frb53-24-5", "frb59-26-1", "frb59-26-2",
        # "frb59-26-3", "frb59-26-5", "frb100-40", "frb35-17-1", "frb35-17-2",
        # "frb35-17-3", "frb35-17-4", "frb35-17-5", "frb40-19-4", "frb40-19-5",
        # "frb45-21-1", "frb45-21-2", "frb45-21-3", "frb45-21-4", "frb45-21-5",
        # "frb59-26-4",
        #  "regular", "tree", "barabasi_albert", "erdos_renyi"
    ]

    # connection = sqlite3.connect("ba.db")
    connection = sqlite3.connect("bhoslib.db")
    #connection = sqlite3.connect("er.db")
    # connection = sqlite3.connect("ruglar.db")
    # connection = sqlite3.connect("tree.db")
    cursor = connection.cursor()

    exclusion_placeholders = ",".join("?" for _ in exclusion_list)
    query = f"""
        SELECT graph_name, canonical_form, cover_size, instance_number, nb_sommets, nb_aretes
        FROM graphes 
        WHERE graph_name NOT IN ({exclusion_placeholders})
    """
    params = exclusion_list.copy()

    if graph_names:
        inclusion_placeholders = ",".join("?" for _ in graph_names)
        query += f" AND graph_name IN ({inclusion_placeholders})"
        params.extend(graph_names)

    cursor.execute(query, params)
    graphs = cursor.fetchall()
    connection.close()

    return graphs

def process_graph(graph_name, canonical_form, cover_size, instance_number, num_nodes, num_edges):
    """
    Traite un graphe en utilisant les propriétés récupérées de la base de données.
    """
    return {
        "graph_name": graph_name,
        "cover_size": cover_size,
        "instance_number": instance_number,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }