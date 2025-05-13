import sqlite3
import networkx as nx

def load_graph_from_db(canonical_form):
    # print(f"[DEBUG] Type de canonical_form: {type(canonical_form)}")
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
        "frb40-19-1","frb40-19-2", "frb40-19-3", "frb40-19-4", "frb40-19-5",
        "frb50-23-1", "frb50-23-2", "frb50-23-3", "frb50-23-4", "frb50-23-5",
        "frb53-24-1", "frb53-24-2", "frb53-24-3", "frb53-24-4", "frb53-24-5",
        "frb59-26-1", "frb59-26-2", "frb59-26-3", "frb59-26-5", "frb100-40",
        # "frb30-15-1", "frb30-15-2", "frb30-15-3", "frb30-15-4", "frb30-15-5",
        "frb35-17-1", "frb35-17-2", "frb35-17-3", "frb35-17-4", "frb35-17-5",
        "frb45-21-1", "frb45-21-2", "frb45-21-3", "frb45-21-4", "frb45-21-5",
        "frb59-26-4",
        #  "regular", "tree", "barabasi_albert", "erdos_renyi"
    ]

    connection = sqlite3.connect("db/fusion.db")
    # connection = sqlite3.connect("db/ba.db")
    # connection = sqlite3.connect("db/bhoslib.db")
    # connection = sqlite3.connect("db/er.db")
    # connection = sqlite3.connect("db/regular.db")
    # connection = sqlite3.connect("db/tree.db")
    cursor = connection.cursor()

    exclusion_placeholders = ",".join("?" for _ in exclusion_list)
    query = f"""
        SELECT id, graph_name, class AS graph_class, canonical_form, cover_size, instance_number, nb_sommets, nb_aretes
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

def get_graphs_from_hog(db_path="db/graphs.db"):
    """
    Récupère les graphes depuis la base House of Graphs avec uniquement les champs nécessaires.

    Returns:
        list: Les graphes avec les propriétés nécessaires.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    query = """
        SELECT id, canonical_form, graph_name, Number_of_Vertices, Number_of_Edges, Vertex_Cover_Number
        FROM graphes
        WHERE canonical_form IS NOT NULL
          AND Vertex_Cover_Number > 0
          AND Number_of_Vertices BETWEEN 20 AND 300
          AND Connected = 1.0
    """

    cursor.execute(query)
    graphs = cursor.fetchall()
    connection.close()

    return graphs

def process_graph(graph_name, cover_size, instance_number, num_nodes, num_edges):
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