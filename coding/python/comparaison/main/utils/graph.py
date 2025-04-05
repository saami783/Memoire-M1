import sqlite3
import networkx as nx

def load_graph_from_db(canonical_form):
    """
    Charge un graphe depuis la base de données SQLite en utilisant sa forme canonique (graph6).
    """
    graph = nx.from_graph6_bytes(canonical_form.encode())
    return graph

def get_graphs_from_db(graph_names):
    """
    Récupère tous les graphes correspondant à plusieurs noms de classes de graphes
    avec leurs propriétés depuis la base de données en une seule requête.
    """
    connection = sqlite3.connect("graphes.db")
    cursor = connection.cursor()

    placeholders = ",".join("?" for _ in graph_names)
    query = f"""
        SELECT graph_name, canonical_form, cover_size, instance_number, nb_sommets, nb_aretes
        FROM graphes 
        WHERE graph_name IN ({placeholders})
    """
    cursor.execute(query, graph_names)
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