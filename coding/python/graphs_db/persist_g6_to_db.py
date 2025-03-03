import os
import sqlite3
import networkx as nx


def create_database(db_name="graphes.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graphes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            graph_name TEXT UNIQUE,
            canonical_form TEXT,
            class TEXT,
            nb_sommets INTEGER,
            nb_aretes INTEGER,
            densite REAL
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
    nb_aretes = graph.number_of_edges()
    densite = (2 * nb_aretes) / (nb_sommets * (nb_sommets - 1)) if nb_sommets > 1 else 0
    canonical_form = nx.to_graph6_bytes(graph, header=False).decode('ascii').strip()

    graph_class = extract_graph_class(file_path, root_dir)

    cursor.execute("""
        INSERT INTO graphes (graph_name, canonical_form, class, nb_sommets, nb_aretes, densite) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (file_path, canonical_form, graph_class, nb_sommets, nb_aretes, densite))

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
