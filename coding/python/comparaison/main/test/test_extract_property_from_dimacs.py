import networkx as nx


def load_dimacs_graph(file_path):
    """
    Charge un graphe à partir d'un fichier DIMACS.
    """
    G = nx.Graph()

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('p'):  # Ligne définissant le nombre de sommets et d'arêtes
                _, _, num_vertices, num_edges = line.split()
                num_vertices, num_edges = int(num_vertices), int(num_edges)
            elif line.startswith('e'):  # Ligne définissant une arête
                _, u, v = line.split()
                G.add_edge(int(u), int(v))

    return G, num_vertices, num_edges


def extract_graph_properties(G):
    """
    Extrait les propriétés du graphe.
    """
    degrees = dict(G.degree())

    max_degree = max(degrees.values())
    min_degree = min(degrees.values())

    max_degree_nodes = [node for node, degree in degrees.items() if degree == max_degree]
    min_degree_nodes = [node for node, degree in degrees.items() if degree == min_degree]

    properties = {
        "Nombre de sommets": G.number_of_nodes(),
        "Nombre d'arêtes": G.number_of_edges(),
        "Degré maximal": max_degree,
        "Sommets de degré maximal": max_degree_nodes,
        "Degré minimal": min_degree,
        "Sommets de degré minimal": min_degree_nodes,
        "Degré moyen": sum(degrees.values()) / len(degrees) if degrees else 0
    }

    return properties


def main(file_path):
    """
    Fonction principale pour charger le graphe et afficher ses propriétés.
    """
    G, num_vertices, num_edges = load_dimacs_graph(file_path)
    properties = extract_graph_properties(G)

    print("\n--- Propriétés du graphe ---")
    for key, value in properties.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    file_path = "main/test/test_graph_tree.dimacs"  # Remplace par ton chemin
    main(file_path)