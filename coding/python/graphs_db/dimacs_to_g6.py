import networkx as nx
from networkx.readwrite.graph6 import write_graph6


def load_graph_from_dimacs(filename):
    """
    Charge un graphe depuis un fichier DIMACS avec correction de l'indexation.
    """
    graph = nx.Graph()

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("c") or line == "":
                continue
            if line.startswith("p"):
                parts = line.split()
                num_nodes = int(parts[2])
                graph.add_nodes_from(range(num_nodes))
            elif line.startswith("e"):
                parts = line.split()
                u, v = int(parts[1]) - 1, int(parts[2]) - 1
                graph.add_edge(u, v)

    return graph


def save_graph_as_g6(graph, output_file):
    """
    Sauvegarde un graphe en format Graph6 et affiche la chaîne générée.
    """
    g6_string = nx.to_graph6_bytes(graph, header=False).decode()
    print("Graph6 généré :", g6_string)

    with open(output_file, "w") as f:
        f.write(g6_string)


if __name__ == "__main__":
    filename = "dimacs_files/regular/regular_28_16_1.dimacs"
    output_file = "graph.g6"

    graph = load_graph_from_dimacs(filename)

    print(f"Nombre de sommets : {graph.number_of_nodes()}")
    print(f"Nombre d'arêtes : {graph.number_of_edges()}")

    write_graph6(graph, "graph.g6")

    save_graph_as_g6(graph, output_file)

    print(f"Conversion réussie : {filename} → {output_file}")
