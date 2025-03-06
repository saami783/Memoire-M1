import networkx as nx

def load_graph_from_dimacs(filename):
    """
    @todo charger les graph6
    Charge un graphe depuis un fichier DIMACS.
    """
    graph = nx.Graph()
    node_mapping = {}  # Pour mapper les noms non numérotés à des entiers
    next_id = 1

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("c") or line == "":
                continue
            if line.startswith("p"):
                parts = line.split()
                num_nodes = int(parts[2])  # Nombre de sommets
                graph.add_nodes_from(range(1, num_nodes + 1))  # Ajout des sommets
            elif line.startswith("e"):
                parts = line.split()
                u, v = parts[1], parts[2]
                # Mapper les sommets non numérotés en entiers
                if u not in node_mapping:
                    node_mapping[u] = next_id
                    next_id += 1
                if v not in node_mapping:
                    node_mapping[v] = next_id
                    next_id += 1
                graph.add_edge(node_mapping[u], node_mapping[v])
    return graph