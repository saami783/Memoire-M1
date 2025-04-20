import random

def sorted_list_right(graph):
    C = set()
    degrees = dict(graph.degree())
    node_list = list(degrees.keys())
    random.shuffle(node_list)
    sorted_nodes = sorted(node_list, key=lambda x: -degrees[x])

    position = {node: idx for idx, node in enumerate(sorted_nodes)}

    for u in reversed(sorted_nodes):
        neighbors_right = [v for v in graph.neighbors(u) if position[v] > position[u]]
        if any(v not in C for v in neighbors_right):
            C.add(u)

    return list(C)
