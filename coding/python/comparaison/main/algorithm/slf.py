import random

def sorted_list_left(graph):
    C = set()
    degrees = dict(graph.degree())
    nodes = list(graph.nodes())
    # random.shuffle(nodes)

    sorted_nodes = sorted(nodes, key=lambda x: -degrees[x])

    for i, u in enumerate(sorted_nodes):
        right_neighbors = sorted_nodes[i+1:]
        has_uncovered_right_neighbor = any(
            (neighbor in right_neighbors) and (neighbor not in C) 
            for neighbor in graph.neighbors(u)
        )
        if has_uncovered_right_neighbor:
            C.add(u)

    return list(C)