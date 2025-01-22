import random

def sorted_list_right(graph):
    C = set()
    degrees = dict(graph.degree())
    node_list = list(degrees.keys())
    random.shuffle(node_list)
    sorted_nodes = sorted(node_list, key=lambda x: -degrees[x])

    for u in reversed(sorted_nodes):
        # Idem que sorted_list_left, pas de choix aléatoire particulier ici
        if any(neighbor not in C for neighbor in graph.neighbors(u)):
            C.add(u)

    return list(C)