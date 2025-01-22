import random

def sorted_list_left(graph):
    C = set()
    degrees = dict(graph.degree())
    # Mélanger les nœuds avant le tri pour éviter un ordre déterministe des nœuds de même degré
    node_list = list(degrees.keys())
    random.shuffle(node_list)
    sorted_nodes = sorted(node_list, key=lambda x: -degrees[x])  # Trie décroissant par degré

    for u in sorted_nodes:
        # Ici, pas de sélection entre plusieurs nœuds d'égale priorité, on suit simplement l'ordre
        if any(neighbor not in C for neighbor in graph.neighbors(u)):
            C.add(u)

    return list(C)