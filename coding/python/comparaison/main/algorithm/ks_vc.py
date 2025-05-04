# https://www.sciencedirect.com/science/article/pii/S0304397511000363
import networkx as nx
import random

def ks_vc(graph):
    G = graph.copy()
    vertex_cover = []
    
    while G.number_of_edges() > 0:
        leaves = [v for v in G.nodes() if G.degree(v) == 1]
        if leaves:
            leaf = random.choice(leaves)
            neighbor = next(G.neighbors(leaf))
            vertex_cover.append(neighbor)
            G.remove_nodes_from([leaf, neighbor])
        else:
            node = random.choice(list(G.nodes()))
            vertex_cover.append(node)
            G.remove_node(node)
    
    return vertex_cover