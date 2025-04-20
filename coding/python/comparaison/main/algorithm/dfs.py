import random
import networkx as nx

def dfs(G) -> list:
    cover = set()
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        root = random.choice(list(subgraph.nodes()))
        dfs_tree = nx.dfs_tree(subgraph, source=root)
        for node in dfs_tree.nodes():
            if dfs_tree.out_degree(node) > 0:
                cover.add(node)
    return list(cover)


