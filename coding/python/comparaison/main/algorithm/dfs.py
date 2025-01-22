import networkx as nx

def dfs_heuristic(graph):
    dfs_tree = nx.dfs_tree(graph)
    internal_nodes = [node for node in dfs_tree.nodes if dfs_tree.degree[node] > 1]
    return internal_nodes