from .ils import ils
import networkx as nx

G = nx.path_graph(10)  # Petit graphe de test
cover = ils(G)
print(cover)  # Devrait marcher sans erreur