import networkx as nx

# retourne le nombre totals de noeuds d'un benchmark pour vérifier si le graphe est bien chargé avec le benchmark initial
input_file = '../benchmarks/bhoslib/frb59-26/cleaned/frb59-26-1_cleaned.dimacs'

graph = nx.read_edgelist(input_file, nodetype=int, data=False)

print(graph)