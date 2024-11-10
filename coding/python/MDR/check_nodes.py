import networkx as nx

input_file = 'benchmarks/frb59-26/cleaned/frb59-26-1_cleaned.dimacs'

graph = nx.read_edgelist(input_file, nodetype=int, data=False)

print(graph)