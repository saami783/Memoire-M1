import networkx as nx

# Nettoie le graphe en supprimant les en-têtes
input_file = '../benchmarks/bhoslib/frb50-23/frb50-23-3.dimacs'
output_file = '../benchmarks/bhoslib/frb50-23/cleaned/frb50-23-3_cleaned.dimacs'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if line.startswith('e'):
            outfile.write(line[2:])
        elif not line.startswith('p'):
            outfile.write(line)

graph = nx.read_edgelist(output_file, nodetype=int, data=False)
print(graph)
