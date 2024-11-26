import networkx as nx

import os
# Nettoie tous les graphes situé dans le dossier benchmarks en supprimant les en-têtes
base_dir = 'out/benchmarks/'
subdirs = ['frb30-15', 'frb40-19', 'frb50-23', 'frb59-26']

for subdir in subdirs:
    input_path = os.path.join(base_dir, subdir)
    cleaned_path = os.path.join(input_path, 'cleaned')


    os.makedirs(cleaned_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if filename.endswith('.dimacs') and not filename.endswith('_cleaned.dimacs'):
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(cleaned_path, filename.replace('.dimacs', '_cleaned.dimacs'))

            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                for line in infile:
                    if line.startswith('e'):
                        outfile.write(line[2:])
                    elif not line.startswith('p'):
                        outfile.write(line)

print("Tous les fichiers nettoyés ont été créés dans le dossier 'cleaned' de chaque répertoire.")
