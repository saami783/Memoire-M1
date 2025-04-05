import os
import networkx as nx
import tqdm

def load_dimacs_graph(filename):
    """Charge un graphe au format DIMACS et le convertit en objet NetworkX."""
    G = nx.Graph()
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if parts[0] == "p":  # Ligne d'entête (nombre de sommets et arêtes)
                num_nodes = int(parts[2])
                num_edges = int(parts[3])
            elif parts[0] == "e":  # Définition des arêtes
                u, v = int(parts[1]), int(parts[2])
                G.add_edge(u, v)
    return G

def process_dimacs_files(input_folder, output_folder):
    """Parcourt récursivement le dossier input_folder, convertit les fichiers .dimacs en .g6
       et les stocke dans un dossier miroir sous output_folder.
    """
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".dimacs"):
                input_path = os.path.join(root, filename)

                # Créer le chemin miroir dans output_folder
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Définir le chemin de sortie en remplaçant l'extension
                output_path = os.path.join(output_dir, filename.replace(".dimacs", ".g6"))

                # Charger et convertir le graphe
                graph = load_dimacs_graph(input_path)
                graph = nx.relabel_nodes(graph, lambda x: x - 1)
                nx.write_graph6(graph, output_path, header=False)

                print(f"Converti : {input_path} -> {output_path}")

if __name__ == "__main__":
    input_folder = "dimacs_files"
    output_folder = "graph6_files"

    process_dimacs_files(input_folder, output_folder)
    print("Conversion terminée !")
