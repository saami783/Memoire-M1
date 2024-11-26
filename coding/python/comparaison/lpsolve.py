import os
import subprocess

# Fonction pour charger un graphe au format DIMACS et générer un fichier LP
def generate_lp_file_from_dimacs(dimacs_path, lp_path):
    """
    Génère un fichier LP à partir d'un graphe DIMACS.
    Args:
        dimacs_path (str): Chemin du fichier DIMACS.
        lp_path (str): Chemin du fichier LP généré.
    """
    with open(dimacs_path, "r") as dimacs_file, open(lp_path, "w") as lp_file:
        # Parse les arêtes du fichier DIMACS
        edges = []
        for line in dimacs_file:
            if line.startswith("e"):
                _, u, v = line.split()
                edges.append((int(u), int(v)))
            elif line.startswith("p"):
                _, _, num_nodes, _ = line.split()
                num_nodes = int(num_nodes)

        # Écrire la fonction objectif (minimiser le nombre de sommets dans la couverture)
        lp_file.write("min: " + " + ".join([f"x{v}" for v in range(1, num_nodes + 1)]) + ";\n")

        # Écrire les contraintes (chaque arête doit être couverte)
        for u, v in edges:
            lp_file.write(f"x{u} + x{v} >= 1;\n")

        # Déclarer les variables comme binaires
        for v in range(1, num_nodes + 1):
            lp_file.write(f"bin x{v};\n")


# Fonction pour résoudre un fichier LP avec LPSolve
def solve_lp_with_lpsolve(lp_path):
    """
    Résout un fichier LP avec LPSolve et retourne la taille de la couverture optimale.
    Args:
        lp_path (str): Chemin du fichier LP.
    Returns:
        int: Taille de la couverture optimale.
    """
    try:
        # Appel à LPSolve en ligne de commande
        result = subprocess.run(["lp_solve", lp_path], stdout=subprocess.PIPE, text=True)
        output = result.stdout

        # Extraire la solution optimale de l'output
        for line in output.splitlines():
            if line.startswith("Value of objective function"):
                return int(float(line.split(":")[1].strip()))
    except FileNotFoundError:
        print("LPSolve n'est pas installé ou n'est pas dans le PATH.")
        raise

    return None


# Fonction principale pour traiter tous les graphes
def process_all_dimacs_files(input_folder, output_csv):
    """
    Résout tous les fichiers DIMACS dans un dossier donné avec LPSolve.
    Args:
        input_folder (str): Dossier contenant les fichiers DIMACS.
        output_csv (str): Fichier CSV pour sauvegarder les résultats.
    """
    results = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".dimacs"):
                dimacs_path = os.path.join(root, file)
                lp_path = dimacs_path.replace(".dimacs", ".lp")

                print(f"Processing {dimacs_path}...")

                # Générer le fichier LP
                generate_lp_file_from_dimacs(dimacs_path, lp_path)

                # Résoudre avec LPSolve
                optimal_size = solve_lp_with_lpsolve(lp_path)
                print(f"Optimal Vertex Cover Size for {file}: {optimal_size}")

                # Ajouter aux résultats
                results.append({"Instance": file, "Optimal Size": optimal_size})

    # Sauvegarder les résultats dans un CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved in {output_csv}")


# Exécution
if __name__ == "__main__":
    # Dossier d'entrée contenant les graphes
    input_folder = "benchmarks"  # Adapter au chemin de vos benchmarks
    output_csv = "optimal_vertex_cover_results.csv"

    # Résoudre les graphes et sauvegarder les résultats
    process_all_dimacs_files(input_folder, output_csv)
