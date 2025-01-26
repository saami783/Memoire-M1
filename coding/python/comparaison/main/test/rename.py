import os

def rename_dimacs_files(directory):
    """
    Renomme tous les fichiers DIMACS dans le répertoire donné en remplaçant les underscores `_`
    par des tirets `-` pour correspondre au format souhaité.

    Args:
        directory (str): Chemin vers le répertoire contenant les fichiers DIMACS.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".dimacs") and "_" in filename:
            # Extraire les parties du fichier
            parts = filename.replace(".dimacs", "").split("_")
            if len(parts) == 3 and parts[0] == "anti-mdg":
                # Construire le nouveau nom de fichier
                new_filename = f"anti_mdg-{parts[1]}-{parts[2]}.dimacs"
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)

                # Renommer le fichier
                os.rename(old_path, new_path)
                print(f"Renommé : {filename} -> {new_filename}")

if __name__ == "__main__":
    directory = "dimacs_files/trees"  # Répertoire contenant les fichiers DIMACS
    if os.path.exists(directory):
        rename_dimacs_files(directory)
    else:
        print(f"Erreur : Le répertoire '{directory}' n'existe pas.")
