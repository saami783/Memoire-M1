import os

def get_dimacs_files(input_dir):
    """Retourne la liste des fichiers DIMACS triés par taille de graphe."""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Le dossier '{input_dir}' n'existe pas.")

    dimacs_files = [f for f in os.listdir(input_dir) if f.endswith(".dimacs")]
    if not dimacs_files:
        raise ValueError(f"Aucun fichier DIMACS trouvé dans '{input_dir}'.")

    # Trier les fichiers par le nombre de sommets dans le nom du fichier
    try:
        return sorted(dimacs_files, key=lambda x: int(x.split("-")[1]))
    except (IndexError, ValueError):
        raise ValueError("Erreur de format dans le nom des fichiers DIMACS.")
