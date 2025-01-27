import os

def get_dimacs_files(input_dir):
    """Retourne la liste des fichiers DIMACS triés par le nombre de sommets
    (supposé être le 2e segment dans le nom, ex: 'tree_50_23_1.dimacs')."""

    # Vérifier que le répertoire existe
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Le dossier '{input_dir}' n'existe pas.")

    # Récupérer uniquement les fichiers .dimacs
    dimacs_files = [f for f in os.listdir(input_dir) if f.endswith(".dimacs")]
    if not dimacs_files:
        raise ValueError(f"Aucun fichier DIMACS trouvé dans '{input_dir}'.")

    # Fonction utilitaire pour extraire le nombre de sommets
    def extract_num_nodes(filename):
        try:
            # Retirer l'extension .dimacs
            base_name = filename.replace(".dimacs", "")
            # Séparer par underscore
            parts = base_name.split("_")
            # Le 2e segment (index=1) doit correspondre au nombre de sommets
            return int(parts[1])
        except (IndexError, ValueError):
            # Si le fichier n'a pas le format attendu, on lève l'exception
            raise ValueError("Erreur de format dans le nom des fichiers DIMACS.")

    # Retourner la liste triée par le nombre de sommets (2e segment)
    return sorted(dimacs_files, key=extract_num_nodes)
