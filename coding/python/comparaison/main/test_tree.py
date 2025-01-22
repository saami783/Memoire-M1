#!/usr/bin/env python3
import random


def generate_random_tree(n, seed=None):
    """
    Génère un arbre aléatoire à n sommets (numérotés de 1 à n)
    en utilisant un mécanisme d'attache préférentielle.

    Paramètres:
    -----------
    - n : nombre de sommets
    - seed : graine de génération pour la reproductibilité (optionnelle)

    Retourne:
    ---------
    - edges : liste de tuples (u, v) représentant les arêtes de l'arbre.
    """
    if seed is not None:
        random.seed(seed)

    # Initialisation : pas d'arêtes, degré de chaque sommet = 0
    edges = []
    degrees = [0] * n  # degrees[i] : degré du sommet i+1 (indexé à 0 dans la liste)

    # On construit l'arbre en connectant chaque nouveau sommet i
    # (pour i=2..n) à un parent parmi [1..i-1].
    # Le parent est choisi avec une probabilité proportionnelle à (degrees[parent] + 1).
    for i in range(1, n):
        # Sommet à relier : i (0-based), donc dans le graphe c'est i+1
        # On calcule la somme des (deg + 1) pour les sommets [0..i-1]
        total = sum((deg + 1) for deg in degrees[:i])
        # On tire un entier aléatoire dans [1..total]
        r = random.randint(1, total)

        # On parcourt les sommets déjà existants pour trouver qui sera le parent
        s = 0
        for parent in range(i):
            s += (degrees[parent] + 1)
            if s >= r:
                # Ajout de l'arête (parent+1, i+1) en notation 1-based
                edges.append((parent + 1, i + 1))
                # Mise à jour des degrés
                degrees[parent] += 1
                degrees[i] += 1
                break

    return edges


def write_dimacs_file(edges, n, filename):
    """
    Écrit la liste d'arêtes edges d'un graphe à n sommets
    dans un fichier au format DIMACS.
    """
    with open(filename, 'w') as f:
        # Ligne de commentaire
        f.write(f"c Random tree with {n} vertices\n")
        # Ligne 'p edge <nombre_sommets> <nombre_aretes>'
        f.write(f"p edge {n} {len(edges)}\n")

        # Chaque arête est notée "e u v" (u < v ou non, peu importe en DIMACS non orienté)
        for (u, v) in edges:
            f.write(f"e {u} {v}\n")


def main():
    # Trois tailles de graphes
    sizes = [200, 500, 750]

    # Pour chaque taille, on crée 5 instances
    for n in sizes:
        for i in range(1, 6):
            # Pour rendre la génération reproductible, on fixe une seed simple
            seed = 1000 * n + i
            edges = generate_random_tree(n, seed=seed)

            # Nom de fichier par convention
            filename = f"tree_n{n}_instance{i}.dimacs"

            # Écriture du graphe au format DIMACS
            write_dimacs_file(edges, n, filename)
            print(f"Fichier généré : {filename}")


if __name__ == "__main__":
    main()
