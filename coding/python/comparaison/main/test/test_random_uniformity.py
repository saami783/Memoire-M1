import numpy as np
import matplotlib.pyplot as plt
import collections

# Génération des nombres aléatoires entre 0 et 100 (10 millions de fois)
num_samples = 10_000_000
values = np.random.randint(0, 100, num_samples)

# Comptage des occurrences de chaque valeur
counter = collections.Counter(values)

# Extraction des données pour le graphique
x_values = sorted(counter.keys())  # Les valeurs uniques (0 à 99)
y_counts = [counter[x] for x in x_values]  # Le nombre d'occurrences de chaque valeur


# Création du graphique
plt.figure(figsize=(12, 8))
plt.bar(x_values, y_counts, width=0.8, align='center',
        edgecolor="black", color="none")

plt.xlabel("Valeurs générées", fontsize=14)
plt.ylabel("Nombre d'occurrences", fontsize=14)
plt.title("Distribution des valeurs générées aléatoirement",  fontsize=16)

plt.xticks(range(0, 101, 10))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Enregistrement avant d'afficher
plt.savefig("uniforme.png", dpi=300)  # dpi=300 pour une meilleure qualité

# Affichage du graphique
plt.show()
