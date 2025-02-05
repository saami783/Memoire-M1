import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(42)

# Définition des intervalles à tester
intervals = [(0, 10), (0, 1), (0, 100), (0, 1000), (0, 1000000)]
num_samples = 10_000_000

# Stockage des résultats
chi2_results = {}

for low, high in intervals:
    # Génération des nombres pseudo-aléatoires
    samples = np.random.randint(low, high + 1, size=num_samples)

    # Comptage des occurrences
    unique, counts = np.unique(samples, return_counts=True)
    observed_freq = dict(zip(unique, counts))

    # Distribution uniforme théorique
    expected_freq = num_samples / (high - low + 1)

    # Liste des fréquences observées et attendues
    observed = np.array([observed_freq.get(i, 0) for i in range(low, high + 1)])
    expected = np.full_like(observed, expected_freq)

    # Normalisation des fréquences attendues
    expected = expected * (observed.sum() / expected.sum())

    # Test du Chi-2
    chi2_stat, p_value = stats.chisquare(observed, expected)
    chi2_results[(low, high)] = {"Chi2 Stat": chi2_stat, "p-value": p_value}

# Affichage des résultats sous forme de tableau
df_chi2 = pd.DataFrame.from_dict(chi2_results, orient="index")
print(df_chi2)

# Génération du graphique pour l'intervalle [0,100]
low, high = (0, 100)
samples = np.random.randint(low, high + 1, size=num_samples)
unique, counts = np.unique(samples, return_counts=True)

plt.figure(figsize=(12, 6))
plt.bar(unique, counts, width=0.8)
plt.xticks(np.arange(low, high + 1, step=10))  # Affichage des abscisses de 10 en 10
plt.xlabel("Valeurs générées")
plt.ylabel("Nombre de fois généré")
plt.title(f"Distribution des valeurs générées pour l'intervalle [{low}, {high}]")
plt.show()