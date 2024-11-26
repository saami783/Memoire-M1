import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# Solution optimale
optimal_solution = 420

# Indiquez le chemin des fichiers pour GIC et MDG
gic_files = sorted(glob.glob("out/bhoslib/frb30-15/gic_results_frb30-15-*_cleaned.csv"))
mdg_files = sorted(glob.glob("out/bhoslib/frb30-15/mdg_results_frb30-15-*_cleaned.csv"))

# Fonction pour calculer le pourcentage d'erreur ajusté
def calculate_adjusted_error_percentage(df, optimal_solution, worst_solution):
    return np.floor(((df['Solution Size'].mean() - optimal_solution) / (worst_solution - optimal_solution)) * 100)

# Collecte des pourcentages d'erreur ajustés pour GIC et MDG
gic_adjusted_errors = []
for file in gic_files:
    df = pd.read_csv(file)
    worst_solution = df['Solution Size'].max()
    gic_adjusted_errors.append(calculate_adjusted_error_percentage(df, optimal_solution, worst_solution))

mdg_adjusted_errors = []
for file in mdg_files:
    df = pd.read_csv(file)
    worst_solution = df['Solution Size'].max()
    mdg_adjusted_errors.append(calculate_adjusted_error_percentage(df, optimal_solution, worst_solution))

# Convertir les résultats en DataFrames pour la visualisation
gic_adjusted_errors_df = pd.DataFrame(gic_adjusted_errors, columns=['Adjusted Error Percentage'])
mdg_adjusted_errors_df = pd.DataFrame(mdg_adjusted_errors, columns=['Adjusted Error Percentage'])

# Graphique pour GIC
plt.figure(figsize=(10, 6))
plt.hist(gic_adjusted_errors_df['Adjusted Error Percentage'], bins=30, edgecolor='black')
plt.title('Distribution des pourcentages d\'erreur ajustés pour GIC')
plt.xlabel('Pourcentage d\'erreur ajusté')
plt.ylabel('Nombre de solutions')
plt.show()

# Graphique pour MDG
plt.figure(figsize=(10, 6))
plt.hist(mdg_adjusted_errors_df['Adjusted Error Percentage'], bins=30, edgecolor='black')
plt.title('Distribution des pourcentages d\'erreur ajustés pour MDG')
plt.xlabel('Pourcentage d\'erreur ajusté')
plt.ylabel('Nombre de solutions')
plt.show()
