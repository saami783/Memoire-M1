import pandas as pd
import matplotlib.pyplot as plt

# Charger les données CSV
mdg_df = pd.read_csv('mdg_results.csv')
gic_df = pd.read_csv('gic_results.csv')

# Histogramme des tailles des solutions pour MDG
plt.figure(figsize=(12, 6))
plt.hist(mdg_df['Solution Size'], bins=20, alpha=0.7, label='MDG')
plt.title('Distribution des tailles des solutions pour l\'algorithme MDG')
plt.xlabel('Taille de la solution')
plt.ylabel('Nombre de solutions')
plt.legend()
plt.grid(True)
plt.show()

# Histogramme des tailles des solutions pour GIC
plt.figure(figsize=(12, 6))
plt.hist(gic_df['Solution Size'], bins=20, alpha=0.7, label='GIC', color='orange')
plt.title('Distribution des tailles des solutions pour l\'algorithme GIC')
plt.xlabel('Taille de la solution')
plt.ylabel('Nombre de solutions')
plt.legend()
plt.grid(True)
plt.show()

# Graphique du pourcentage d'erreur
plt.figure(figsize=(12, 6))
plt.plot(mdg_df['Run Number'], mdg_df['Approximation Ratio'], label='MDG', marker='o')
plt.plot(gic_df['Run Number'], gic_df['Approximation Ratio'], label='GIC', marker='x')
plt.title('Ratio d\'approximation par itération')
plt.xlabel('Numéro de l\'itération')
plt.ylabel('Ratio d\'approximation')
plt.axhline(y=1, color='red', linestyle='--', label='Solution optimale')
plt.legend()
plt.grid(True)
plt.show()

# Graphique des temps d'exécution moyens
mdg_avg_time = mdg_df['Execution Time (s)'].mean()
gic_avg_time = gic_df['Execution Time (s)'].mean()

plt.figure(figsize=(8, 6))
plt.bar(['MDG', 'GIC'], [mdg_avg_time, gic_avg_time], color=['blue', 'orange'])
plt.title('Temps d\'exécution moyen par algorithme')
plt.ylabel('Temps (secondes)')
plt.grid(axis='y')
plt.show()
