import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis le fichier Excel
file_path = 'Classeur1.xlsx'
df = pd.read_excel(file_path)

algorithms = df['Heuristic'].unique()
for algo in algorithms:
    algo_df = df[df['Heuristic'] == algo]

# # Choisir un algorithme spécifique
# selected_algo = 'DFS'  # Remplace par l'algorithme de ton choix

    # Filtrer les données pour cet algorithme
    algo_df = df[df['Heuristic'] == algo]

    # Configuration de la charte graphique pour correspondre à l'exemple fourni
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (8, 6),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": False,
    })

    # Créer le graphique
    plt.figure()

    # Tracer l'histogramme en utilisant les valeurs existantes
    plt.hist(algo_df['Approximation_Ratio'], bins=50, edgecolor='black', alpha=1, linewidth=1.2)

    # Ajouter les labels et les limites de l'axe des x
    plt.xlabel(r"\textbf{Pourcentage d'erreur}")
    plt.ylabel(r"\textbf{Nombre de solutions}")
    plt.xlim(0, 100)

    # Ajouter le titre correspondant
    plt.title(r"\textbf{" + algo.replace('_', ' ') + r"}")

    # Ajustement des axes pour correspondre au style attendu
    plt.xticks(range(0, 110, 20))
    plt.yticks(range(0, 300, 50))

    # Afficher le graphique
    plt.show()
