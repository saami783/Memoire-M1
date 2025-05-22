import pandas as pd
import numpy as np
from collections import defaultdict

# Calcul le nombre d'instances dominées par un algorithme pour une mesure (pire cas par exemple).

EXCEL_FILE = "performances.xlsx"

def generate_stats(sheet_name, metric, excel_file=EXCEL_FILE) : 

    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Vérification des colonnes nécessaires
    required_columns = {'Id', 'Heuristic', 'Optimal_Size', 'Worst_Size'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    all_heuristics = df['Heuristic'].unique()

    dominance_counter = {heur: 0 for heur in all_heuristics}

    for instance_id, group in df.groupby('Id'):
        min_pire_cas = group[metric].min()
        dominants = group[group[metric] == min_pire_cas]['Heuristic']
        for algo in dominants:
            dominance_counter[algo] += 1

    # Construction du tableau de classement
    dominance_df = pd.DataFrame(
        list(dominance_counter.items()), columns=['Heuristic', 'Instances dominées']
    )
    dominance_df = dominance_df.sort_values(by='Instances dominées', ascending=False)

    print("\nClassement par nombre d'instances dominées en " + (metric) + " pour la feuille " + sheet_name + " :")
    print(dominance_df.to_string(index=False))

# generate_stats("bhoslib", "Pire cas")
# generate_stats("barabasi_albert", "Pire cas")
# generate_stats("erdos_renyi", "Pire cas")
# generate_stats("regular", "Pire cas")
# generate_stats("tree", "Pire cas")
# generate_stats("HoG", "Pire cas")
generate_stats("kernels_hog", "Meilleur cas")