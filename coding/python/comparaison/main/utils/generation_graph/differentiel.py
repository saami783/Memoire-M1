import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

file_name = "rapport approximation differentiel.xlsx"
sheet_name = "bhoslib"

df = pd.read_excel(file_name, sheet_name=sheet_name, usecols=["Heuristic", "Rapport"])

print(f"🔍 Données brutes chargées : {len(df)} lignes")
print(df.head())

df["Rapport"] = pd.to_numeric(df["Rapport"], errors='coerce')

df = df.dropna()

output_folder = f"graphics/differentiel/{sheet_name}"
os.makedirs(output_folder, exist_ok=True)

grouped = df.groupby(["Heuristic", "Rapport"]).size().reset_index(name="Nombre de solutions")

heuristics = grouped["Heuristic"].unique()
print(f"\n📌 Heuristiques trouvées : {heuristics}")

# Générer un graphique pour chaque heuristique
for heuristic in heuristics:
    df_filtered = grouped[grouped["Heuristic"] == heuristic]

    if df_filtered.empty:
        print(f"⚠ Aucune donnée pour {heuristic}, graphique ignoré.")
        continue

    print(f"🔹 Heuristic : {heuristic} - {len(df_filtered)} entrées")
    print(df_filtered.head())

    y_max = df_filtered["Nombre de solutions"].max()

    # Définition dynamique du pas des Y-ticks
    if y_max <= 10:
        y_step = 1
    elif y_max <= 50:
        y_step = 5
    elif y_max <= 100:
        y_step = 10
    else:
        y_step = 20

    y_ticks = np.arange(0, y_max + y_step, y_step)

    # Créer le graphique
    plt.figure(figsize=(12, 8))
    plt.bar(df_filtered["Rapport"], df_filtered["Nombre de solutions"], width=1,
            edgecolor="black", color="none")  # Barres creuses avec bordures noires

    # Configuration du graphique
    plt.title(f"{heuristic}", fontsize=16)
    plt.xlabel("Pourcentage d'erreur", fontsize=14)
    plt.ylabel("Nombre de solutions", fontsize=14)

    # Ajustement des échelles des axes
    plt.xticks(range(0, 101, 10))  # Graduation tous les 10%
    plt.yticks(y_ticks)  # Graduation cohérente pour y

    # Suppression de la ligne Y et de la grille Y
    plt.grid(axis='y', linestyle='', alpha=0)  # Désactive la grille Y

    plt.tight_layout()

    # Enregistrer le graphique
    output_filename = os.path.join(output_folder, f"{sheet_name}_{heuristic}.png")
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"✅ Graphique enregistré : {output_filename}")
