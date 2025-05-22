# plot_rapport.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import math

def plot_like_rapport(file_name, sheet_name, original_col, output_folder):
    """
    Lit la colonne 'original_col' dans la feuille 'sheet_name' de 'file_name'.
    Renomme cette colonne en 'Rapport', arrondit à l'inférieur, et trace
    des histogrammes (Heuristic vs Nombre de solutions) en pourcentage d'erreur.
    - Si la valeur max dépasse 100, on étend l'axe X au multiple de 10 supérieur.
    - Sinon, on reste sur l'axe classique [0..100].
    Sauvegarde les graphiques dans 'output_folder'.
    """

    # 1) Préparation du dossier de sortie
    os.makedirs(output_folder, exist_ok=True)

    # 2) Lecture du fichier Excel (uniquement Heuristic + original_col)
    df = pd.read_excel(file_name, sheet_name=sheet_name, usecols=["Heuristic", original_col])
    # Renommer la colonne en "Rapport"
    df.rename(columns={original_col: "Rapport"}, inplace=True)

    # 3) Conversion et filtrage
    df["Rapport"] = pd.to_numeric(df["Rapport"], errors='coerce')
    df.dropna(subset=["Rapport"], inplace=True)

    # 4) Arrondi à l’inférieur (floor) pour réduire les valeurs décimales
    df["Rapport"] = df["Rapport"].apply(math.floor)

    # 5) Groupby (Heuristic, Rapport)
    grouped = df.groupby(["Heuristic", "Rapport"]).size().reset_index(name="Nombre de solutions")
    heuristics = grouped["Heuristic"].unique()

    # 6) Boucle : tracer un histogramme par Heuristic
    for heuristic in heuristics:
        df_filtered = grouped[grouped["Heuristic"] == heuristic]
        if df_filtered.empty:
            print(f"Aucune donnée pour {heuristic}, graphique ignoré.")
            continue

        # Détermination de l'axe Y
        y_max = df_filtered["Nombre de solutions"].max()
        if y_max <= 10:
            y_step = 1
        elif y_max <= 50:
            y_step = 5
        elif y_max <= 100:
            y_step = 10
        else:
            y_step = 20
        y_ticks = np.arange(0, y_max + y_step, y_step)

        # 7) Détermination de l'axe X : si le max dépasse 100, on élargit
        max_rapport_val = df_filtered["Rapport"].max()
        if max_rapport_val > 100:
            # Arrondir au multiple de 10 supérieur
            x_lim = int(math.ceil(max_rapport_val / 10.0)) * 10
        else:
            x_lim = 100

        # 8) Tracé
        plt.figure(figsize=(12, 8))
        plt.bar(
            df_filtered["Rapport"],
            df_filtered["Nombre de solutions"],
            width=1,
            edgecolor="black",
            color="none"
        )

        # Titres et labels
        plt.title(f"{heuristic}", fontsize=16)
        plt.xlabel(f"Pourcentage d'erreur", fontsize=14)
        plt.ylabel("Nombre de solutions", fontsize=14)

        # Axe X dynamique
        # De 0 à x_lim par pas de 10
        plt.xticks(range(0, x_lim+1, 10))
        plt.xlim([0, x_lim])

        # Axe Y (déjà déterminé)
        plt.yticks(y_ticks)

        # Grille horizontale désactivée
        plt.grid(axis='y', linestyle='', alpha=0)

        plt.tight_layout()

        # 9) Sauvegarde
        output_filename = os.path.join(output_folder, f"{heuristic}.png")
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"Graphique enregistré : {output_filename}")
