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

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_excel(file_name, sheet_name=sheet_name, usecols=["Heuristic", original_col])
    df.rename(columns={original_col: "Rapport"}, inplace=True)

    df["Rapport"] = pd.to_numeric(df["Rapport"], errors='coerce')
    df.dropna(subset=["Rapport"], inplace=True)

    df["Rapport"] = df["Rapport"].apply(math.floor)

    grouped = df.groupby(["Heuristic", "Rapport"]).size().reset_index(name="Nombre de solutions")
    heuristics = grouped["Heuristic"].unique()

    for heuristic in heuristics:
        df_filtered = grouped[grouped["Heuristic"] == heuristic]
        if df_filtered.empty:
            print(f"Aucune donnée pour {heuristic}, graphique ignoré.")
            continue

        # Axe Y : max et pas
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

        if len(y_ticks) > 10:
            y_ticks = np.linspace(0, y_max, 10, dtype=int)

        max_rapport_val = df_filtered["Rapport"].max()
        if max_rapport_val > 100:
            x_lim = int(math.ceil(max_rapport_val / 10.0)) * 10
        else:
            x_lim = 100

        plt.figure(figsize=(12, 8))
        bars = plt.bar(
            df_filtered["Rapport"],
            df_filtered["Nombre de solutions"],
            width=1,
            edgecolor="black",
            color="none"
        )

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    str(height),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=90
                )

        plt.title(f"{heuristic}", fontsize=16)
        plt.xlabel("Pourcentage d'erreur", fontsize=14)
        plt.ylabel("Nombre de solutions", fontsize=14)
        plt.xticks(range(0, x_lim+1, 10))
        plt.xlim([0, x_lim])
        plt.yticks(y_ticks)
        plt.grid(axis='y', linestyle='', alpha=0)

        plt.tight_layout()

        output_filename = os.path.join(output_folder, f"{heuristic}.png")
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"Graphique enregistré : {output_filename}")
