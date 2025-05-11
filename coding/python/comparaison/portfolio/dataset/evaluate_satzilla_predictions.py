# Ce script évalue les prédictions faites par le modèle SatZilla-like
# en les comparant aux vraies performances issues de la base de données.
# Il génère un rapport de classification et exporte les résultats détaillés.

import pandas as pd
import sqlite3
from sklearn.metrics import classification_report

# === 1. Charger les prédictions faites localement ===
predictions_df = pd.read_csv("satzilla_predictions.csv")  # colonnes : 'id', 'heuristique_prédite'

# === 2. Charger les vraies performances depuis la base de données ===
conn = sqlite3.connect("../performances.db")
performances_df = pd.read_sql("SELECT graphe_id, heuristique, rapport FROM performances", conn)

# === 3. Identifier les heuristiques présentes dans les prédictions ===
valid_heuristics = predictions_df["heuristique_prédite"].unique()
filtered_perf = performances_df[performances_df["heuristique"].isin(valid_heuristics)]

# === 4. Trouver la vraie meilleure heuristique pour chaque graphe ===
min_idx = filtered_perf.groupby("graphe_id")["rapport"].idxmin()
true_best = filtered_perf.loc[min_idx].reset_index(drop=True)
true_best = true_best.rename(columns={"heuristique": "heuristique_réelle"})


# === 5. Fusionner les prédictions et les vraies valeurs ===
eval_df = predictions_df.merge(true_best, left_on="id", right_on="graphe_id")
eval_df["succès"] = eval_df["heuristique_prédite"] == eval_df["heuristique_réelle"]

# === 6. Générer et afficher le rapport de classification ===
report = classification_report(
    eval_df["heuristique_réelle"],
    eval_df["heuristique_prédite"],
    zero_division=0  # évite les warnings quand support=0
)

print("=== Rapport de classification ===")
print(report)

# === 7. Sauvegarder les résultats détaillés ===
eval_df[["id", "heuristique_prédite", "heuristique_réelle", "succès"]].to_csv("satzilla_evaluation.csv", index=False)

print("\n=== Répartition des heuristiques prédites ===")
print(eval_df["heuristique_prédite"].value_counts())