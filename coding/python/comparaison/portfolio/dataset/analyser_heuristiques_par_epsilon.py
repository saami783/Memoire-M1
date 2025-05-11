# Ce script analyse pour chaque heuristique combien de fois elle est la meilleure
# et combien de fois elle est "bonne" au sens de rapport ≤ ε * meilleur_rapport

import sqlite3
import pandas as pd

# Connexion à la base
conn = sqlite3.connect("performances.db")
performances_df = pd.read_sql("SELECT graphe_id, heuristique, rapport FROM performances", conn)

# Étape 1 : trouver le meilleur rapport pour chaque graphe
min_rapports = performances_df.groupby("graphe_id")["rapport"].min().reset_index()
min_rapports = min_rapports.rename(columns={"rapport": "best_rapport"})

# Étape 2 : fusionner avec les données d'origine
merged = performances_df.merge(min_rapports, on="graphe_id")

# Étape 3 : définir les heuristiques "bonnes" selon epsilon
epsilon = 1.05  # seuil de tolérance
merged["is_good"] = merged["rapport"] <= epsilon * merged["best_rapport"]

# Étape 4 : compter les bons cas et les cas exacts
good_counts = merged[merged["is_good"]].groupby("heuristique").size().reset_index(name="nb_bonnes_avec_epsilon")
exact_counts = merged[merged["rapport"] == merged["best_rapport"]].groupby("heuristique").size().reset_index(name="nb_exactement_meilleure")

# Fusionner les deux comptages
summary = pd.merge(good_counts, exact_counts, on="heuristique", how="outer").fillna(0)
summary = summary.sort_values(by="nb_bonnes_avec_epsilon", ascending=False)

# Affichage
print("\n=== Résumé des heuristiques (epsilon =", epsilon, ") ===")
print(summary)

# Optionnel : sauvegarder dans un CSV
summary.to_csv("analyse_heuristiques_epsilon.csv", index=False)
print("\n✅ Résultats sauvegardés dans analyse_heuristiques_epsilon.csv")
