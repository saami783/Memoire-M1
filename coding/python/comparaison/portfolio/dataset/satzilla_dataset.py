import pandas as pd
import sqlite3

# Connexion à la base de données SQLite
conn = sqlite3.connect("../performances.db")

# Charger la table des graphes
graphes_df = pd.read_sql("SELECT * FROM graphes", conn)

# Nettoyage : suppression des colonnes inutiles
graphes_df = graphes_df.drop(columns=["graph_name", "canonical_form", "class", "cover_size"], errors="ignore")

# Conversion de 'largest_eigenvalue' au format numérique (important)
graphes_df["largest_eigenvalue"] = pd.to_numeric(graphes_df["largest_eigenvalue"], errors="coerce")

# Supprimer les lignes incomplètes
graphes_df = graphes_df.dropna()

# Sauvegarde des features utiles dans un fichier CSV
satzilla_testset = graphes_df.reset_index(drop=True)
satzilla_testset.to_csv("satzilla_testset.csv", index=False)

print("✅ Fichier satzilla_testset.csv généré avec succès.")
