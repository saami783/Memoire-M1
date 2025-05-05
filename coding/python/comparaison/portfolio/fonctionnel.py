import sqlite3
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Connexion à la base SQLite
conn = sqlite3.connect("performances.db")

# Chargement des graphes
df_graphs = pd.read_sql_query("SELECT * FROM graphes", conn, index_col="id")

# Meilleure heuristique par graphe
df_best = pd.read_sql_query("""
WITH min_score AS (
  SELECT graphe_id, MIN(Rapport) AS best_rapport
  FROM performances
  GROUP BY graphe_id
)
SELECT p.graphe_id, p.heuristique AS best_heuristic
FROM performances p
JOIN min_score m
  ON p.graphe_id = m.graphe_id
 AND p.Rapport   = m.best_rapport
""", conn)

conn.close()

# Fusion
df = df_best.merge(df_graphs, left_on="graphe_id", right_index=True, how="left")

# Nettoyage
df = df.drop(columns=["graph_name", "canonical_form"], errors="ignore")

# Conversion explicite de la colonne mal typée
df["largest_eigenvalue"] = pd.to_numeric(df["largest_eigenvalue"], errors="coerce")

# Encodage de la classe
le_class = LabelEncoder()
df["class"] = le_class.fit_transform(df["class"])

# Encodage de l’heuristique cible
le_heuristic = LabelEncoder()
y = le_heuristic.fit_transform(df["best_heuristic"])

# Features
X = df.drop(columns=["graphe_id", "best_heuristic"])

# Ne garder que les colonnes numériques
X = X.select_dtypes(include=["number"])

# Imputation des NaN par la moyenne
imputer = SimpleImputer(strategy="mean")

# Création du pipeline complet
model = make_pipeline(imputer, KNeighborsClassifier())
model.fit(X, y)

# Score sur les mêmes données
print("Score :", model.score(X, y))

# Prédiction d’un exemple
pred = model.predict(X.iloc[:1])
print("Prédiction sur le premier exemple :", le_heuristic.inverse_transform(pred)[0])
