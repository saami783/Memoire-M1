import sqlite3
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

clf = load("portfolio/portfolio_model.pkl")
le = load("portfolio/portfolio_label_encoder.pkl")

conn = sqlite3.connect("db/graphs.db")

# Charger features et labels valides
valid_ids_df = pd.read_sql("SELECT DISTINCT graphe_id FROM performances", conn)
valid_ids = set(valid_ids_df["graphe_id"])

labels_df = pd.read_sql("SELECT graphe_id, heuristique FROM best_avg_size_per_heuristic", conn)
labels_df = labels_df[labels_df["graphe_id"].isin(valid_ids)]

features_df = pd.read_sql("SELECT * FROM graphes", conn)
features_df = features_df[features_df["id"].isin(labels_df["graphe_id"])]
features_df = features_df.drop(columns=["id", "canonical_form", "graph_name"], errors="ignore")

# Nettoyage
bool_cols = features_df.select_dtypes(include=["object"]).columns
for col in bool_cols:
    features_df[col] = features_df[col].map({"True": 1, "False": 0, "true": 1, "false": 0})
features_df = features_df.apply(pd.to_numeric, errors="coerce")
features_df = features_df.dropna(axis=1, thresh=int(0.5 * len(features_df)))
features_df["graphe_id"] = labels_df["graphe_id"].values
features_df = features_df.dropna()

# Fusion
df = features_df.merge(labels_df, on="graphe_id")
X = df.drop(columns=["graphe_id", "heuristique"])
y = df["heuristique"]
graphe_ids = df["graphe_id"]

# Encoder
le_full = LabelEncoder()
y_enc = le_full.fit_transform(y)
y_pred = clf.predict(X)

# Comparaison
decoded_y = le_full.inverse_transform(y_enc)
decoded_pred = le_full.inverse_transform(y_pred)

misclassified = pd.DataFrame({
    "graphe_id": graphe_ids,
    "heuristique_vraie": decoded_y,
    "heuristique_predite": decoded_pred
})
misclassified = misclassified[misclassified["heuristique_vraie"] != misclassified["heuristique_predite"]]

# Affichage
print(f"{len(misclassified)} erreurs de prédiction sur {len(df)} instances.")
print(misclassified.head(10))  # affichage des premières erreurs

misclassified.to_csv("portfolio/misclassifications.csv", index=False)
print("Résultats sauvegardés dans 'misclassifications.csv'")
