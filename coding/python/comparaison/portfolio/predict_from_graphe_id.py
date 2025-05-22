import sqlite3
import pandas as pd
from joblib import load

# graphe_id_to_predict = 981
graphe_id_to_predict = 1160

clf = load("portfolio/model/portfolio_model.pkl")
le = load("portfolio/model/portfolio_label_encoder.pkl")
used_features = load("portfolio/model/used_features.pkl")

conn = sqlite3.connect("db/graphs.db")

features_df = pd.read_sql(
    "SELECT * FROM graphes WHERE id = ?",
    conn,
    params=(graphe_id_to_predict,)
)
if features_df.empty:
    print(f"Aucun graphe trouvé avec l'id {graphe_id_to_predict}")
    exit()

features_df = features_df.drop(columns=["id", "canonical_form", "graph_name"], errors="ignore")

bool_cols = features_df.select_dtypes(include=["object"]).columns
for col in bool_cols:
    features_df[col] = features_df[col].map({"True": 1, "False": 0, "true": 1, "false": 0})

features_df = features_df.apply(pd.to_numeric, errors="coerce")

features_df = features_df.reindex(columns=used_features)

features_df = features_df.fillna(0)

prediction = clf.predict(features_df)[0]
heuristique = le.inverse_transform([prediction])[0]

print(f"Heuristique prédite pour le graphe {graphe_id_to_predict} : **{heuristique}**")
