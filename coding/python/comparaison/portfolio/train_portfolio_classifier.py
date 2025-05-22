import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump

conn = sqlite3.connect("db/graphs.db")

valid_ids_df = pd.read_sql("SELECT DISTINCT graphe_id FROM performances", conn)
valid_ids = set(valid_ids_df["graphe_id"])

labels_df = pd.read_sql("SELECT graphe_id, heuristique FROM best_avg_size_per_heuristic", conn)
labels_df = labels_df[labels_df["graphe_id"].isin(valid_ids)]

features_df = pd.read_sql("SELECT * FROM graphes", conn)
features_df = features_df[features_df["id"].isin(labels_df["graphe_id"])]

features_df = features_df.drop(columns=["id", "canonical_form", "graph_name"], errors="ignore")

# Convertir colonnes booléennes
bool_cols = features_df.select_dtypes(include=["object"]).columns
for col in bool_cols:
    features_df[col] = features_df[col].map({"True": 1, "False": 0, "true": 1, "false": 0})

# Forcer conversion float
features_df = features_df.apply(pd.to_numeric, errors="coerce")

# Supprimer colonnes trop vides (>50% NULL) puis lignes restantes avec NULL
features_df = features_df.dropna(axis=1, thresh=int(0.5 * len(features_df)))
features_df["graphe_id"] = labels_df["graphe_id"].values
features_df = features_df.dropna()

# Jointure features + labels
df = features_df.merge(labels_df, on="graphe_id")
X_full = df.drop(columns=["graphe_id", "heuristique"])
y_full = df["heuristique"]

# === Split 80%/20% sur les instances ===
instance_ids = df["graphe_id"].unique()
train_ids, test_ids = train_test_split(instance_ids, test_size=0.2, random_state=42)

X_train = X_full[df["graphe_id"].isin(train_ids)]
y_train = y_full[df["graphe_id"].isin(train_ids)]
X_test = X_full[df["graphe_id"].isin(test_ids)]
y_test = y_full[df["graphe_id"].isin(test_ids)]


used_features = X_train.columns.tolist()
dump(used_features, "portfolio/model/used_features.pkl")

# === Encodage des labels ===
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# === Entraînement ===
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train_enc)



# === Évaluation ===
y_pred = clf.predict(X_test)
print("=== Rapport de classification (Portfolio Apprenant) ===")
print(classification_report(
    y_test_enc,
    y_pred,
    labels=le.transform(le.classes_),
    target_names=le.classes_,
    zero_division=0
))

dump(clf, "portfolio/model/portfolio_model.pkl")
dump(le, "portfolio/model/portfolio_label_encoder.pkl")

print("Modèle sauvegardé dans 'portfolio_model.pkl'")
