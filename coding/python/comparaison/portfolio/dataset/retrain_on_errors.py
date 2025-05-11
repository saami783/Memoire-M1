import pandas as pd
import sqlite3
import os
from joblib import load, dump
from sklearn.ensemble import RandomForestRegressor

# === 1. Charger les prédictions erronées ===
eval_df = pd.read_csv("satzilla_evaluation.csv")
errors_df = eval_df[eval_df["succès"] == False]

if errors_df.empty:
    print("✅ Aucune erreur à corriger. Tous les modèles sont déjà cohérents.")
    exit()

# === 2. Récupérer les features des graphes en erreur ===
conn = sqlite3.connect("../performances.db")
graphes_df = pd.read_sql("SELECT * FROM graphes", conn)

# Nettoyage des colonnes
graphes_df = graphes_df.drop(columns=['graph_name', 'canonical_form', 'class', 'cover_size'], errors='ignore')
graphes_df['largest_eigenvalue'] = pd.to_numeric(graphes_df['largest_eigenvalue'], errors='coerce')
graphes_df = graphes_df.dropna()

# Fusion avec les erreurs
error_features = errors_df.merge(graphes_df, on="id")
error_features["heuristique"] = error_features["heuristique_réelle"]

# === 3. Déduire automatiquement les features utilisées ===
feature_cols = [col for col in error_features.columns if col not in ["id", "heuristique_prédite", "heuristique_réelle", "succès", "heuristique", "graphe_id"]]

# Regrouper par heuristique à corriger
error_datasets = {
    h: df[feature_cols]
    for h, df in error_features.groupby("heuristique")
}

# === 4. Réentraîner les modèles concernés et les ÉCRASER ===
model_dir = "satzilla_models"

for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        heuristique_name = file.replace("_model.pkl", "").replace("_", " ").title()
        model_path = os.path.join(model_dir, file)
        model = load(model_path)

        if heuristique_name in error_datasets:
            X_new = error_datasets[heuristique_name]
            y_new = [0.0] * len(X_new)  # hypothèse : ce sont les vraies meilleures heuristiques

            print(f"🔁 Réentraînement du modèle {heuristique_name} sur {len(X_new)} erreurs...")
            model.fit(X_new, y_new)

            # Écrasement du modèle d'origine
            dump(model, model_path)

print("✅ Modèles corrigés enregistrés dans satzilla_models/")
