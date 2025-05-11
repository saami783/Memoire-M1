# Ce scripte permet d'entraîner des modèles de prédiction de performance pour chaque heuristique
# en utilisant un ensemble de données de graphes. Il utilise un modèle de forêt aléatoire
# pour prédire le rapport de performance en fonction des caractéristiques du graphe.
# puis genere un fichier .plk pour chaque heuristique.
import pandas as pd
import sqlite3
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump

conn = sqlite3.connect("../performances.db")

# Charger les données
performances_df = pd.read_sql("SELECT graphe_id, heuristique, rapport FROM performances", conn)
graphes_df = pd.read_sql("SELECT * FROM graphes", conn)

# Nettoyage
graphes_df = graphes_df.drop(columns=['graph_name', 'canonical_form', 'class', 'cover_size'], errors='ignore')
graphes_df['largest_eigenvalue'] = pd.to_numeric(graphes_df['largest_eigenvalue'], errors='coerce')
graphes_df = graphes_df.dropna()

# Fusion
merged_df = performances_df.merge(graphes_df, left_on='graphe_id', right_on='id').drop(columns=['id'])
merged_df = merged_df.dropna()

# Filtrer heuristiques fréquentes
heuristic_counts = merged_df['heuristique'].value_counts()
valid_heuristics = heuristic_counts[heuristic_counts >= 100].index
filtered_df = merged_df[merged_df['heuristique'].isin(valid_heuristics)]

# Déduire dynamiquement toutes les features (en excluant graphe_id, heuristique, rapport)
feature_cols = [col for col in filtered_df.columns if col not in ['graphe_id', 'heuristique', 'rapport']]


# Dossier de sortie
os.makedirs("satzilla_models", exist_ok=True)

# Entraînement et sauvegarde d'un modèle par heuristique
for heuristic in valid_heuristics:
    data = filtered_df[filtered_df['heuristique'] == heuristic]
    X = data[feature_cols]
    y = data['rapport']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Sauvegarde du modèle
    filename = f"{heuristic.lower().replace(' ', '_')}_model.pkl"
    dump(model, os.path.join("satzilla_models", filename))

print("✅ Tous les modèles SatZilla-like ont été entraînés et sauvegardés.")
