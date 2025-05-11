# Ce script applique les modèles de prédiction de chaque algorithmes
# puis les enregistre dans un fichier CSV satzilla_predictions.csv
# requires : satzilla_testset.csv
import pandas as pd
from joblib import load
import os

# Charger les features extraites (fichier généré plus tôt)
df = pd.read_csv("satzilla_testset.csv")

# Dossier contenant les modèles
model_dir = "satzilla_models"

# Chargement automatique de tous les modèles disponibles
models = {}
for file in os.listdir(model_dir):
    if file.endswith("_model.pkl"):
        heuristique_name = file.replace("_model.pkl", "").replace("_", " ").title()
        models[heuristique_name] = load(os.path.join(model_dir, file))

# Fonction pour prédire le meilleur heuristique pour un graphe donné
def predict_best(instance):
    instance_df = instance.to_frame().T  # transforme la Series en DataFrame 1 ligne
    predictions = {
        name: model.predict(instance_df)[0]
        for name, model in models.items()
    }
    return min(predictions, key=predictions.get)

# Application aux graphes
df["heuristique_prédite"] = df.drop(columns=["id"]).apply(predict_best, axis=1)

# Sauvegarde du fichier avec les prédictions
df.to_csv("satzilla_predictions.csv", index=False)
print("✅ Prédictions sauvegardées dans satzilla_predictions.csv")
