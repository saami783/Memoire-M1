import subprocess

print("🔁 Étape 1 : Génération du dataset SatZilla...")
subprocess.run(["python", "satzilla_dataset.py"], check=True)

print("🧠 Étape 2 : Entraînement des modèles pour chaque heuristique...")
subprocess.run(["python", "train_satzilla_models.py"], check=True)

print("🤖 Étape 3 : Prédiction de la meilleure heuristique pour chaque graphe...")
subprocess.run(["python", "dataset_gen.py"], check=True)

print("📊 Étape 4 : Évaluation des performances des prédictions...")
subprocess.run(["python", "evaluate_satzilla_predictions.py"], check=True)

print("📊 Étape 5 : Réévaluation après mise à jour...")
subprocess.run(["python", "evaluate_satzilla_predictions.py"], check=True)

print("✅ Pipeline SatZilla exécuté avec succès.")

