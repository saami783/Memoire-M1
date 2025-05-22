Dans le dossier `utils` :
- Le fichier `import_performances_to_db.py` créer une table `performances` dans la base de données, en fonction des données du fichier
`performances.xlsx`. Chaque performance sera persistée dans la base de données avec une jointure sur chaque instance de la table `graphes`.

- Le script `meilleure_heuristique_en_moyenne.py` créer une table `best_avg_size_per_heuristic` afin de permet de connaître la liste des graphe_id (instances) pour lesquels chaque heuristique est la meilleure en moyenne, selon avg_size.
- Le script `predict_from_graph_id.py` Donne l’heuristique prédite pour un graphe donné (par son id).
- Le script `analyze_misclassifications.py` Affiche les erreurs de prédiction avec :
- - l'ID du graphe
- - la vraie heuristique
- - la prédiction
- - les features associées