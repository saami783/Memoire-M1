import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from extract_properties import extract_graph_properties
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

if __name__ == "__main__":
    conn = sqlite3.connect("performances.db")

    df_graphs = pd.read_sql_query("""
      SELECT *, CAST(largest_eigenvalue AS REAL) AS largest_eigenvalue
      FROM graphes
    """, conn, index_col="id")

    df_best = pd.read_sql_query("""
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY graphe_id 
                ORDER BY rapport ASC, heuristique
            ) AS rn
            FROM performances
        )
        SELECT graphe_id, heuristique AS best_heuristic
        FROM ranked
        WHERE rn = 1
    """, conn)

    conn.close()

    # Fusion et nettoyage
    df = df_best.merge(df_graphs, left_on="graphe_id", right_index=True, how='inner')
    df = df.drop(columns=['graph_name', 'canonical_form'], errors='ignore')

    # Encodage de la cible
    le_heuristic = LabelEncoder()
    y = le_heuristic.fit_transform(df['best_heuristic'])

    # Extraction des features numériques
    X = df.drop(columns=['graphe_id', 'best_heuristic', 'cover_size'], errors='ignore')
    X = X.select_dtypes(include=np.number)

    # Modèle XGBoost
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        verbosity=0
    )

    model.fit(X, y)

    # Prédiction sur un nouveau graphe
    graph_data = extract_graph_properties()
    new_graph_df = pd.DataFrame([graph_data])
    new_graph_df = new_graph_df.reindex(columns=X.columns, fill_value=np.nan)

    pred = model.predict(new_graph_df)
    print(f"Heuristique recommandée : {le_heuristic.inverse_transform(pred)[0]}")
