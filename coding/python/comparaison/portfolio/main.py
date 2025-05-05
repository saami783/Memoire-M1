import sqlite3
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from extract_properties import *

if __name__ == "__main__":
    conn = sqlite3.connect("performances.db")

    df_graphs = pd.read_sql_query("""
      SELECT *, CAST(largest_eigenvalue AS REAL) AS largest_eigenvalue
      FROM graphes
      """, conn, index_col="id")

    df_best = pd.read_sql_query("""
        WITH ranked AS (SELECT *,ROW_NUMBER() OVER(
        PARTITION BY graphe_id 
        ORDER BY Rapport ASC, heuristique
            ) AS rn
        FROM performances)
        SELECT graphe_id, heuristique AS best_heuristic
        FROM ranked
        WHERE rn = 1
        """, conn)

    conn.close()

    df = df_best.merge(df_graphs, left_on="graphe_id", right_index=True, how='inner')

    df = df.drop(columns=['graph_name', 'canonical_form'], errors='ignore')

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    le_heuristic = LabelEncoder()
    y = le_heuristic.fit_transform(df['best_heuristic'])

    X = df[numeric_cols].drop(columns=['graphe_id', 'best_heuristic'], errors='ignore')

    model = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=15,
            weights='distance',
            metric='cosine'
        )
    )

    model.fit(X, y)

    graph_data = extract_graph_properties()
    new_graph_df = pd.DataFrame([graph_data])

    new_graph_df = new_graph_df.reindex(columns=X.columns, fill_value=np.nan)

    pred = model.predict(new_graph_df)
    print(f"Heuristique recommandée : {le_heuristic.inverse_transform(pred)[0]}")