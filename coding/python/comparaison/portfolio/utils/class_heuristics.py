import sqlite3
import pandas as pd

DB_PATH = "../performances.db"

conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    p.heuristique,
    p.rapport,
    g.class
FROM performances AS p
JOIN graphes AS g
    ON p.graphe_id = g.id
"""
df = pd.read_sql_query(query, conn)
df['rapport'] = df['rapport'].astype(float)

df_summary = (
    df
    .groupby(['class', 'heuristique'])['rapport']
    .agg(min_rapport='min', max_rapport='max')
    .reset_index()
)
df_summary['score_global'] = (df_summary['min_rapport'] + df_summary['max_rapport']) / 2

idx_min = df_summary.groupby('class')['score_global'].idxmin()
df_best = df_summary.loc[idx_min, ['class', 'heuristique', 'score_global']]

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS best_by_class (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class TEXT NOT NULL,
    best_heuristic TEXT NOT NULL,
    score_global REAL NOT NULL
);
""")

cursor.execute("DELETE FROM best_by_class;")

for _, row in df_best.iterrows():
    cursor.execute("""
        INSERT INTO best_by_class (class, best_heuristic, score_global)
        VALUES (?, ?, ?)
    """, (row['class'], row['heuristique'], row['score_global']))

conn.commit()
conn.close()
