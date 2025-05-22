import sqlite3
import pandas as pd

conn = sqlite3.connect("db/graphs.db")

df = pd.read_sql("SELECT graphe_id, heuristique, avg_size FROM performances", conn)

idx_min = df.groupby("graphe_id")["avg_size"].idxmin()
best_df = df.loc[idx_min][["graphe_id", "heuristique"]]

conn.execute("""
CREATE TABLE IF NOT EXISTS best_avg_size_per_heuristic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    heuristique TEXT NOT NULL,
    graphe_id INTEGER NOT NULL
)
""")

conn.execute("DELETE FROM best_avg_size_per_heuristic")

best_df.to_sql("best_avg_size_per_heuristic", conn, if_exists="append", index=False)
conn.commit()
conn.close()

print("Table best_avg_size_per_heuristic créée et remplie avec succès.")
