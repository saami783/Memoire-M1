import sqlite3
import pandas as pd
from tqdm import tqdm

df = pd.read_excel("performances.xlsx", sheet_name="algos")

conn = sqlite3.connect("../performances.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS performances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        graphe_id INTEGER NOT NULL,
        heuristique TEXT NOT NULL,
        average_size REAL,
        best_size INTEGER,
        worst_size INTEGER,
        pire_cas INTEGER,
        moyen_cas INTEGER,
        meilleur_cas INTEGER,
        rapport INTEGER,
        FOREIGN KEY (graphe_id) REFERENCES graphes(id)
    );
""")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Insertion des performances"):
    cursor.execute("""
        INSERT INTO performances (
            graphe_id,
            heuristique,
            average_size,
            best_size,
            worst_size,
            pire_cas,
            moyen_cas,
            meilleur_cas,
            rapport
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int(row["Id"]),
        str(row["Heuristic"]),
        float(row["Average_Size.1"]),
        int(row["Best_Size"]),
        int(row["Worst_Size"]),
        int(row["Pire cas"]),
        int(row["Moyen cas"]),
        int(row["Meilleur cas"]),
        int(row["Rapport"])
    ))

conn.commit()
conn.close()
print("Insertion terminée avec succès.")
