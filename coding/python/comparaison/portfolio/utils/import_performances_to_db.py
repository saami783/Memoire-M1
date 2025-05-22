import pandas as pd
import sqlite3

excel_file = "performances.xlsx"
df = pd.read_excel(excel_file, sheet_name="HoG")

# Nettoyage et sélection des colonnes utiles
df = df.rename(columns={
    "Id": "graphe_id",
    "Heuristic": "heuristique",
    "Best_Size": "best_size",
    "Average_Size.1": "avg_size",
    "Worst_Size": "worst_size",
    "Average_Time": "avg_time",
    "Best_Time": "best_time",
    "Worst_Time": "worst_time",
    "Total_Time": "total_time",
    "Pire cas": "pire_cas",
    "Moyen cas": "moyen_cas",
    "Meilleur cas": "meilleur_cas",
    "Rapport": "rapport"
})

columns = [
    "graphe_id", "heuristique", "avg_size", "best_size", "worst_size",
    "avg_time", "best_time", "worst_time", "total_time",
    "pire_cas", "moyen_cas", "meilleur_cas", "rapport"
]

df = df[columns]

conn = sqlite3.connect("db/graphs.db")

conn.execute("""
CREATE TABLE IF NOT EXISTS performances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graphe_id INTEGER,
    heuristique TEXT,
    avg_size REAL,
    best_size REAL,
    worst_size REAL,
    avg_time REAL,
    best_time REAL,
    worst_time REAL,
    total_time REAL,
    pire_cas REAL,
    moyen_cas REAL,
    meilleur_cas REAL,
    rapport REAL
)
""")

df.to_sql("performances", conn, if_exists="append", index=False)

conn.commit()
conn.close()

print("Données de performances importées avec succès dans la base graphs.db.")
