import sqlite3
import glob

# Liste de tes fichiers SQLite à fusionner
db_files = ['bhoslib.db', 'ba.db', 'er.db', 'ruglar.db', 'tree.db']

# Connexion à la base fusionnée
conn_fusion = sqlite3.connect('fusion.db')
cursor_fusion = conn_fusion.cursor()

# On copie la structure de la table depuis la première base
with sqlite3.connect(db_files[0]) as conn_source:
    cursor_source = conn_source.cursor()
    # Récupère la définition de la table
    cursor_source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='graphes'")
    table_create_sql = cursor_source.fetchone()[0]
    cursor_fusion.execute(table_create_sql)
    conn_fusion.commit()

# Pour chaque base source
for db_file in db_files:
    with sqlite3.connect(db_file) as conn_source:
        cursor_source = conn_source.cursor()
        # Récupère les données sans l'id
        cursor_source.execute("SELECT * FROM graphes")
        rows = cursor_source.fetchall()
        # Récupère les noms de colonnes
        col_names = [description[0] for description in cursor_source.description]

        # On enlève 'id' si présent
        if 'id' in col_names:
            id_index = col_names.index('id')
            col_names.pop(id_index)
            rows = [tuple(v for i, v in enumerate(row) if i != id_index) for row in rows]

        # Prépare l'insertion
        placeholders = ", ".join(["?"] * len(col_names))
        insert_sql = f"INSERT INTO graphes ({', '.join(col_names)}) VALUES ({placeholders})"

        cursor_fusion.executemany(insert_sql, rows)
        conn_fusion.commit()

conn_fusion.close()
print("Fusion terminée.")
