import sqlite3

conn = sqlite3.connect('../../performances.db')
cursor = conn.cursor()

cursor.execute("SELECT rowid, graph_name, class FROM graphes")
rows = cursor.fetchall()

# Échange propre des valeurs graph_name <-> class
for rowid, graph_name, class_name in rows:
    cursor.execute(
        "UPDATE graphes SET graph_name = ?, class = ? WHERE rowid = ?",
        (class_name, graph_name, rowid)
    )

# Remplace toutes les valeurs de "class" commençant par "frb" par "bhoslib"
cursor.execute("""
    UPDATE graphes
    SET class = 'bhoslib'
    WHERE class LIKE 'frb%';
""")

conn.commit()
conn.close()
