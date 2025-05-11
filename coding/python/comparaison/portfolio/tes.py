import sqlite3
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def compter_heuristiques_gagnantes(db_path="performances.db"):
    conn = sqlite3.connect(db_path)

    # Pour chaque graphe, on sélectionne l'heuristique au plus petit rapport
    query = """
        WITH ranked AS (
            SELECT graphe_id, heuristique, rapport,
                   ROW_NUMBER() OVER (
                       PARTITION BY graphe_id
                       ORDER BY rapport ASC, heuristique
                   ) AS rn
            FROM performances
        )
        SELECT graphe_id, heuristique
        FROM ranked
        WHERE rn = 1
    """
    df_winners = pd.read_sql_query(query, conn)
    conn.close()

    # Compter les heuristiques gagnantes
    counts = Counter(df_winners['heuristique'])
    df_counts = pd.DataFrame(counts.items(), columns=["Heuristique", "Nombre de victoires"])
    df_counts = df_counts.sort_values(by="Nombre de victoires", ascending=False)

    # Affichage
    print(df_counts.to_string(index=False))

    # Visualisation
    sns.barplot(x="Heuristique", y="Nombre de victoires", data=df_counts)
    plt.xticks(rotation=45)
    plt.title("Nombre de victoires par heuristique (rapport minimal)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compter_heuristiques_gagnantes()
