import sqlite3
import pandas as pd

def enrich_excel_with_original_nodes(
    excel_path="performances.xlsx",
    sheet_name="kernels_hog",
    db_path="db/graphs.db",
    output_path="performances_updated.xlsx"
):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    foreign_ids = df["Foreign_Id"].unique().tolist()
    placeholders = ",".join(["?"] * len(foreign_ids))

    query = f"""
        SELECT id, Number_of_Vertices
        FROM graphes
        WHERE id IN ({placeholders})
    """
    cursor.execute(query, foreign_ids)
    mapping = dict(cursor.fetchall())

    conn.close()

    df["Node_Graph_Original"] = df["Foreign_Id"].map(mapping)

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Fichier mis à jour avec succès : {output_path}")

if __name__ == "__main__":
    enrich_excel_with_original_nodes()
