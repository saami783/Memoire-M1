import sqlite3
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

conn = sqlite3.connect("performances.db")
df_graphs = pd.read_sql_query("SELECT * FROM graphes", conn, index_col="id")
df_best = pd.read_sql_query("""
WITH min_score AS (
  SELECT graphe_id, MIN(Rapport) AS best_rapport
  FROM performances
  GROUP BY graphe_id
)
SELECT p.graphe_id, p.heuristique AS best_heuristic
FROM performances p
JOIN min_score m
  ON p.graphe_id = m.graphe_id
 AND p.Rapport   = m.best_rapport
""", conn)
conn.close()

df = df_best.merge(df_graphs, left_on="graphe_id", right_index=True, how="left")

X = df.select_dtypes(include=["number"]).drop(
    columns=["id", "graphe_id", "cover_size", "instance_number"], errors="ignore"
)
y = df["best_heuristic"].astype(str)

X = X.fillna(X.median())

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("=== Évaluation sur le set de test ===")
print("Accuracy :", accuracy_score(y_test, y_pred))

labels_present = sorted(set(y_test))
target_names = le.inverse_transform(labels_present)

print(classification_report(
    y_test,
    y_pred,
    labels=labels_present,
    target_names=target_names
))

