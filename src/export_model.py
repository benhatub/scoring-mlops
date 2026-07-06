"""Exporte le modele champion du registry vers models/model.pkl
(pour que l'API fonctionne sans serveur MLflow, ex. dans le cloud)."""

import mlflow
import joblib
from pathlib import Path

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.sklearn.load_model("models:/credit-scoring-model@champion")
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("[OK] Modele champion exporte vers models/model.pkl")