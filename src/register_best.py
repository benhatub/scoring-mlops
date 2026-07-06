"""
Reentraine le RandomForest gagnant avec ses meilleurs hyperparametres
(recuperes depuis MLflow) et l'enregistre dans le Model Registry
avec l'alias 'champion'.
"""

import mlflow
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "credit-scoring"
TARGET = "SeriousDlqin2yrs"
DATA_DIR = Path("data/processed")
MODEL_NAME = "credit-scoring-model"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
client = mlflow.MlflowClient()

# --- 1. Retrouver le run 'randomforest' et ses meilleurs params
exp = client.get_experiment_by_name(EXPERIMENT)
runs = client.search_runs(
    exp.experiment_id,
    filter_string="attributes.run_name = 'randomforest'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
rf_run = runs[0]
n_estimators = int(rf_run.data.params["best_n_estimators"])
max_depth = int(rf_run.data.params["best_max_depth"])
print(f"[PARAMS] n_estimators={n_estimators}, max_depth={max_depth}")

# --- 2. Reentrainer sur le train complet
train = pd.read_csv(DATA_DIR / "train.csv")
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("[TRAIN] RandomForest reentraine")

# --- 3. Logger + enregistrer dans le Model Registry
with mlflow.start_run(run_name="champion-randomforest"):
    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": "balanced",
    })
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=MODEL_NAME,
        input_example=X_train.head(3),
    )

# --- 4. Alias 'champion' sur la derniere version
versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
latest = max(int(v.version) for v in versions)
client.set_registered_model_alias(MODEL_NAME, "champion", str(latest))
print(f"[REGISTRY] {MODEL_NAME} v{latest} -> alias 'champion' OK")