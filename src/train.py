"""
Etape 4 : Entrainement et comparaison de modeles
- Baseline + LogisticRegression + RandomForest + XGBoost
- GridSearchCV, gestion du desequilibre, seuil metier optimal
- Tracking MLflow complet + Model Registry
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # pas d'affichage, on sauvegarde les figures
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from business_score import business_cost, business_score, find_best_threshold

# ---------------------------------------------------------------- config
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "credit-scoring"
TARGET = "SeriousDlqin2yrs"
DATA_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)


# ---------------------------------------------------------------- data
def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------- eval
def evaluate_and_log(model, name, X_test, y_test):
    """Calcule toutes les metriques + seuil metier et logge dans MLflow."""
    y_proba = model.predict_proba(X_test)[:, 1]

    # Seuil metier optimal
    best_t, best_cost = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_t).astype(int)
    y_pred_05 = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "best_threshold": best_t,
        "business_cost": best_cost,
        "business_score": business_score(y_test, y_pred),
        "business_cost_at_05": business_cost(y_test, y_pred_05),
    }
    mlflow.log_metrics(metrics)

    # Courbe ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"ROC - {name}")
    roc_path = FIG_DIR / f"roc_{name}.png"
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(roc_path))

    # Matrice de confusion (au seuil metier)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
    ax.set_title(f"Confusion (seuil={best_t:.2f}) - {name}")
    cm_path = FIG_DIR / f"cm_{name}.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))

    print(f"  ROC-AUC={metrics['roc_auc']:.4f} | cout={best_cost:.4f} "
          f"| score metier={metrics['business_score']:.4f} | seuil={best_t:.2f}")
    return metrics


def log_feature_importance(model, name, X_test):
    """Importance des features (SHAP pour XGBoost, sinon native/coefs)."""
    try:
        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        clf = estimator.named_steps.get("clf", estimator) if hasattr(estimator, "named_steps") else estimator

        if isinstance(clf, XGBClassifier):
            import shap
            sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(sample)
            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, sample, show=False)
            path = FIG_DIR / f"shap_{name}.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(path))
            print("  [SHAP] summary plot logge")
        elif hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=X_test.columns).sort_values()
            fig, ax = plt.subplots(figsize=(7, 6))
            imp.plot.barh(ax=ax)
            ax.set_title(f"Feature importance - {name}")
            path = FIG_DIR / f"importance_{name}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(path))
        elif hasattr(clf, "coef_"):
            coefs = pd.Series(clf.coef_[0], index=X_test.columns).sort_values()
            fig, ax = plt.subplots(figsize=(7, 6))
            coefs.plot.barh(ax=ax)
            ax.set_title(f"Coefficients - {name}")
            path = FIG_DIR / f"coefs_{name}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(path))
    except Exception as e:
        print(f"  [WARN] importance non calculee : {e}")


# ---------------------------------------------------------------- main
def main():
    X_train, X_test, y_train, y_test = load_splits()
    ratio = (y_train == 0).sum() / (y_train == 1).sum()  # pour XGBoost
    results = {}

    # ---------- 1. BASELINE ----------
    print("\n[1/5] Baseline (classe majoritaire)")
    with mlflow.start_run(run_name="baseline-dummy"):
        mlflow.log_param("model", "DummyClassifier")
        dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
        results["baseline"] = evaluate_and_log(dummy, "baseline", X_test, y_test)

    # ---------- 2. LOGISTIC REGRESSION (class_weight) ----------
    print("\n[2/5] LogisticRegression + GridSearch")
    with mlflow.start_run(run_name="logreg-classweight"):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ])
        grid = GridSearchCV(
            pipe, {"clf__C": [0.01, 0.1, 1.0]},
            scoring="roc_auc", cv=3, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("imbalance", "class_weight=balanced")
        mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
        results["logreg"] = evaluate_and_log(grid, "logreg", X_test, y_test)
        log_feature_importance(grid, "logreg", X_test)

    # ---------- 3. LOGISTIC REGRESSION + SMOTE (comparaison) ----------
    print("\n[3/5] LogisticRegression + SMOTE")
    with mlflow.start_run(run_name="logreg-smote"):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", LogisticRegression(max_iter=2000)),
        ])
        pipe.fit(X_train, y_train)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("imbalance", "SMOTE")
        results["logreg_smote"] = evaluate_and_log(pipe, "logreg_smote", X_test, y_test)

    # ---------- 4. RANDOM FOREST ----------
    print("\n[4/5] RandomForest + GridSearch")
    with mlflow.start_run(run_name="randomforest"):
        rf = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        grid = GridSearchCV(
            rf,
            {"n_estimators": [100, 200], "max_depth": [8, 12]},
            scoring="roc_auc", cv=3, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("imbalance", "class_weight=balanced")
        mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
        results["rf"] = evaluate_and_log(grid, "rf", X_test, y_test)
        log_feature_importance(grid, "rf", X_test)

    # ---------- 5. XGBOOST ----------
    print("\n[5/5] XGBoost + GridSearch")
    with mlflow.start_run(run_name="xgboost") as run:
        xgb = XGBClassifier(
            scale_pos_weight=ratio, eval_metric="auc",
            random_state=42, n_jobs=-1,
        )
        grid = GridSearchCV(
            xgb,
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.1],
            },
            scoring="roc_auc", cv=3, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("imbalance", f"scale_pos_weight={ratio:.1f}")
        mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
        results["xgb"] = evaluate_and_log(grid, "xgb", X_test, y_test)
        log_feature_importance(grid, "xgb", X_test)
        xgb_run_id = run.info.run_id
        best_xgb = grid.best_estimator_

    # ---------- Comparaison finale ----------
    print("\n========== COMPARAISON ==========")
    df_res = pd.DataFrame(results).T[
        ["roc_auc", "business_cost", "business_score", "best_threshold"]
    ]
    print(df_res.round(4))

    # ---------- Enregistrement du champion dans le Model Registry ----------
    best_name = df_res["business_cost"].idxmin()
    print(f"\n[REGISTRY] Meilleur modele (cout metier minimal) : {best_name}")

    if best_name == "xgb":
        with mlflow.start_run(run_id=xgb_run_id):
            mlflow.sklearn.log_model(
                best_xgb,
                name="model",
                registered_model_name="credit-scoring-model",
                input_example=X_test.head(3),
            )
        # Alias champion sur la derniere version
        client = mlflow.MlflowClient()
        latest = client.get_registered_model("credit-scoring-model").latest_versions
        version = max(int(v.version) for v in latest) if latest else 1
        client.set_registered_model_alias("credit-scoring-model", "champion", str(version))
        print(f"[REGISTRY] credit-scoring-model v{version} -> alias 'champion'")
    else:
        print("[REGISTRY] Adapter le bloc registry au modele gagnant si ce n'est pas XGBoost.")

    # Sauvegarde locale du seuil optimal pour l'API (Etape 5)
    best_t = results[best_name]["best_threshold"]
    Path("models").mkdir(exist_ok=True)
    with open("models/threshold.txt", "w") as f:
        f.write(str(best_t))
    print(f"[OK] Seuil metier {best_t:.2f} sauvegarde dans models/threshold.txt")


if __name__ == "__main__":
    main()