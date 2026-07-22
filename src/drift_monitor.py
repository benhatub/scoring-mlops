"""
Etape 7 : Analyse du data drift
Indicateurs :
- PSI (Population Stability Index) : standard du credit scoring
    PSI < 0.1  -> stable
    0.1 - 0.2  -> drift modere (surveillance)
    PSI > 0.2  -> drift significatif (ALERTE)
- Test de Kolmogorov-Smirnov : p-value < 0.05 -> distributions differentes

Strategie :
1. Reference = donnees d'entrainement
2. Production = nouvelles donnees recues par l'API (ici : simulees)
3. Si drift significatif -> alerte -> investigation -> reentrainement
   via la pipeline existante (train.py -> registry -> CI/CD -> deploiement)
"""

import numpy as np
import pandas as pd
import mlflow
import json
from pathlib import Path
from scipy.stats import ks_2samp

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "drift-monitoring"
TARGET = "SeriousDlqin2yrs"
DATA_DIR = Path("data/processed")

FEATURES_A_SURVEILLER = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
    "MonthlyIncome", "TotalPastDue", "NumberOfOpenCreditLinesAndLoans",
]


def calcul_psi(reference: pd.Series, production: pd.Series, bins: int = 10) -> float:
    """Population Stability Index entre deux distributions."""
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # variable quasi constante
        return 0.0
    ref_counts = np.histogram(reference.dropna(), bins=edges)[0] / len(reference.dropna())
    prod_counts = np.histogram(production.dropna(), bins=edges)[0] / len(production.dropna())
    ref_counts = np.clip(ref_counts, 1e-6, None)
    prod_counts = np.clip(prod_counts, 1e-6, None)
    return float(np.sum((prod_counts - ref_counts) * np.log(prod_counts / ref_counts)))


def simuler_production(df: pd.DataFrame, drift: bool = True, seed: int = 42) -> pd.DataFrame:
    """Simule des donnees de production.
    Si drift=True : crise economique simulee (revenus -20%, endettement +30%,
    plus de retards de paiement)."""
    rng = np.random.default_rng(seed)
    prod = df.sample(n=min(20000, len(df)), random_state=seed).copy()
    if drift:
        prod["MonthlyIncome"] *= 0.80
        prod["DebtRatio"] *= 1.30
        prod["RevolvingUtilizationOfUnsecuredLines"] = np.clip(
            prod["RevolvingUtilizationOfUnsecuredLines"] * 1.25, 0, None)
        surplus = rng.binomial(1, 0.15, len(prod))
        prod["TotalPastDue"] = prod["TotalPastDue"] + surplus
    return prod


def analyser_drift(reference: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    """Calcule PSI et KS pour chaque feature surveillee."""
    lignes = []
    for feat in FEATURES_A_SURVEILLER:
        psi = calcul_psi(reference[feat], production[feat])
        ks_stat, ks_pval = ks_2samp(reference[feat].dropna(), production[feat].dropna())
        statut = "OK" if psi < 0.1 else ("SURVEILLANCE" if psi < 0.2 else "ALERTE")
        lignes.append({"feature": feat, "psi": round(psi, 4),
                       "ks_pvalue": round(float(ks_pval), 6), "statut": statut})
    return pd.DataFrame(lignes).sort_values("psi", ascending=False)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    reference = pd.read_csv(DATA_DIR / "train.csv")
    export = {"genere_le": pd.Timestamp.now().strftime("%d/%m/%Y"),
              "seuils": {"ok": 0.1, "surveillance": 0.2},
              "scenarios": []}

    scenarios = [
        ("sans-drift", False, "Donnees de production conformes a la reference"),
        ("avec-drift", True, "Crise economique simulee : revenus -20%, endettement +30%"),
    ]

    for nom, drift, libelle in scenarios:
        production = simuler_production(reference, drift=drift)
        rapport = analyser_drift(reference, production)
        n_alertes = int((rapport["statut"] == "ALERTE").sum())

        print(f"\n===== Scenario : {nom} =====")
        print(rapport.to_string(index=False))
        print(f"Alertes : {n_alertes}")
        if n_alertes > 0:
            print(">>> DRIFT SIGNIFICATIF DETECTE : reentrainement recommande !")

        export["scenarios"].append({
            "nom": nom,
            "libelle": libelle,
            "n_production": int(len(production)),
            "nb_alertes": n_alertes,
            "features": rapport.to_dict(orient="records"),
        })

        with mlflow.start_run(run_name=f"drift-{nom}"):
            mlflow.log_param("scenario", nom)
            mlflow.log_param("n_production", len(production))
            for _, r in rapport.iterrows():
                mlflow.log_metric(f"psi_{r['feature']}", r["psi"])
            mlflow.log_metric("nb_alertes", n_alertes)
            rapport.to_csv("drift_report.csv", index=False)
            mlflow.log_artifact("drift_report.csv")

    # Export pour l'interface web (lisible sans MLflow ni donnees brutes)
    out_dir = Path("static/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "drift_report.json", "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print("\n[OK] Rapports logges dans MLflow (experience 'drift-monitoring')")
    print("[OK] Export interface : static/data/drift_report.json")


if __name__ == "__main__":
    main()