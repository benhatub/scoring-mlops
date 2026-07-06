"""
Etape 2 : Preparation et traitement des donnees
Dataset : Give Me Some Credit (Kaggle)
Objectif : nettoyer les donnees, creer de nouvelles features,
           et produire les ensembles train/test.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Chemins
RAW_PATH = Path("data/raw/cs-training.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "SeriousDlqin2yrs"


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Charge les donnees brutes."""
    df = pd.read_csv(path, index_col=0)
    print(f"[LOAD] {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"[LOAD] Taux de defaut : {df[TARGET].mean():.2%}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage : valeurs manquantes + outliers."""
    df = df.copy()
    n_avant = len(df)

    # --- 1. Outlier : age = 0 (impossible pour un emprunteur)
    df = df[df["age"] > 0]

    # --- 2. Codes aberrants 96/98 dans les retards de paiement
    # Ces codes signifient "autre" dans la doc du dataset, pas de vrais retards.
    # On les plafonne a un maximum realiste (20).
    delay_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in delay_cols:
        df[col] = df[col].clip(upper=20)

    # --- 3. RevolvingUtilization > 10 : valeurs absurdes (le taux devrait
    # rester proche de [0, 1]). On plafonne au quantile 99%.
    cap_util = df["RevolvingUtilizationOfUnsecuredLines"].quantile(0.99)
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=cap_util)

    # --- 4. DebtRatio : meme logique, plafonnement au quantile 99%
    cap_debt = df["DebtRatio"].quantile(0.99)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=cap_debt)

    # --- 5. Valeurs manquantes
    # MonthlyIncome (~20% manquant) : imputation par la MEDIANE
    # (robuste aux revenus extremes) + indicateur de valeur manquante
    df["IncomeMissing"] = df["MonthlyIncome"].isna().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    # NumberOfDependents (~2.6% manquant) : imputation par le MODE (0)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    print(f"[CLEAN] {n_avant - len(df)} lignes supprimees, {len(df)} restantes")
    print(f"[CLEAN] Valeurs manquantes restantes : {df.isna().sum().sum()}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Creation de nouvelles variables metier."""
    df = df.copy()

    # Nombre total d'incidents de paiement
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )

    # A deja eu au moins un incident (binaire)
    df["HasPastDue"] = (df["TotalPastDue"] > 0).astype(int)

    # Revenu mensuel par personne du foyer
    df["IncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    # Dette mensuelle estimee
    df["MonthlyDebt"] = df["DebtRatio"] * df["MonthlyIncome"]

    # Tranches d'age (le risque varie fortement avec l'age)
    df["AgeGroup"] = pd.cut(
        df["age"], bins=[0, 30, 45, 60, 120], labels=[0, 1, 2, 3]
    ).astype(int)

    print(f"[FEAT] {df.shape[1]} colonnes apres feature engineering")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Separation train/test STRATIFIEE (car classes desequilibrees)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"[SPLIT] Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"[SPLIT] Taux defaut train : {y_train.mean():.2%} | test : {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def main():
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    # Sauvegarde du dataset nettoye complet
    df.to_csv(PROCESSED_DIR / "credit_clean.csv", index=False)

    # Sauvegarde des splits
    X_train, X_test, y_train, y_test = split_data(df)
    X_train.assign(**{TARGET: y_train}).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("\n[OK] Fichiers generes dans data/processed/ :")
    print("  - credit_clean.csv (dataset complet nettoye)")
    print("  - train.csv / test.csv (split stratifie 80/20)")


if __name__ == "__main__":
    main()