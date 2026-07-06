"""
Etape 5 : API REST de scoring credit (FastAPI)

- POST /predict : recoit les infos d'un client, renvoie proba + decision
- GET  /health  : verification de l'etat de l'API
Le modele est charge depuis models/model.pkl (exporte du registry MLflow).
Le seuil de decision vient de models/threshold.txt (seuil metier optimal).
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------- chargement
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "model.pkl")
THRESHOLD = float((BASE_DIR / "models" / "threshold.txt").read_text().strip())

app = FastAPI(
    title="API Credit Scoring",
    description="Prediction du risque de defaut de paiement (Give Me Some Credit)",
    version="1.0.0",
)


# ---------------------------------------------------------------- schema d'entree
class ClientData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, example=0.5)
    age: int = Field(..., gt=0, le=120, example=45)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, example=0)
    DebtRatio: float = Field(..., ge=0, example=0.35)
    MonthlyIncome: float = Field(..., ge=0, example=5000)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, example=8)
    NumberOfTimes90DaysLate: int = Field(..., ge=0, example=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, example=1)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, example=0)
    NumberOfDependents: int = Field(..., ge=0, example=2)


def build_features(c: ClientData) -> pd.DataFrame:
    """Reconstruit EXACTEMENT les memes features que data_prep.py."""
    d30 = min(c.NumberOfTime30_59DaysPastDueNotWorse, 20)
    d60 = min(c.NumberOfTime60_89DaysPastDueNotWorse, 20)
    d90 = min(c.NumberOfTimes90DaysLate, 20)
    total_past_due = d30 + d60 + d90
    age_group = 0 if c.age <= 30 else 1 if c.age <= 45 else 2 if c.age <= 60 else 3

    row = {
        "RevolvingUtilizationOfUnsecuredLines": c.RevolvingUtilizationOfUnsecuredLines,
        "age": c.age,
        "NumberOfTime30-59DaysPastDueNotWorse": d30,
        "DebtRatio": c.DebtRatio,
        "MonthlyIncome": c.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": c.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": d90,
        "NumberRealEstateLoansOrLines": c.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": d60,
        "NumberOfDependents": c.NumberOfDependents,
        "IncomeMissing": 0,
        "TotalPastDue": total_past_due,
        "HasPastDue": int(total_past_due > 0),
        "IncomePerPerson": c.MonthlyIncome / (c.NumberOfDependents + 1),
        "MonthlyDebt": c.DebtRatio * c.MonthlyIncome,
        "AgeGroup": age_group,
    }
    df = pd.DataFrame([row])
    return df[model.feature_names_in_]  # ordre exact des colonnes du training


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model": "credit-scoring-model@champion",
            "threshold": round(THRESHOLD, 2)}


@app.post("/predict")
def predict(client: ClientData):
    X = build_features(client)
    proba = float(model.predict_proba(X)[0, 1])
    decision = "REFUSE" if proba >= THRESHOLD else "ACCEPTE"
    return {
        "probabilite_defaut": round(proba, 4),
        "seuil_metier": round(THRESHOLD, 2),
        "decision": decision,
        "message": (
            "Risque eleve de defaut : credit refuse."
            if decision == "REFUSE"
            else "Risque faible : credit accorde."
        ),
    }