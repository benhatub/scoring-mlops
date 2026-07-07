"""
Etape 5 : API REST de scoring credit (FastAPI)

- POST /predict : recoit les infos d'un client, renvoie proba + decision + facteurs
- GET  /health  : verification de l'etat de l'API
- GET  /        : interface web (static/index.html)
Le modele est charge depuis models/model.pkl (exporte du registry MLflow).
Le seuil de decision vient de models/threshold.txt (seuil metier optimal).
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
    version="1.1.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


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


def analyser_facteurs(c: ClientData) -> list:
    """Identifie les facteurs de risque/protection propres a ce client.
    Seuils inspires des distributions du dataset et de la feature importance."""
    facteurs = []

    d30, d60, d90 = (c.NumberOfTime30_59DaysPastDueNotWorse,
                     c.NumberOfTime60_89DaysPastDueNotWorse,
                     c.NumberOfTimes90DaysLate)
    total_retards = d30 + d60 + d90

    # --- Facteurs aggravants
    if d90 > 0:
        facteurs.append({"label": f"{d90} retard(s) grave(s) de 90+ jours",
                         "impact": "fort", "sens": "risque"})
    if total_retards >= 3:
        facteurs.append({"label": f"{total_retards} incidents de paiement au total",
                         "impact": "fort", "sens": "risque"})
    elif total_retards > 0:
        facteurs.append({"label": f"{total_retards} incident(s) de paiement",
                         "impact": "modere", "sens": "risque"})
    if c.RevolvingUtilizationOfUnsecuredLines >= 0.8:
        facteurs.append({"label": f"Credit utilise a {c.RevolvingUtilizationOfUnsecuredLines:.0%} (saturation)",
                         "impact": "fort", "sens": "risque"})
    elif c.RevolvingUtilizationOfUnsecuredLines >= 0.5:
        facteurs.append({"label": f"Credit utilise a {c.RevolvingUtilizationOfUnsecuredLines:.0%}",
                         "impact": "modere", "sens": "risque"})
    if c.DebtRatio >= 0.6:
        facteurs.append({"label": f"Endettement eleve ({c.DebtRatio:.0%} des revenus)",
                         "impact": "modere", "sens": "risque"})
    if c.MonthlyIncome < 1500:
        facteurs.append({"label": f"Revenu mensuel faible ({c.MonthlyIncome:.0f} $)",
                         "impact": "modere", "sens": "risque"})
    if c.age < 25:
        facteurs.append({"label": f"Age jeune ({c.age} ans), historique court",
                         "impact": "modere", "sens": "risque"})

    # --- Facteurs protecteurs
    if total_retards == 0:
        facteurs.append({"label": "Aucun incident de paiement",
                         "impact": "fort", "sens": "protection"})
    if c.RevolvingUtilizationOfUnsecuredLines < 0.3:
        facteurs.append({"label": f"Faible utilisation du credit ({c.RevolvingUtilizationOfUnsecuredLines:.0%})",
                         "impact": "modere", "sens": "protection"})
    if c.MonthlyIncome >= 5000:
        facteurs.append({"label": f"Revenu confortable ({c.MonthlyIncome:.0f} $)",
                         "impact": "modere", "sens": "protection"})
    if 35 <= c.age <= 65:
        facteurs.append({"label": f"Age de stabilite financiere ({c.age} ans)",
                         "impact": "modere", "sens": "protection"})
    if c.DebtRatio < 0.35:
        facteurs.append({"label": f"Endettement maitrise ({c.DebtRatio:.0%})",
                         "impact": "modere", "sens": "protection"})

    # Les risques d'abord, puis forts avant moderes, max 5 affiches
    facteurs.sort(key=lambda f: (f["sens"] != "risque", f["impact"] != "fort"))
    return facteurs[:5]


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
        "facteurs": analyser_facteurs(client),
    }


@app.get("/", include_in_schema=False)
def interface():
    return FileResponse(BASE_DIR / "static" / "index.html")