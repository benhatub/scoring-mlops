"""Tests automatises de l'API (executes par la CI GitHub Actions)."""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

BON_CLIENT = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30_59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}

CLIENT_RISQUE = {**BON_CLIENT,
                 "RevolvingUtilizationOfUnsecuredLines": 0.98,
                 "MonthlyIncome": 800,
                 "NumberOfTimes90DaysLate": 5}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_bon_client():
    r = client.post("/predict", json=BON_CLIENT)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probabilite_defaut"] <= 1
    assert body["decision"] == "ACCEPTE"


def test_predict_client_risque():
    r = client.post("/predict", json=CLIENT_RISQUE)
    assert r.status_code == 200
    assert r.json()["decision"] == "REFUSE"


def test_donnees_invalides():
    r = client.post("/predict", json={**BON_CLIENT, "age": -5})
    assert r.status_code == 422  # rejete par la validation Pydantic