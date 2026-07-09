# Scoring MLOps — Prédiction du risque de défaut de paiement

Projet d'examen **MLOps & Déploiement** — Master 2 Informatique de Gestion, UCAO Saint-Michel de Dakar.

**Auteur : Ben Hatub Hamdi Abdillah** (matricule 1068501) · Enseignant : M. Aidara Chamsedine, Tech Lead Data & IA

| Livrable | Lien |
|---|---|
|  Application en ligne | https://scoring-mlops-ben.onrender.com |
|  Dépôt GitHub (code + CI/CD) | https://github.com/benhatub/scoring-mlops |
|  Documentation API (Swagger) | https://scoring-mlops-ben.onrender.com/docs |

---

## Le projet en une phrase

Un système complet de **scoring crédit** : à partir des données d'un client, le modèle prédit sa probabilité de défaut de paiement à 2 ans et décide d'accorder ou de refuser le crédit selon un **seuil optimisé sur les coûts métier** — le tout industrialisé (MLflow, API cloud, CI/CD, monitoring de drift).

**Dataset** : [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle) — 150 000 clients, 6,68 % de défauts (classes déséquilibrées).

---

## Architecture

```
Données Kaggle ──► data_prep.py ──► MLflow (tracking + registry) ──► modèle @champion
                                                                          │
        ┌─────────────────────────────────────────────────────────────────┘
        ▼
   API FastAPI ──► GitHub ──► GitHub Actions (tests) ──► Render (cloud) ──► Interface web
        ▲                                                                        │
        └──────────── réentraînement ◄──── alerte ◄──── drift_monitor.py ◄──────┘
```

Chaque `git push` déclenche les tests automatisés ; **le déploiement n'a lieu que si les tests passent** (Auto-Deploy Render désactivé, déploiement piloté par webhook).

---

## Résultats clés

| Modèle | ROC-AUC | Coût métier ↓ | Score métier ↑ | Seuil optimal |
|---|---|---|---|---|
| Baseline (Dummy) | 0,500 | 0,668 | 0,000 | — |
| Régression Logistique | 0,860 | 0,346 | 0,482 | 0,52 |
| LogReg + SMOTE | 0,860 | 0,346 | 0,482 | 0,58 |
| **Random Forest** | 0,867 | **0,327** | **0,511** | **0,56** |
| XGBoost | **0,870** | 0,327 | 0,510 | 0,54 |

 **Le message clé** : XGBoost a le meilleur ROC-AUC, mais Random Forest a le coût métier le plus bas (faux négatif = 10, faux positif = 1) → c'est lui le champion. *La meilleure métrique ML n'est pas toujours le meilleur modèle métier.*

**Data drift** (scénario « crise économique » simulé) : PSI MonthlyIncome = 1,405 → ALERTE détectée ; scénario sans dérive : 0 fausse alerte.

---

## Structure du projet

```
scoring-mlops/
│── data/
│   ├── raw/cs-training.csv        # données brutes Kaggle
│   └── processed/                 # données nettoyées + splits train/test
│── src/
│   ├── data_prep.py               # Étape 2 : nettoyage + feature engineering + split
│   ├── business_score.py          # Étape 3 : coûts FP/FN + seuil optimal
│   ├── train.py                   # Étape 4 : 5 modèles, GridSearch, MLflow, SHAP
│   ├── register_best.py           # enregistrement du champion (registry @champion)
│   ├── export_model.py            # export du champion vers models/model.pkl
│   ├── drift_monitor.py           # Étape 7 : PSI + KS, scénarios avec/sans drift
│   └── test_mlflow.py             # Étape 1 : test de journalisation MLflow
│── api/main.py                    # Étape 5 : API FastAPI (/predict, /health, /)
│── static/index.html              # Étape 6 : interface web (jauge, seuil, facteurs)
│── tests/test_api.py              # 4 tests automatisés (exécutés par la CI)
│── models/                        # model.pkl + threshold.txt (seuil métier 0,56)
│── .github/workflows/ci-cd.yml    # pipeline : test → deploy (webhook Render)
│── requirements.txt
│── .python-version                # Python 3.13.2 épinglé (cohérence dev/prod)
```

---

## Installation et exécution locale

Prérequis : Python 3.13, Git.

```bash
# 1. Cloner et installer
git clone https://github.com/benhatub/scoring-mlops.git
cd scoring-mlops
python -m venv .venv
.venv\Scripts\activate          
pip install -r requirements.txt

# 2. Lancer le serveur MLflow (terminal 1)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

# 3. Pipeline complet (terminal 2)
python src/data_prep.py          # nettoyage + features + split
python src/train.py              # entraînement des 5 modèles (~15 min)
python src/register_best.py     # champion → Model Registry @champion
python src/export_model.py      # export vers models/model.pkl
python src/drift_monitor.py     # analyse du data drift (2 scénarios)

# 4. Lancer l'API + interface
uvicorn api.main:app --reload --port 8000
# → interface : http://127.0.0.1:8000  ·  Swagger : http://127.0.0.1:8000/docs

# 5. Tests
pip install pytest httpx
python -m pytest tests -v
```

> Le zip de rendu contient déjà `data/processed/` et `models/model.pkl` : l'API (étape 4) fonctionne immédiatement, sans rejouer l'entraînement.

---

## Choix techniques défendus

- **Score métier plutôt qu'accuracy** : coût FN = 10 (perte du capital) vs FP = 1 (perte de marge) ; le seuil de décision (0,56) minimise ce coût au lieu du 0,50 naïf.
- **Backend MLflow SQLite** : suffisant en mono-utilisateur ; passage à PostgreSQL en changeant une seule URI en entreprise.
- **Déséquilibre des classes** : `class_weight=balanced`, SMOTE et `scale_pos_weight` comparés.
- **Explicabilité** : feature importance (RF) + SHAP (XGBoost) — exigence réglementaire en finance ; l'API renvoie des facteurs explicatifs pour chaque client.
- **CI/CD stricte** : la pipeline a réellement détecté un problème de portabilité (`pywin32`, dépendance Windows) avant tout déploiement.
- **Cloud Render** (plan gratuit) : le service s'endort après 15 min d'inactivité — la première requête peut prendre ~50 s.

---

## Perspectives

PostgreSQL pour MLflow multi-utilisateurs · SHAP temps réel par prédiction · drift monitoring planifié (cron / GitHub Actions) · réentraînement automatique déclenché par alerte · authentification API.
