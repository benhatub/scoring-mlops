"""
Etape 3 : Definition du score metier
Contexte : scoring credit (defaut de paiement)

- Faux negatif (FN) : mauvais payeur accepte -> perte du capital  (cout = 10)
- Faux positif (FP) : bon client refuse     -> perte de la marge (cout = 1)

Le score metier = cout total normalise. Plus il est BAS, mieux c'est.
On cherche aussi le seuil de decision qui minimise ce cout,
au lieu du seuil naif de 0.5.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

# Couts metier (ratio 10:1, classique en credit scoring)
COST_FN = 10   # defaut non detecte : perte du capital prete
COST_FP = 1    # bon client refuse : perte de la marge d'interet


def business_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    """Cout metier total normalise par le nombre de clients.
    Plus c'est bas, mieux c'est."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (cost_fn * fn + cost_fp * fp) / len(y_true)


def business_score(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    """Score metier normalise entre 0 et 1 (1 = parfait).
    On compare le cout du modele au pire cout possible
    (celui d'un modele qui se trompe sur tout)."""
    cost = business_cost(y_true, y_pred, cost_fn, cost_fp)
    # Pire cas de reference : tout predire "bon client" (aucun defaut detecte)
    y_naive = np.zeros_like(np.asarray(y_true))
    cost_naive = business_cost(y_true, y_naive, cost_fn, cost_fp)
    return 1 - (cost / cost_naive) if cost_naive > 0 else 1.0


def find_best_threshold(y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP):
    """Cherche le seuil de probabilite qui minimise le cout metier.
    Retourne (meilleur_seuil, cout_minimal)."""
    thresholds = np.arange(0.05, 0.96, 0.01)
    costs = [
        business_cost(y_true, (y_proba >= t).astype(int), cost_fn, cost_fp)
        for t in thresholds
    ]
    idx = int(np.argmin(costs))
    return float(thresholds[idx]), float(costs[idx])


if __name__ == "__main__":
    # Test de validation avec des donnees imparfaites (chevauchement realiste)
    rng = np.random.default_rng(42)
    n = 5000
    y_true = rng.binomial(1, 0.07, size=n)  # ~7% de defauts

    # Les defauts ont une proba moyenne plus haute, mais les distributions
    # se chevauchent : le modele simule n'est pas parfait.
    y_proba = np.clip(
        rng.normal(loc=0.25 + 0.30 * y_true, scale=0.15, size=n), 0, 1
    )

    seuil, cout = find_best_threshold(y_true, y_proba)
    y_pred = (y_proba >= seuil).astype(int)
    y_pred_05 = (y_proba >= 0.5).astype(int)

    print(f"Seuil optimal      : {seuil:.2f}")
    print(f"Cout metier        : {cout:.4f}")
    print(f"Score metier (0-1) : {business_score(y_true, y_pred):.4f}")
    print(f"--- Comparaison avec le seuil naif 0.5 ---")
    print(f"Cout au seuil 0.5  : {business_cost(y_true, y_pred_05):.4f}")
    print(f"Score au seuil 0.5 : {business_score(y_true, y_pred_05):.4f}")