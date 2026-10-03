import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

rng = np.random.default_rng(42)
n = 5000

tenure_months = rng.integers(0, 73, n)
monthly_charges = rng.uniform(20, 120, n)
contract = rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])  # 0=month-to-month,1=1yr,2=2yr
support_calls = rng.poisson(1.5, n)
avg_monthly_usage_gb = rng.gamma(shape=2.0, scale=15, size=n)
total_charges = tenure_months * monthly_charges * rng.uniform(0.9, 1.1, n)

# Genuine signal: higher risk with short tenure, month-to-month contract, high support calls,
# low usage; some noise so it's not trivially separable.
logit = (
    -1.8
    - 0.05 * tenure_months
    + 0.015 * monthly_charges
    - 1.1 * (contract == 2)
    - 0.5 * (contract == 1)
    + 0.35 * support_calls
    - 0.01 * avg_monthly_usage_gb
    + rng.normal(0, 1.0, n)  # noise
)
prob = 1 / (1 + np.exp(-logit))
churned = (rng.uniform(0, 1, n) < prob).astype(int)

X = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "contract_duration": contract,
    "support_calls": support_calls,
    "avg_monthly_usage_gb": avg_monthly_usage_gb,
    "total_charges": total_charges,
})
y = churned
print(f"n={n}, positive rate={y.mean():.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "extra_trees": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
    "lightgbm": LGBMClassifier(random_state=42, verbosity=-1),
    "xgboost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
}

results = []
for name, model in candidates.items():
    t0 = time.time()
    scores = cross_validate(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    elapsed = time.time() - t0
    mean_auc = scores["test_score"].mean()
    std_auc = scores["test_score"].std()
    rank_score = mean_auc - 0.25 * std_auc
    results.append((name, mean_auc, std_auc, elapsed, rank_score))

results.sort(key=lambda r: (-r[4], r[3]))
print(f"\n{'model':24s} {'mean_auc':>10s} {'std':>8s} {'elapsed_s':>10s} {'rank_score':>12s}")
for name, mean_auc, std_auc, elapsed, rank_score in results:
    print(f"{name:24s} {mean_auc:10.4f} {std_auc:8.4f} {elapsed:10.3f} {rank_score:12.4f}")

print(f"\nSelected (rank by mean - 0.25*std, then runtime): {results[0][0]}")
