"""Generate part1-style classification walkthrough notebooks for each external project.

Style mirrors `part1_hyper_ack_classification.ipynb`: short markdown section headers +
executable code for EDA, cleaning, visualization, FE, safe/unsafe baselines, and
optimized models.
"""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

from general_pipeline.external_catalog import DATASET_CATALOG
from general_pipeline.external_project import project_dir
from general_pipeline.playbook.policy import get_policy

_FIND_ROOT = '''
from pathlib import Path
import sys

def _find_repo_root() -> Path:
    starts = [Path.cwd().resolve()]
    try:
        starts.append(Path(__file__).resolve().parent)  # type: ignore[name-defined]
    except NameError:
        pass
    for start in starts:
        for p in [start, *start.parents]:
            if (p / "general_pipeline").is_dir() and (p / "external_data").is_dir():
                return p
    raise RuntimeError("Could not find R&D repo root (needs general_pipeline/ and external_data/).")

ROOT = _find_repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hyperack_exp"))
print("REPO ROOT:", ROOT)
'''.strip()


def _md(text: str):
    return nbf.v4.new_markdown_cell(text)


def _code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n") + "\n")


def build_walkthrough(key: str) -> Path:
    spec = next(s for s in DATASET_CATALOG if s.key == key)
    policy = get_policy(key)
    leak_cols = policy.unsafe_only_features
    leak_txt = ", ".join(f"`{c}`" for c in leak_cols) if leak_cols else "_none_"
    cells = []

    # ---- Title ----
    cells.append(_md(f"""# {spec.name} — Classification Walkthrough

**Goal:** predict the binary target for **{spec.name}** using a HyperAck-style protocol.

| Item | Detail |
|---|---|
| Source | [{spec.url}]({spec.url}) |
| Description | {spec.description} |
| Leakage / unsafe-only features | {leak_txt} |
| Has classic leakage | **{policy.has_leakage}** |
| Policy | {policy.rationale} |

This notebook follows the same teaching style as `part1_hyper_ack_classification.ipynb`:

1. Load & explore data  
2. Clean  
3. Visualize  
4. Define **safe** vs **unsafe** feature sets  
5. Feature engineering ladder  
6. Train baseline models (safe & unsafe)  
7. Train **optimized** models  
8. Compare which approach wins  
"""))

    # ---- Import libraries ----
    cells.append(_md("## Import libraries"))
    cells.append(_code(f"""
{_FIND_ROOT}

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from general_pipeline.playbook.policy import get_policy
from general_pipeline.playbook.features import build_feature_matrix
from general_pipeline.playbook.ladder import results_frame
from shared.protocol import evaluate

sns.set_theme(style="whitegrid")
KEY = "{key}"
RANDOM_STATE = 42
policy = get_policy(KEY)
print(policy)
"""))

    # ---- Load ----
    cells.append(_md("## Load Dataset\n\nCached UCI parquet from `external_data/` (same rows used by the playbook ladder)."))
    cells.append(_code("""
DATA = ROOT / "external_data" / KEY
X_raw = pd.read_parquet(DATA / "X.parquet")
y_raw = pd.read_parquet(DATA / "y.parquet")["target"].astype(int)
meta = pd.read_json(DATA / "meta.json", typ="series")

df = X_raw.copy()
df["target"] = y_raw.values
print("shape:", df.shape)
print("pos_rate:", float(df["target"].mean()))
meta
"""))

    # ---- EDA ----
    cells.append(_md("## Exploratory Data Analysis (EDA)"))
    cells.append(_code("df.head()"))
    cells.append(_code("df.info()"))
    cells.append(_code("df.describe().T"))
    cells.append(_code("""
print("Missing values per column:")
print(df.isnull().sum().sort_values(ascending=False).head(20))
print("\\nTarget value counts:")
print(df["target"].value_counts(normalize=True))
"""))

    # ---- Cleaning ----
    cells.append(_md("""## Data Cleaning

Handling missing values and ensuring all predictors are numeric for tree / linear models.
Categorical fields were factorized at download time; we still coerce dtypes defensively.
"""))
    cells.append(_code("""
# Coerce features to numeric; keep target as int
feature_cols = [c for c in df.columns if c != "target"]
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("NaNs before median fill:", int(df[feature_cols].isna().sum().sum()))
# Median fill is done inside model pipelines (train-only). For EDA plots we fill temporarily.
df_eda = df.copy()
df_eda[feature_cols] = df_eda[feature_cols].fillna(df_eda[feature_cols].median())
print("Ready for visualization. Rows:", len(df_eda))
"""))

    # ---- Visualization ----
    cells.append(_md("## Data Visualization"))
    cells.append(_code("""
plt.figure(figsize=(6, 4))
sns.countplot(x=df_eda["target"])
plt.title(f"{KEY}: Target class balance")
plt.xlabel("target")
plt.ylabel("count")
plt.show()
"""))
    cells.append(_code("""
# Distribution of a few top-variance features by class
variances = df_eda[feature_cols].var().sort_values(ascending=False)
top_feats = variances.index[:4].tolist()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()
for ax, col in zip(axes, top_feats):
    sns.kdeplot(data=df_eda, x=col, hue="target", fill=True, common_norm=False, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()
"""))
    cells.append(_code("""
plt.figure(figsize=(10, 8))
corr = df_eda[feature_cols + ["target"]].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title(f"{KEY}: Feature correlation heatmap")
plt.show()
print("Top correlations with target:")
print(corr["target"].drop("target").abs().sort_values(ascending=False).head(10))
"""))

    # ---- Safe vs Unsafe ----
    cells.append(_md(f"""## Safe vs Unsafe feature sets

Same idea as HyperAck final fares:

- **Unsafe:** may include post-outcome / contested columns → inflated lab scores  
- **Safe:** drops those columns → honest decision-time model  

For this dataset, unsafe-only columns = **{leak_txt}**.

{policy.rationale}
"""))
    cells.append(_code("""
LEAK_COLS = list(policy.unsafe_only_features)
print("Leakage / unsafe-only columns:", LEAK_COLS)

def make_xy(frame: pd.DataFrame, mode: str):
    y = frame["target"].astype(int)
    X = frame.drop(columns=["target"])
    if mode == "safe" and LEAK_COLS:
        X = X.drop(columns=[c for c in LEAK_COLS if c in X.columns], errors="ignore")
    return X, y

X_safe, y = make_xy(df, "safe")
X_unsafe, _ = make_xy(df, "unsafe")
print("Safe feature count:  ", X_safe.shape[1], "cols =", list(X_safe.columns)[:12], "...")
print("Unsafe feature count:", X_unsafe.shape[1])
print("Dropped for safe:    ", sorted(set(X_unsafe.columns) - set(X_safe.columns)))
"""))

    # ---- Split ----
    cells.append(_md("""## Locked train / test split

Stratified 80/20, `random_state=42` — identical protocol to HyperAck / playbook ladder.
Large datasets are capped at 20,000 rows (stratified) for notebook runtime.
"""))
    cells.append(_code("""
MAX_ROWS = 20000
X_s, y_s = X_safe.copy(), y.copy()
X_u = X_unsafe.copy()

if len(X_s) > MAX_ROWS:
    X_s, _, y_s, _ = train_test_split(X_s, y_s, train_size=MAX_ROWS, stratify=y_s, random_state=RANDOM_STATE)
    X_u = X_u.loc[X_s.index]

Xtr_s, Xte_s, ytr, yte = train_test_split(
    X_s.reset_index(drop=True), y_s.reset_index(drop=True),
    test_size=0.2, stratify=y_s, random_state=RANDOM_STATE,
)
Xtr_u, Xte_u, _, _ = train_test_split(
    X_u.reset_index(drop=True), y_s.reset_index(drop=True),
    test_size=0.2, stratify=y_s, random_state=RANDOM_STATE,
)

print(f"Train={len(Xtr_s)}  Test={len(Xte_s)}  pos_rate_train={ytr.mean():.3f}")
"""))

    # ---- Helper ----
    cells.append(_md("## Helper: evaluate a classifier\n\nFits on train, scores ROC-AUC / Precision / Recall / F1 / Accuracy on the locked test set."))
    cells.append(_code("""
def score_model(model, X_train, y_train, X_test, y_test, name="model"):
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    row = {
        "model": name,
        "roc_auc": metrics["roc_auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "n_features": X_train.shape[1],
    }
    print(
        f"{name:<32} AUC={row['roc_auc']:.4f}  F1={row['f1']:.4f}  "
        f"Rec={row['recall']:.4f}  Acc={row['accuracy']:.4f}  feats={row['n_features']}"
    )
    return row, metrics

leaderboard = []
"""))

    # ---- FE ----
    cells.append(_md("""## Feature Engineering ladder

Add one idea at a time (HyperAck playbook):

| Stage | What we add |
|---|---|
| `raw` | Original numeric matrix |
| `logs` | `log1p` on skewed non-negative columns |
| `ratios` | Pairwise intensity ratios |
| `interactions` | Products of top-MI features |
| `full_fe` | + KMeans clusters + quantile bins (**fit on train only**) |
| `selected` | Top mutual-information subset |
"""))
    cells.append(_code("""
fe_rows = []
for mode, Xtr, Xte in [("safe", Xtr_s, Xte_s), ("unsafe", Xtr_u, Xte_u)]:
    for stage in ["raw", "logs", "ratios", "interactions", "full_fe", "selected"]:
        A, B, fe_meta = build_feature_matrix(Xtr, ytr, Xte, stage=stage)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
        ])
        row, _ = score_model(model, A, ytr, B, yte, name=f"{mode}/{stage}/lgbm")
        row.update({"mode": mode, "stage": stage, "top_mi": fe_meta.get("top_mi", [])[:5]})
        fe_rows.append(row)

fe_df = pd.DataFrame(fe_rows)
fe_df.sort_values(["mode", "roc_auc"], ascending=[True, False])
"""))
    cells.append(_code("""
# Visualize FE lift (safe mode)
safe_fe = fe_df[fe_df["mode"] == "safe"].copy()
plt.figure(figsize=(10, 4))
sns.barplot(data=safe_fe, x="stage", y="roc_auc", color="#16845B")
plt.title(f"{KEY}: Feature engineering ladder (Safe, LightGBM)")
plt.ylim(max(0.5, safe_fe["roc_auc"].min() - 0.05), min(1.02, safe_fe["roc_auc"].max() + 0.03))
plt.xticks(rotation=20)
plt.show()
"""))

    # ---- Safe baseline models ----
    cells.append(_md("""## Safe baseline models

Train classic baselines on **safe + full FE** (honest feature set):

- Logistic Regression  
- LightGBM  
- XGBoost  
"""))
    cells.append(_code("""
Xtr_safe_fe, Xte_safe_fe, _ = build_feature_matrix(Xtr_s, ytr, Xte_s, stage="full_fe")

safe_baselines = {
    "safe/baseline/logistic": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    "safe/baseline/lightgbm": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
    ]),
    "safe/baseline/xgboost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
    ]),
}

for name, model in safe_baselines.items():
    row, _ = score_model(model, Xtr_safe_fe, ytr, Xte_safe_fe, yte, name=name)
    leaderboard.append(row)
"""))

    # ---- Unsafe baseline ----
    cells.append(_md("""## Unsafe baseline models

Same algorithms on **unsafe + full FE**. If leakage columns exist, scores may jump — that lift is usually **not deployable**.
"""))
    cells.append(_code("""
Xtr_unsafe_fe, Xte_unsafe_fe, _ = build_feature_matrix(Xtr_u, ytr, Xte_u, stage="full_fe")

unsafe_baselines = {
    "unsafe/baseline/logistic": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    "unsafe/baseline/lightgbm": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
    ]),
    "unsafe/baseline/xgboost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
    ]),
}

for name, model in unsafe_baselines.items():
    row, _ = score_model(model, Xtr_unsafe_fe, ytr, Xte_unsafe_fe, yte, name=name)
    leaderboard.append(row)
"""))

    cells.append(_md("## Safe vs Unsafe baseline comparison"))
    cells.append(_code("""
lb = pd.DataFrame(leaderboard)
cmp = lb.pivot_table(index=lb["model"].str.split("/").str[-1], columns=lb["model"].str.split("/").str[0], values="roc_auc")
print(cmp)
if "safe" in cmp.columns and "unsafe" in cmp.columns:
    cmp["leakage_gap"] = cmp["unsafe"] - cmp["safe"]
    print("\\nLeakage gap (unsafe - safe):")
    print(cmp["leakage_gap"])

plt.figure(figsize=(8, 4))
plot_df = lb[lb["model"].str.contains("baseline")].copy()
plot_df["mode"] = plot_df["model"].str.split("/").str[0]
plot_df["algo"] = plot_df["model"].str.split("/").str[-1]
sns.barplot(data=plot_df, x="algo", y="roc_auc", hue="mode", palette={"safe": "#16845B", "unsafe": "#B42318"})
plt.title(f"{KEY}: Safe vs Unsafe baselines (ROC-AUC)")
plt.show()
"""))

    # ---- Optimized safe ----
    cells.append(_md("""## Optimized Safe models

Optimization methods from the HyperAck playbook (on **safe + full FE** only):

1. **RandomizedSearchCV** LightGBM (maximize ROC-AUC, 3-fold CV)  
2. **RandomizedSearchCV** XGBoost  
3. Hand-tuned ExtraTrees-style strong LightGBM bag proxy (more trees / lower LR)  
4. Soft-style blend: average LGBM + XGB probabilities  
"""))
    cells.append(_code("""
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# 1) Tuned LightGBM
lgbm_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
])
lgbm_search = RandomizedSearchCV(
    lgbm_pipe,
    {
        "model__n_estimators": [200, 300, 450],
        "model__learning_rate": [0.02, 0.03, 0.05],
        "model__num_leaves": [16, 24, 31],
        "model__min_child_samples": [20, 40, 60],
        "model__subsample": [0.7, 0.85],
        "model__colsample_bytree": [0.7, 0.85],
        "model__reg_lambda": [0.1, 1.0, 2.0],
    },
    n_iter=12,
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)
row, _ = score_model(lgbm_search, Xtr_safe_fe, ytr, Xte_safe_fe, yte, name="safe/optimized/tuned_lightgbm")
leaderboard.append(row)
print("Best LGBM params:", lgbm_search.best_params_)
"""))
    cells.append(_code("""
# 2) Tuned XGBoost
xgb_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
])
xgb_search = RandomizedSearchCV(
    xgb_pipe,
    {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.02, 0.03, 0.05],
        "model__max_depth": [4, 5, 6],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.75, 0.9],
        "model__colsample_bytree": [0.7, 0.85],
        "model__reg_lambda": [1.0, 2.0, 4.0],
    },
    n_iter=10,
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)
row, _ = score_model(xgb_search, Xtr_safe_fe, ytr, Xte_safe_fe, yte, name="safe/optimized/tuned_xgboost")
leaderboard.append(row)
print("Best XGB params:", xgb_search.best_params_)
"""))
    cells.append(_code("""
# 3) Soft probability blend of the two tuned models (HyperAck exp-15 style)
class SoftBlend:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1 / len(models)] * len(models)
    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self
    def predict_proba(self, X):
        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        p = sum(w * pr for w, pr in zip(self.weights, probs))
        return np.column_stack([1 - p, p])

blend = SoftBlend(
    models=[lgbm_search.best_estimator_, xgb_search.best_estimator_],
    weights=[0.55, 0.45],
)
row, _ = score_model(blend, Xtr_safe_fe, ytr, Xte_safe_fe, yte, name="safe/optimized/soft_blend_lgbm_xgb")
leaderboard.append(row)
"""))

    # ---- Optimized unsafe ----
    cells.append(_md("""## Optimized Unsafe models

Same search recipe on the **unsafe** matrix. Use this only to measure the leakage ceiling — do **not** deploy if leakage columns are required for the lift.
"""))
    cells.append(_code("""
lgbm_u = RandomizedSearchCV(
    Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
    ]),
    {
        "model__n_estimators": [200, 300, 450],
        "model__learning_rate": [0.02, 0.05],
        "model__num_leaves": [16, 31],
        "model__min_child_samples": [20, 40],
        "model__subsample": [0.8, 0.9],
        "model__colsample_bytree": [0.8, 0.9],
    },
    n_iter=8,
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)
row, _ = score_model(lgbm_u, Xtr_unsafe_fe, ytr, Xte_unsafe_fe, yte, name="unsafe/optimized/tuned_lightgbm")
leaderboard.append(row)
"""))

    # ---- Final leaderboard ----
    cells.append(_md("""## Final leaderboard

Rank every model we trained in this notebook. Then load the full playbook ladder (15 experiments × safe/unsafe) for the complete picture.
"""))
    cells.append(_code("""
final = pd.DataFrame(leaderboard).sort_values("roc_auc", ascending=False).reset_index(drop=True)
final["rank"] = np.arange(1, len(final) + 1)
display_cols = ["rank", "model", "roc_auc", "f1", "recall", "precision", "accuracy", "n_features"]
final[display_cols]
"""))
    cells.append(_code("""
plt.figure(figsize=(10, max(4, 0.35 * len(final))))
sns.barplot(data=final, y="model", x="roc_auc", color="#2563EB")
plt.title(f"{KEY}: Notebook model leaderboard (ROC-AUC)")
plt.xlim(max(0.5, final["roc_auc"].min() - 0.05), min(1.02, final["roc_auc"].max() + 0.02))
plt.show()

best = final.iloc[0]
print(f"WINNER: {best['model']}  AUC={best['roc_auc']:.4f}  F1={best['f1']:.4f}")
"""))

    cells.append(_md("""## Full playbook ladder results (precomputed)

If you already ran `general_pipeline/playbook/run_playbook.py`, load the 01–15 experiment JSON here for FE / optimization rankings without re-training everything.
"""))
    cells.append(_code("""
ladder = results_frame(KEY)
if ladder.empty:
    print("No ladder results yet. Run:")
    print(f"  .venv/bin/python general_pipeline/playbook/run_playbook.py --dataset {KEY}")
else:
    print("Ladder rows:", len(ladder))
    safe_rank = ladder[ladder["mode"] == "safe"].sort_values("roc_auc", ascending=False)
    print("\\nTop safe ladder experiments:")
    print(safe_rank[["exp_id", "exp_name", "optimization_method", "roc_auc", "f1", "feature_count"]].head(10).to_string(index=False))
    print("\\nBest optimization method (safe):",
          safe_rank.iloc[0]["optimization_method"], "/", safe_rank.iloc[0]["exp_name"])
"""))

    cells.append(_md(f"""## Summary & next steps

| Question | Where to look |
|---|---|
| Did FE help? | FE ladder bar chart above |
| Safe vs Unsafe gap? | Baseline comparison + leakage policy ({leak_txt}) |
| Which optimization wins? | Notebook leaderboard + playbook `OPTIMIZATION_REPORT.md` |
| Deployable model? | Prefer **safe/optimized** winner |

Project artifacts:

- `external_projects/{key}_exp/FEATURE_ENGINEERING_REPORT.md`
- `external_projects/{key}_exp/OPTIMIZATION_REPORT.md`
- `external_projects/{key}_exp/COMPLETE_MASTER_REPORT.md`
- `external_projects/{key}_exp/notebooks/` (this walkthrough + ladder notebooks)
"""))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    pdir = project_dir(key)
    ndir = pdir / "notebooks"
    ndir.mkdir(parents=True, exist_ok=True)
    path = ndir / f"part1_{key}_classification.ipynb"
    path.write_text(nbf.writes(nb))
    return path


def generate_all_walkthroughs(keys=None):
    keys = keys or [s.key for s in DATASET_CATALOG]
    paths = []
    for key in keys:
        path = build_walkthrough(key)
        print(f"[{key}] wrote {path.relative_to(path.parents[2])}")
        paths.append(path)
    return paths


if __name__ == "__main__":
    generate_all_walkthroughs()
