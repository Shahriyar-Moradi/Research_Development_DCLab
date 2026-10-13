"""
leakage_audit_samples.py
========================

Generates SFT training samples for the LEAKAGE AUDIT step of the DCLab
workflow agent, using "planted problems" on the real datasets in
`data/public/`.

Idea
----
Take a real dataset, add ONE column whose truth we control (a copy of the
target, a value recorded after the outcome, an ID that follows sort order,
a perfectly legitimate feature with a scary name, ...). Because we planted
it, we know the correct decision for certain. Every number shown to the
model (univariate ROC-AUC, missingness by class, CV score with/without the
column, ...) is COMPUTED on the real data, the same way the repo's own
leakage audit (`dclab_rnd/science.py::suspicious_features`) computes it.

Real columns are labelled too, from the repo's own data contracts
(`dclab_rnd/agentic/catalog.py`: decision moment + blocked columns).

Each sample is one decision about one column:
    input  = stage card + dataset card + column + computed evidence + allowed actions
    output = DECISION / WHY / RULE / CONFIDENCE / NEXT

A built-in checker drops any sample whose computed evidence does not match
the label its generator intended (e.g. a "copy of the target" that did not
reach high AUC), drops duplicates, and checks that the answer never appears
in the input.

Splits are by DATASET, never by row, so the test set measures skill on
datasets the model has not seen.

Usage
-----
    python research/llm-fine-tuning/experiments/sft_factory/leakage_audit_samples.py                 # all datasets, defaults
    python research/llm-fine-tuning/experiments/sft_factory/leakage_audit_samples.py --variants 4    # quicker run
    python research/llm-fine-tuning/experiments/sft_factory/leakage_audit_samples.py --test-datasets online_shoppers german_credit --val-datasets wine_quality
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # research/llm-fine-tuning/experiments/sft_factory -> repo root
DATA = ROOT / "data" / "public"
sys.path.insert(0, str(ROOT))
from dclab_rnd.agentic.catalog import DATASETS as CATALOG  # noqa: E402

# Same pattern the repo's leakage audit uses (dclab_rnd/science.py).
SUSPICIOUS_NAME = re.compile(
    r"(^|_)(target|label|outcome|result|response|accepted|approved|converted|defaulted|churned|final)(_|$)",
    re.IGNORECASE,
)

ACTIONS = ["keep", "exclude", "flag_for_review", "rebuild_within_folds"]

SYSTEM_PROMPT = (
    "You are the DCLab workflow agent. You build ML models step by step: "
    "data_understanding -> leakage_audit -> feature_engineering -> model_selection -> "
    "optimization_reliability. At each step you decide using only the evidence given, "
    "choose exactly one allowed action, explain why with the numbers, cite the rule, "
    "and say what happens next."
)

STAGE_CARD = (
    "Leakage audit: decide whether one column may be used by a model that predicts at the "
    "declared decision moment. A column must exist at that moment and must not be derived "
    "from the outcome or from information across train/test splits."
)

ACTION_GUIDE = (
    "keep = available at the decision moment, no leakage evidence | "
    "exclude = not available at the decision moment, a copy or consequence of the outcome, "
    "or an ordering/identifier artifact | "
    "flag_for_review = strong signal but availability cannot be confirmed from the evidence | "
    "rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold"
)

TARGETS = {
    "adult": "whether a person earns more than $50K per year",
    "bank_marketing": "whether a client subscribes to a term deposit",
    "breast_cancer": "whether a tumour is malignant",
    "credit_default": "whether a client defaults on next month's credit-card payment",
    "german_credit": "whether a loan applicant is a bad credit risk",
    "heart_disease": "whether heart disease is present",
    "mushroom": "whether a mushroom is poisonous",
    "online_shoppers": "whether a website session ends in a purchase",
    "spambase": "whether an email is spam",
    "wine_quality": "whether a wine is rated quality 6 or higher",
}

# Real data-dictionary entries (from the UCI dataset documentation) for the real columns
# that the repo's data contracts block, so the decision is made from stated meaning.
REAL_DICTIONARY = {
    "bank_marketing": {"duration": "Last contact duration, in seconds."},
    "online_shoppers": {
        "Administrative": "Number of administrative pages visited in the session.",
        "Administrative_Duration": "Total time spent on administrative pages in the session.",
        "Informational": "Number of informational pages visited in the session.",
        "Informational_Duration": "Total time spent on informational pages in the session.",
        "ProductRelated": "Number of product-related pages visited in the session.",
        "ProductRelated_Duration": "Total time spent on product-related pages in the session.",
        "BounceRates": "Average bounce rate of the pages visited in the session.",
        "ExitRates": "Average exit rate of the pages visited in the session.",
        "PageValues": "Average value of the pages visited before completing a transaction.",
    },
}

# Real columns whose correct label is genuinely contested in the repo's own contracts.
SKIP_REAL = {"adult": {"fnlwgt"}}


# ----------------------------------------------------------------------------
# Data loading and evidence (computed, never invented)
# ----------------------------------------------------------------------------

def load(key: str, max_rows: int, seed: int):
    X = pd.read_parquet(DATA / key / "X.parquet")
    y = pd.read_parquet(DATA / key / "y.parquet").iloc[:, 0].astype(int)
    meta = json.loads((DATA / key / "meta.json").read_text())
    if len(X) > max_rows:  # stratified subsample, so evidence is fast but representative
        rng = np.random.default_rng(seed)
        idx = []
        for cls in (0, 1):
            cls_idx = np.flatnonzero(y.to_numpy() == cls)
            take = int(round(max_rows * len(cls_idx) / len(y)))
            idx.extend(rng.choice(cls_idx, size=take, replace=False))
        idx = np.sort(np.array(idx))
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True), meta


def univariate_auc(values: pd.Series, y: pd.Series) -> float | None:
    num = pd.to_numeric(values, errors="coerce")
    if num.notna().sum() < 20 or num.nunique() < 2:
        return None
    raw = roc_auc_score(y, num.fillna(num.median()))
    return float(max(raw, 1 - raw))


def cv_auc(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> float:
    if X.shape[1] == 0:
        return 0.5
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        model = HistGradientBoostingClassifier(max_iter=80, random_state=seed)
        model.fit(X.iloc[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(scores))


@dataclass
class Evidence:
    kind: str
    missing_rate: float
    missing_pos: float
    missing_neg: float
    unique_ratio: float
    auc: float | None
    exact_target_copy: bool
    suspicious_name: bool
    after_fraction: float | None  # date columns only
    cv_with: float
    cv_without: float
    consecutive_integers: bool = False

    @property
    def cv_change(self) -> float:
        return self.cv_with - self.cv_without

    def render(self) -> str:
        lines = [
            f"- type: {'date (stored as days relative to the decision moment)' if self.kind == 'date' else 'numeric'}",
            f"- missing: {self.missing_rate:.1%} (positives {self.missing_pos:.1%}, negatives {self.missing_neg:.1%})",
            f"- unique values: {self.unique_ratio:.1%} of rows",
            f"- univariate ROC-AUC: {'n/a' if self.auc is None else f'{self.auc:.3f}'}",
            f"- identical to the target (or its inverse): {'yes' if self.exact_target_copy else 'no'}",
            f"- name matches the suspicious-word list: {'yes' if self.suspicious_name else 'no'}",
            f"- values form a run of consecutive integers (row-number pattern): {'yes' if self.consecutive_integers else 'no'}",
        ]
        if self.after_fraction is not None:
            lines.append(f"- values after the decision moment: {self.after_fraction:.1%}")
        lines.append(
            f"- 3-fold CV ROC-AUC with the column: {self.cv_with:.3f} | without: {self.cv_without:.3f} "
            f"| change: {self.cv_change:+.3f}"
        )
        return "\n".join(lines)


def compute_evidence(name, values, y, base_X, base_auc, kind="numeric", in_base=False) -> Evidence:
    s = pd.Series(values)
    num = pd.to_numeric(s, errors="coerce")
    yv = y.to_numpy()
    exact = False
    if num.notna().all() and num.nunique() <= 2:
        codes = pd.factorize(num, sort=True)[0]
        exact = bool(np.array_equal(codes, yv) or np.array_equal(codes, 1 - yv))
    if in_base:  # real column already in the safe base: compare base vs base-minus-column
        cv_with, cv_without = base_auc, cv_auc(base_X.drop(columns=[name]), y)
    else:
        cv_with, cv_without = cv_auc(base_X.assign(**{name: num.to_numpy()}), y), base_auc
    nn = num.dropna()
    consecutive = bool(len(nn) > 20 and (nn % 1 == 0).all() and nn.nunique() / len(s) > 0.9
                       and (nn.max() - nn.min() + 1) / nn.nunique() <= 1.05)
    return Evidence(
        consecutive_integers=consecutive,
        kind=kind,
        missing_rate=float(num.isna().mean()),
        missing_pos=float(num[yv == 1].isna().mean()),
        missing_neg=float(num[yv == 0].isna().mean()),
        unique_ratio=float(s.nunique(dropna=False) / len(s)),
        auc=univariate_auc(num, y),
        exact_target_copy=exact,
        suspicious_name=bool(SUSPICIOUS_NAME.search(name)),
        after_fraction=float((num > 0).mean()) if kind == "date" else None,
        cv_with=cv_with,
        cv_without=cv_without,
    )


# ----------------------------------------------------------------------------
# Planted problems. Each returns a Plant; `sane` checks the computed evidence
# really supports the intended label (otherwise the sample is dropped).
# ----------------------------------------------------------------------------

@dataclass
class Plant:
    generator: str
    name: str
    values: np.ndarray
    action: str
    rule: str
    confidence: str
    description: str | None = None
    kind: str = "numeric"
    why: Callable[["Evidence", "Plant"], str] = None
    sane: Callable[["Evidence", "Plant"], bool] = lambda e, p: True
    extra: dict = field(default_factory=dict)


def _pick(rng, items):
    return items[rng.integers(len(items))]


def _strength(e):
    a = e.auc or 0.5
    return "very strong" if a >= 0.9 else "strong" if a >= 0.8 else "moderate" if a >= 0.65 else "weak"


def _auc(e):
    return "n/a" if e.auc is None else f"{e.auc:.3f}"


def gen_target_copy(X, y, rng):
    flip = _pick(rng, [0.0, 0.0, 0.01, 0.02])
    v = y.to_numpy().copy()
    mask = rng.random(len(v)) < flip
    v[mask] = 1 - v[mask]
    if rng.random() < 0.5:
        v = 1 - v
    v = v * _pick(rng, [1, 1, 2, 5]) + _pick(rng, [0, 0, 1, 3])
    return Plant(
        "target_copy", _pick(rng, SHARED + SCARY_NAMES),
        v.astype(float), "exclude", "DCLAB-R04 (target-derived leakage)", "high",
        description=_pick(rng, [None, None, "Internal status code from the CRM export."]),
        why=lambda e, p: (
            f"`{p.name}` reproduces the outcome almost perfectly (univariate ROC-AUC {_auc(e)}"
            + ("; it is identical to the target" if e.exact_target_copy else "")
            + f"). A column like this is the label in disguise, so it cannot exist before the outcome. "
            f"Adding it moves CV ROC-AUC by {e.cv_change:+.3f}, which is the size of the illusion, not real skill."
        ),
        sane=lambda e, p: e.auc is not None and e.auc >= 0.95,
    )


POST_DESC = [
    "Recorded by the operations team after the case is closed.",
    "Filled in once the final status is known.",
    "Updated at month end, after the outcome period has finished.",
    "Entered during the follow-up review that happens after the result.",
]
POST_NAMES = ["resolution_days", "closing_amount", "account_review_score", "collections_contacts",
              "settlement_value", "case_close_score", "handling_minutes"]
NEUTRAL_NAMES = ["metric_17", "score_b", "ops_value", "kpi_3", "field_x9", "index_c", "measure_42",
                 "signal_a", "attr_88", "value_q", "status_code", "segment_tag", "tier_code", "review_points"]
# Names that match the repo's suspicious-word regex. Used by leaky AND clean generators,
# so a name never decides the label by itself (DCLAB-R05).
SCARY_NAMES = ["prior_outcome_count", "previous_result_score", "final_grade_last_year", "target_segment",
               "response_rate_last_quarter", "final_status_score", "outcome_index", "label_group"]
ID_NAMES = ["record_id", "row_number", "ticket_no", "load_order", "entry_seq", "case_id"]
# Shared pools: each name appears under several different correct actions.
SHARED = NEUTRAL_NAMES  # meaningless names; post-outcome-sounding names never get 'keep'


def _shifted(y, rng, lo, hi):
    d = rng.uniform(lo, hi)
    return rng.normal(0, 1, len(y)) + y.to_numpy() * d * _pick(rng, [1, -1])


def gen_post_outcome_documented(X, y, rng):
    return Plant(
        "post_outcome_documented", _pick(rng, SHARED + POST_NAMES + SCARY_NAMES), _shifted(y, rng, 1.0, 2.6),
        "exclude", "DCLAB-R04 (post-outcome leakage)", "high", description=_pick(rng, POST_DESC),
        why=lambda e, p: (
            f"The data dictionary says `{p.name}` is filled in after the outcome is known, so it does not exist at the "
            f"decision moment. Its univariate ROC-AUC of {_auc(e)} and CV change of {e.cv_change:+.3f} come from "
            f"seeing the result, not from predicting it."
        ),
        sane=lambda e, p: e.auc is not None and e.auc >= 0.70,
    )


def gen_post_outcome_undocumented(X, y, rng):
    return Plant(
        "strong_signal_unknown_origin", _pick(rng, SHARED + POST_NAMES + SCARY_NAMES), _shifted(y, rng, 2.4, 3.6),
        "flag_for_review", "DCLAB-R05 (high AUC starts a review, it does not prove leakage)", "medium",
        description=None,
        why=lambda e, p: (
            f"`{p.name}` is unusually strong (univariate ROC-AUC {_auc(e)} vs {p.extra['top']:.3f} for the strongest real feature) and "
            f"raises CV ROC-AUC by {e.cv_change:+.3f}, but there is no data-dictionary entry saying when it is recorded. "
            f"A strong score alone does not prove leakage, and it does not prove availability either, so a person "
            f"with the source lineage has to confirm it."
        ),
        sane=lambda e, p: e.auc is not None and 0.88 <= e.auc < 0.995 and e.auc > p.extra['top'] + 0.02
                          and not e.exact_target_copy and not e.consecutive_integers,
    )


def gen_full_data_target_encoding(X, y, rng):
    cands = [c for c in X.columns if 20 <= X[c].nunique() <= 300] or [c for c in X.columns if 5 <= X[c].nunique() <= 300]
    if cands:
        col = _pick(rng, cands)
        groups = X[col]
    else:
        col = _pick(rng, list(X.columns))
        groups = pd.qcut(X[col].rank(method="first"), 20, labels=False)
    rate = y.groupby(groups).transform("mean").to_numpy()
    name = _pick(rng, [f"{col}_target_rate", f"avg_outcome_by_{col}", f"{col}_mean_encoded"])
    desc = _pick(rng, [
        f"Average outcome rate for each value of `{col}`, computed over the full dataset before splitting.",
        f"Mean of the target per `{col}` group, calculated once using all rows, including evaluation rows.",
    ])
    return Plant(
        "full_data_target_encoding", re.sub(r"[^A-Za-z0-9_]", "_", name), rate, "rebuild_within_folds",
        "DCLAB-R04 (cross-split aggregate leakage)", "high", description=desc, extra={"source": col},
        why=lambda e, p: (
            f"Encoding `{p.extra['source']}` by its outcome rate is a legitimate idea, but this version was computed on all "
            f"rows, so every evaluation row's own label is inside its feature value. "
            + (f"Here that already shows up as a CV gain of {e.cv_change:+.3f}. " if e.cv_change > 0.01 else
               f"Here the visible effect is small (CV change {e.cv_change:+.3f}) because each group is large, but the method "
               f"is still wrong and the same mistake on a high-cardinality column produces large fake gains. ")
            + "The fix is to recompute the rate inside each training fold, not to drop the idea."
        ),
        sane=lambda e, p: p.description is not None,
    )


def gen_sorted_id(X, y, rng):
    order = np.lexsort((rng.random(len(y)), y.to_numpy()))
    ids = np.empty(len(y))
    ids[order] = np.arange(len(y)) + rng.integers(1000, 90000)
    return Plant(
        "sorted_identifier", _pick(rng, ID_NAMES),
        ids, "exclude", "DCLAB-R04 (identifier / ordering leakage)", "high",
        description=_pick(rng, [None, "Sequential number assigned when rows were exported."]),
        why=lambda e, p: (
            f"`{p.name}` is a run of consecutive integers, unique for {e.unique_ratio:.0%} of rows: a row number, not a "
            f"measurement. Yet it separates the classes "
            f"(univariate ROC-AUC {_auc(e)}). That only happens when rows were ordered by outcome before numbering. "
            f"New rows will not follow that order, so the signal disappears in production."
        ),
        sane=lambda e, p: e.consecutive_integers and e.auc is not None and e.auc >= 0.9,
    )


DATE_NAMES = ["last_status_update", "last_activity_date", "record_date", "event_date", "signup_date",
              "account_opened", "first_seen_date", "updated_on"]


def gen_random_id(X, y, rng):
    ids = rng.permutation(len(y)).astype(float) + rng.integers(1000, 90000)
    return Plant(
        "random_identifier", _pick(rng, ID_NAMES), ids, "exclude", "DCLAB-R04 (identifier / entity leakage)", "high",
        description=_pick(rng, [None, "Sequential number assigned when rows were exported."]),
        why=lambda e, p: (
            f"`{p.name}` is a run of consecutive integers, unique for {e.unique_ratio:.0%} of rows: a row number. It shows "
            f"no signal here (univariate ROC-AUC {_auc(e)}), but identifiers describe how the data was stored, not the "
            f"entity, so they can only add noise now or leak later if the row order changes. Identifiers stay out."
        ),
        sane=lambda e, p: e.consecutive_integers and e.auc is not None and e.auc < 0.6,
    )


def gen_future_date(X, y, rng):
    n = len(y)
    v = -rng.integers(1, 365, n).astype(float)
    pos = np.flatnonzero(y.to_numpy() == 1)
    share = rng.uniform(0.4, 0.8)
    after = rng.choice(pos, size=int(len(pos) * share), replace=False)
    v[after] = rng.integers(1, 120, len(after))
    return Plant(
        "date_after_decision", _pick(rng, DATE_NAMES),
        v, "exclude", "DCLAB-R04 (future leakage)", "high",
        description=_pick(rng, [None, "Date of the most recent change to the record."]), kind="date",
        why=lambda e, p: (
            f"{e.after_fraction:.0%} of `{p.name}` values fall after the decision moment, and those late dates are concentrated "
            f"in one class (univariate ROC-AUC {_auc(e)}). At prediction time these dates do not exist yet, so the "
            f"column leaks the future."
        ),
        sane=lambda e, p: e.after_fraction is not None and e.after_fraction > 0.03,
    )


def gen_clean_date(X, y, rng):
    return Plant(
        "date_before_decision", _pick(rng, DATE_NAMES),
        -rng.integers(30, 1500, len(y)).astype(float), "keep", "DCLAB-R04 (checked: no future leakage)", "high",
        description=_pick(rng, [None, "Date of the most recent change to the record."]), kind="date",
        why=lambda e, p: (
            f"All `{p.name}` values are before the decision moment ({e.after_fraction:.0%} after), and the column carries "
            f"no outcome signal (univariate ROC-AUC {_auc(e)}). It is safe to use; whether it is useful is a "
            f"feature-engineering question, not a leakage one."
        ),
        sane=lambda e, p: e.after_fraction == 0,
    )


def gen_missing_by_outcome(X, y, rng, documented: bool):
    v = rng.normal(50, 10, len(y))
    yv = y.to_numpy()
    p_miss_pos, p_miss_neg = (0.08, 0.9) if rng.random() < 0.5 else (0.9, 0.08)
    miss = np.where(yv == 1, rng.random(len(y)) < p_miss_pos, rng.random(len(y)) < p_miss_neg)
    v[miss] = np.nan
    name = _pick(rng, SHARED + POST_NAMES + ["followup_score", "survey_rating", "callback_length"])
    if documented:
        return Plant(
            "missingness_after_outcome", name, v, "exclude", "DCLAB-R04 (post-outcome leakage via missingness)", "high",
            description="Collected only during the follow-up that happens after the outcome.",
            why=lambda e, p: (
                f"The values of `{p.name}` are noise, but whether a value exists depends on the outcome: missing for "
                f"{e.missing_pos:.0%} of positives vs {e.missing_neg:.0%} of negatives. The dictionary says it is collected "
                f"after the outcome, so its presence is a consequence of the result (CV change {e.cv_change:+.3f})."
            ),
            sane=lambda e, p: abs(e.missing_pos - e.missing_neg) > 0.5,
        )
    return Plant(
        "missingness_unknown_origin", name, v, "flag_for_review",
        "DCLAB-R05 (strong signal needs a semantic review)", "medium", description=None,
        why=lambda e, p: (
            f"Missingness in `{p.name}` splits the classes sharply ({e.missing_pos:.0%} missing for positives vs "
            f"{e.missing_neg:.0%} for negatives), which typically means the field is filled in at a stage tied to "
            f"the outcome. With no data-dictionary entry this cannot be confirmed either way, so it needs review "
            f"before any model uses it."
        ),
        sane=lambda e, p: abs(e.missing_pos - e.missing_neg) > 0.5,
    )


def gen_clean_noise(X, y, rng):
    base = X[_pick(rng, list(X.columns))].rank(pct=True).to_numpy()
    v = base * rng.uniform(0, 0.4) + rng.normal(0, 1, len(y))
    return Plant(
        "clean_weak_feature", _pick(rng, SHARED + ["visits_last_quarter", "household_size", "branch_code"]),
        v, "keep", "DCLAB-R04 (checked: no leakage pattern)", "high",
        description=_pick(rng, [None, "Captured when the record is created."]),
        why=lambda e, p: (
            f"`{p.name}` shows no leakage pattern: it is not a copy of the target, it has no outcome-linked missingness, "
            f"it is not a row-number sequence, and its signal is weak (univariate ROC-AUC {_auc(e)}, CV change "
            f"{e.cv_change:+.3f}). Keep it; weak features are "
            f"pruned later in feature engineering, not in the leakage audit."
        ),
        sane=lambda e, p: e.auc is not None and e.auc < 0.65 and not e.suspicious_name and not e.consecutive_integers,
    )


def gen_scary_name_clean(X, y, rng, safe_cols):
    col = _pick(rng, safe_cols)
    v = X[col].rank(pct=True).to_numpy() * rng.uniform(0.3, 1.0) + rng.normal(0, 0.5, len(y))
    name = _pick(rng, SCARY_NAMES)
    return Plant(
        "suspicious_name_but_clean", name, v, "keep", "DCLAB-R05 (a suspicious name starts a review, it does not prove leakage)",
        "high", description=_pick(rng, [
            f"Known before the decision moment; derived from `{col}` and from earlier, already-closed periods.",
            "Computed from historical periods that ended before the decision moment.",
        ]),
        why=lambda e, p: (
            f"The name of `{p.name}` matches the suspicious-word list, but the data dictionary places it before the "
            f"decision moment and the evidence agrees: no target copy, no outcome-linked missingness, {_strength(e)} signal "
            f"(univariate ROC-AUC {_auc(e)}). A word in a column name is a reason to look, not a verdict."
        ),
        sane=lambda e, p: e.suspicious_name and e.auc is not None and e.auc < 0.9 and not e.exact_target_copy,
    )


def gen_strong_legit(X, y, rng, top_cols):
    a, b = rng.choice(top_cols, size=2, replace=False) if len(top_cols) >= 2 else (top_cols[0], top_cols[0])
    v = X[a].rank(pct=True).to_numpy() + X[b].rank(pct=True).to_numpy() * rng.uniform(0.3, 1.0)
    name = _pick(rng, SHARED + ["risk_index", "combined_score", "profile_score"])
    return Plant(
        "strong_but_available", name, v, "keep", "DCLAB-R05 (high AUC alone does not prove leakage)", "high",
        description=f"Computed at the decision moment from `{a}` and `{b}`.", extra={"a": a, "b": b},
        why=lambda e, p: (
            f"`{p.name}` has {_strength(e)} signal (univariate ROC-AUC {_auc(e)}), and its lineage is clear: it is computed at the decision "
            f"moment from `{p.extra['a']}` and `{p.extra['b']}`, which are themselves available then. Predictive "
            f"power from available inputs is the goal, not a leak."
        ),
        sane=lambda e, p: p.description is not None and not e.exact_target_copy,
    )


# ----------------------------------------------------------------------------
# Sample assembly and checks
# ----------------------------------------------------------------------------

def dataset_card(key, meta, n_rows, pos_rate, top_auc) -> str:
    return (
        f"DATASET: {meta['name']} ({key}). Predicts {TARGETS[key]}.\n"
        f"DECISION MOMENT: {CATALOG[key]['decision']}\n"
        f"ROWS: {meta['n_rows']:,} (evidence computed on a stratified sample of {n_rows:,}) | "
        f"POSITIVE RATE: {pos_rate:.1%} | STRONGEST REAL FEATURE (univariate ROC-AUC): {top_auc:.3f}"
    )


def build_sample(card, name, desc, ev: Evidence, action, why, rule, confidence, meta) -> dict:
    user = (
        f"STAGE: leakage_audit\n{STAGE_CARD}\n\n{card}\n\n"
        f"COLUMN: `{name}`\n"
        f"DATA DICTIONARY: {desc if desc else 'no entry'}\n"
        f"EVIDENCE:\n{ev.render()}\n\n"
        f"ALLOWED ACTIONS: {' | '.join(ACTIONS)}\n({ACTION_GUIDE})"
    )
    nxt = {
        "keep": f"Keep `{name}` and continue the audit with the next column.",
        "exclude": f"Remove `{name}` from the feature list, record the reason in the decision ledger, and continue the audit.",
        "flag_for_review": f"Keep `{name}` out of all models until the data owner confirms when it is recorded; continue the audit meanwhile.",
        "rebuild_within_folds": f"Drop this version of `{name}`, recompute it out-of-fold inside each training split, and re-audit it.",
    }[action]
    assistant = (
        f"DECISION: {action}\nWHY: {why}\nRULE: {rule}\nCONFIDENCE: {confidence}\nNEXT: {nxt}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": meta,
    }


ANSWER_WORDS = ["exclude", "flag_for_review", "rebuild_within_folds"]


def answer_leaks_into_input(sample) -> bool:
    """The column section (name, dictionary, evidence) must not contain the answer.
    The dataset card is excluded from the check: it quotes the repo's real data contract,
    which may legitimately say e.g. "Exclude completed-session counts"."""
    user = sample["messages"][1]["content"]
    body = user.split("COLUMN:")[1].split("ALLOWED ACTIONS:")[0].lower()
    gen = sample["metadata"]["generator"].lower()
    return any(w in body for w in ANSWER_WORDS) or (gen in body and gen not in ("real_column",))


def real_column_samples(key, X, y, meta, card, base_X, base_auc, rng) -> list[tuple]:
    blocked = [c for c in CATALOG[key].get("blocked", []) if c in X.columns]
    safe = [c for c in X.columns if c not in blocked and c not in SKIP_REAL.get(key, set())]
    aucs = {c: (univariate_auc(X[c], y) or 0.5) for c in safe}
    top = sorted(safe, key=lambda c: -aucs[c])[:3]
    rest = [c for c in safe if c not in top]
    chosen = top + ([rng.choice(rest)] if rest else [])
    out = []
    for c in blocked:
        ev = compute_evidence(c, X[c].to_numpy(), y, base_X, base_auc)
        why = (
            f"The dataset's decision moment rules `{c}` out: "
            + (f"by its definition ({REAL_DICTIONARY[key][c].rstrip('.').lower()}) " if c in REAL_DICTIONARY.get(key, {}) else "")
            + f"it describes what happens during or after the event being predicted, so it does not exist when the prediction is made. Its signal (univariate ROC-AUC {_auc(ev)}, "
            f"CV change {ev.cv_change:+.3f}) would not be available in production."
        )
        out.append((c, REAL_DICTIONARY.get(key, {}).get(c), ev, "exclude", why, "DCLAB-R04 (post-outcome leakage)", "high", "real_blocked"))
    for c in chosen:
        ev = compute_evidence(c, X[c].to_numpy(), y, base_X, base_auc, in_base=True)
        strong = ev.auc is not None and ev.auc >= 0.8
        why = (
            f"`{c}` is an original field of this dataset and the decision moment says such fields are recorded before "
            f"the prediction is made. "
            + (f"Its univariate ROC-AUC of {_auc(ev)} is high, but a strong score alone is a reason to review, not proof "
               f"of leakage, and nothing else points to leakage (no target copy"
               + (", no outcome-linked missingness)." if abs(ev.missing_pos - ev.missing_neg) < 0.1 else ").")
               if strong else
               f"The evidence shows no leakage pattern (univariate ROC-AUC {_auc(ev)}, CV change {ev.cv_change:+.3f}).")
        )
        rule = "DCLAB-R05 (high AUC alone does not prove leakage)" if strong else "DCLAB-R04 (checked: no leakage pattern)"
        out.append((c, None, ev, "keep", why, rule, "high", "real_available"))
    return out


def build_dataset(key, variants, max_rows, seed) -> tuple[list[dict], Counter]:
    rng = np.random.default_rng(seed)
    X, y, meta = load(key, max_rows, seed)
    blocked = [c for c in CATALOG[key].get("blocked", []) if c in X.columns]
    base_X = X.drop(columns=blocked)
    base_auc = cv_auc(base_X, y)
    safe_cols = list(base_X.columns)
    aucs = {c: (univariate_auc(X[c], y) or 0.5) for c in safe_cols}
    top_auc = max(aucs.values())
    top_cols = sorted(safe_cols, key=lambda c: -aucs[c])[:4]
    card = dataset_card(key, meta, len(X), float(y.mean()), top_auc)

    samples, drops = [], Counter()

    for c, desc, ev, action, why, rule, conf, gen in real_column_samples(key, X, y, meta, card, base_X, base_auc, rng):
        samples.append(build_sample(card, c, desc, ev, action, why, rule, conf,
                                    {"dataset": key, "generator": gen, "column": c, "action": action, "source": "real"}))

    makers = [
        gen_target_copy, gen_post_outcome_documented, gen_post_outcome_undocumented,
        gen_full_data_target_encoding, gen_sorted_id, gen_random_id, gen_future_date, gen_clean_date,
        lambda X, y, r: gen_missing_by_outcome(X, y, r, True),
        lambda X, y, r: gen_missing_by_outcome(X, y, r, False),
        gen_clean_noise,
        lambda X, y, r: gen_scary_name_clean(X, y, r, safe_cols),
        lambda X, y, r: gen_strong_legit(X, y, r, top_cols),
    ]
    for _ in range(variants):
        for make in makers:
            p: Plant = make(X, y, rng)
            p.extra.setdefault("top", top_auc)
            if p.name in X.columns:
                p.name = p.name + "_v2"
            ev = compute_evidence(p.name, p.values, y, base_X, base_auc, kind=p.kind)
            if not p.sane(ev, p):
                drops[p.generator] += 1
                continue
            samples.append(build_sample(card, p.name, p.description, ev, p.action, p.why(ev, p), p.rule, p.confidence,
                                        {"dataset": key, "generator": p.generator, "column": p.name,
                                         "action": p.action, "source": "planted"}))
    return samples, drops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(TARGETS))
    ap.add_argument("--variants", type=int, default=8, help="random variants per generator per dataset")
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-datasets", nargs="*", default=["wine_quality"])
    ap.add_argument("--test-datasets", nargs="*", default=["online_shoppers", "german_credit"])
    ap.add_argument("--out", type=Path, default=HERE / "out" / "leakage_audit")
    args = ap.parse_args()

    all_samples, all_drops, seen, dupes, leaks = [], Counter(), set(), 0, 0
    for i, key in enumerate(args.datasets):
        t = time.time()
        samples, drops = build_dataset(key, args.variants, args.max_rows, args.seed + i)
        all_drops.update(drops)
        kept = 0
        for s in samples:
            h = hashlib.sha256(s["messages"][1]["content"].encode()).hexdigest()
            if h in seen:
                dupes += 1
                continue
            if answer_leaks_into_input(s) or s["metadata"]["action"] not in ACTIONS:
                leaks += 1
                continue
            seen.add(h)
            all_samples.append(s)
            kept += 1
        print(f"{key:16s} {kept:4d} samples  ({sum(drops.values())} dropped by checker)  {time.time() - t:5.1f}s")

    split_of = {**{k: "val" for k in args.val_datasets}, **{k: "test" for k in args.test_datasets}}
    splits = {"train": [], "val": [], "test": []}
    for s in all_samples:
        splits[split_of.get(s["metadata"]["dataset"], "train")].append(s)
    random.Random(args.seed).shuffle(splits["train"])

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "total": len(all_samples),
        "splits": {k: len(v) for k, v in splits.items()},
        "split_datasets": {"val": args.val_datasets, "test": args.test_datasets},
        "by_action": dict(Counter(s["metadata"]["action"] for s in all_samples)),
        "by_generator": dict(Counter(s["metadata"]["generator"] for s in all_samples)),
        "by_action_per_split": {k: dict(Counter(s["metadata"]["action"] for s in v)) for k, v in splits.items()},
        "dropped_by_checker": dict(all_drops),
        "duplicates_removed": dupes,
        "answer_leak_rejections": leaks,
        "settings": {"variants": args.variants, "max_rows": args.max_rows, "seed": args.seed},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
