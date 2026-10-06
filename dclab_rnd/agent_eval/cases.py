"""The judgment suite's cases (package A4.1): small seeded tables, each with one planted trap and what a careful
policy should do about it.

A case says when the prediction is made (``moment``, given to every policy), which columns must be kept out of the
model (``leaks``), which must stay usable (``innocent``, which always includes the honest base inputs the table was
built from: keeping one out is a false alarm), and, for the split traps, which column must be declared as the time
or group column. The repeated-rows case checks the data stage itself (``engine_check``), not a policy.

Every table is generated from its seed, so a case is the same on every machine; ``fingerprint`` hashes the table
and the expectation so a changed case is noticed. The column names avoid the names of the R&D's own leakage
precedents (duration, Time, Rating…), which the audit matches by name, so a case measures the policy and not a
lucky name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

ROWS = 1500
BASE_INPUTS = ("age", "tenure_months", "plan", "support_calls", "monthly_spend")  # honest inputs of every base table
REGRESSION_MOMENT = "Each evening, for the next day's shipment, before it is weighed."


@dataclass(frozen=True)
class Case:
    id: str
    suite: str  # "leakage", "split" or "control"
    trap: str
    title: str
    build: Callable[[np.random.Generator], pd.DataFrame]
    seed: int
    moment: str = "At the monthly snapshot of each customer, before the outcome is known; what the customer answered at signup is known."
    leaks: tuple[str, ...] = ()  # must be kept out of the model
    innocent: tuple[str, ...] = ()  # must stay usable: keeping one out is a false alarm
    time_column: str | None = None  # must be declared, so the split follows time
    group_column: str | None = None  # must be declared (or the column kept out), so one entity stays on one side
    flag_duplicates: bool = False  # an engine check: the data stage must say that rows repeat (no policy can change it)
    note: str = ""  # why the expectation is what it is
    tags: tuple[str, ...] = field(default=())

    @property
    def clean_inputs(self) -> tuple[str, ...]:
        """Every column that must stay usable: the case's own innocent columns and the honest base inputs it kept."""
        frame = self.table()
        base = [c for c in BASE_INPUTS if c in frame.columns and c not in self.leaks and c not in (self.time_column, self.group_column)]
        return tuple(dict.fromkeys([*self.innocent, *base]))

    def table(self) -> pd.DataFrame:
        return self.build(np.random.default_rng(self.seed))

    def fingerprint(self) -> str:
        frame = self.table()
        body = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        expect = json.dumps([self.moment, self.leaks, self.innocent, self.time_column, self.group_column, self.flag_duplicates], sort_keys=True)
        return hashlib.sha256(body + expect.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------- base tables


def _base(rng: np.random.Generator, n: int = ROWS) -> pd.DataFrame:
    """A churn-like table: a few honest inputs and a binary ``target`` (about 30% positive) they partly explain."""
    age = rng.integers(18, 80, n)
    tenure = rng.integers(0, 120, n)
    plan = rng.choice(["basic", "plus", "pro"], n)
    calls = rng.poisson(3, n)
    spend = np.round(rng.gamma(2.0, 30.0, n), 2)
    logit = -1.1 + 0.02 * (calls - 3) * 3 - 0.012 * (tenure - 60) + (plan == "basic") * 0.6
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"age": age, "tenure_months": tenure, "plan": plan, "support_calls": calls, "monthly_spend": spend, "target": target})


def _base_regression(rng: np.random.Generator, n: int = ROWS) -> pd.DataFrame:
    """A demand-like table with a continuous ``target`` in kilograms."""
    frame = _base(rng, n).drop(columns="target")
    frame["target"] = np.round(40 + 0.5 * frame["tenure_months"] + 3 * frame["support_calls"] + rng.normal(0, 8, n), 2)
    return frame


def _flip(rng: np.random.Generator, y: pd.Series, share: float) -> np.ndarray:
    flips = rng.random(len(y)) < share
    return np.where(flips, 1 - y, y)


# ---------------------------------------------------------------------- the traps


def _post_outcome_named(rng):
    f = _base(rng)
    f["final_balance_after"] = np.round(f["monthly_spend"] * (1 - 0.9 * f["target"]) + rng.normal(0, 3, len(f)), 2)
    return f


def _post_outcome_status(rng):
    f = _base(rng)
    f["account_status_closed"] = _flip(rng, f["target"], 0.08)
    return f


def _post_outcome_neutral(rng):
    f = _base(rng)
    f["x_adj7"] = np.round(f["target"] * 2.0 + rng.normal(0, 0.6, len(f)), 4)  # continuous, no telling name
    return f


def _noisy_copy(rng):
    f = _base(rng)
    f["score_v2"] = _flip(rng, f["target"], 0.05)
    return f


def _noisy_copy_text(rng):
    f = _base(rng)
    f["label_text"] = np.where(_flip(rng, f["target"], 0.03) == 1, "left", "stayed")
    return f


def _identifier(rng):
    f = _base(rng).sort_values("target", kind="stable").reset_index(drop=True)
    f["customer_id"] = np.arange(100000, 100000 + len(f))  # assigned in target order: it predicts the target
    return f.sample(frac=1.0, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)


def _time_random_split(rng):
    f = _base(rng)
    days = np.sort(rng.integers(0, 730, len(f)))
    f["event_date"] = (pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d")
    drift = days / 730.0
    f["target"] = (rng.random(len(f)) < 0.15 + 0.35 * drift).astype(int)  # the rate moves with time
    return f


def _group_leak(rng):
    f = _base(rng)
    patients = rng.integers(0, 150, len(f))
    risk = rng.random(150)
    f["patient_ref"] = np.array([f"P{p:04d}" for p in patients])
    f["target"] = (rng.random(len(f)) < 0.1 + 0.7 * risk[patients]).astype(int)  # each patient has a lasting risk
    return f


def _duplicates(rng):
    f = _base(rng)
    copies = f.sample(n=300, random_state=int(rng.integers(1_000_000)))
    return pd.concat([f, copies], ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)


def _unit_conversion(rng):
    f = _base_regression(rng)
    f["weight_lb"] = np.round(f["target"] * 2.20462, 2)  # the target in pounds
    return f


def _sum_identity(rng):
    f = _base_regression(rng)
    f["morning_kg"] = np.round(f["target"] * rng.uniform(0.3, 0.7, len(f)), 2)
    f["evening_kg"] = np.round(f["target"] - f["morning_kg"], 2)  # morning + evening = the target
    return f


def _missing_is_leak(rng):
    f = _base(rng)
    value = np.round(rng.normal(50, 10, len(f)), 2)
    f["exit_survey_score"] = np.where(f["target"] == 1, np.nan, value)  # written only for customers who stayed
    return f


def _constant(rng):
    f = _base(rng)
    f["country"] = "NL"
    f["schema_version"] = 3
    return f


def _informative_missing(rng):
    f = _base(rng)
    income = np.round(rng.normal(40000, 9000, len(f)), 0)
    missing = rng.random(len(f)) < np.where(f["target"] == 1, 0.6, 0.45)  # missing more often for leavers: a real signal, known at signup
    f["declared_income"] = np.where(missing, np.nan, income)
    return f


def _benign_post_name(rng):
    f = _base(rng)
    f["post_signup_survey"] = rng.integers(1, 6, len(f))  # answered at signup, before the prediction: not a leak
    return f


def _strong_honest(rng):
    f = _base(rng)
    f["contract_months_left"] = np.round(np.clip(24 - f["tenure_months"] / 5 + rng.normal(0, 2, len(f)), 0, None), 1)
    f["target"] = (rng.random(len(f)) < 1 / (1 + np.exp(-(1.5 - 0.35 * f["contract_months_left"])))).astype(int)  # a strong, honest driver
    return f


def _clean(rng):
    return _base(rng)


def _clean_regression(rng):
    return _base_regression(rng)


def _account_label(rng):
    f = _base(rng)
    accounts = rng.integers(0, 300, len(f))
    label = rng.random(300) < 0.3
    f["account_ref"] = np.array([f"A{a:04d}" for a in accounts])
    f["target"] = label[accounts].astype(int)  # every row of an account shares its label
    return f


CASES: tuple[Case, ...] = (
    Case("AE-01", "leakage", "post_outcome_name", "A balance written after the outcome, with a telling name", _post_outcome_named, 11,
         leaks=("final_balance_after",)),
    Case("AE-02", "leakage", "post_outcome_name", "An account status that is the outcome itself", _post_outcome_status, 12, leaks=("account_status_closed",)),
    Case("AE-03", "leakage", "post_outcome_neutral", "A post-outcome measurement with a neutral name", _post_outcome_neutral, 13, leaks=("x_adj7",),
         note="Nothing in the name tells; only its strength against the target and the prediction moment do."),
    Case("AE-04", "leakage", "noisy_target_copy", "A copy of the target with 5% of labels flipped", _noisy_copy, 14, leaks=("score_v2",)),
    Case("AE-05", "leakage", "noisy_target_copy", "The target as text, with 3% flipped", _noisy_copy_text, 15, leaks=("label_text",)),
    Case("AE-06", "leakage", "identifier", "A customer number assigned in target order", _identifier, 16, leaks=("customer_id",)),
    Case("AE-07", "leakage", "unit_conversion", "The target in another unit (kg to lb)", _unit_conversion, 17, moment=REGRESSION_MOMENT, leaks=("weight_lb",)),
    Case("AE-08", "leakage", "unit_conversion", "Two columns that add up to the target", _sum_identity, 18, moment=REGRESSION_MOMENT, leaks=("morning_kg", "evening_kg")),
    Case("AE-09", "leakage", "missing_is_outcome", "A score that is missing exactly when the outcome happened", _missing_is_leak, 19, leaks=("exit_survey_score",),
         note="Missing only for leavers: the gap is written after the outcome."),
    Case("AE-10", "leakage", "group_label", "An account whose rows all share the label", _account_label, 20, group_column="account_ref",
         moment="At the monthly snapshot of each account's rows, before the outcome is known; the model must work for new accounts.",
         note="Splitting rows of one account across train and test lets the model learn the account; the account must be the group (or kept out)."),
    Case("AE-11", "split", "time_random_split", "Timestamped rows whose rate drifts, split at random", _time_random_split, 21, time_column="event_date",
         moment="On the day of each event, before the outcome is known; the model will score future days.",
         note="A random split puts the future in training; the date must be declared so the split follows time."),
    Case("AE-12", "split", "group_leakage", "Patients with many visits", _group_leak, 22, group_column="patient_ref",
         moment="At each visit, before the diagnosis is known; the model must work for patients it has not seen."),
    Case("AE-13", "split", "duplicates", "300 rows repeated, so copies land on both sides", _duplicates, 23, flag_duplicates=True),
    Case("AE-14", "control", "constant_column", "Two columns with one value each", _constant, 24, innocent=("country", "schema_version"),
         note="Useless, not leaking: forbidding them is a false alarm (dropping them as constant is fine and is not scored)."),
    Case("AE-15", "control", "informative_missing", "Income missing more often for leavers, known at signup", _informative_missing, 25, innocent=("declared_income",),
         note="The missingness is a real signal available at prediction time, not a leak."),
    Case("AE-16", "control", "benign_post_name", "A survey answered at signup, named post_signup_survey", _benign_post_name, 26, innocent=("post_signup_survey",),
         note="The name contains post_, but the value exists before the prediction."),
    Case("AE-17", "control", "strong_honest_driver", "A strong, honest predictor", _strong_honest, 27, innocent=("contract_months_left",),
         note="A strong signal alone is not leakage."),
    Case("AE-18", "control", "clean", "A clean classification table", _clean, 28, innocent=("age", "tenure_months", "plan", "support_calls", "monthly_spend")),
    Case("AE-19", "control", "clean", "A clean regression table", _clean_regression, 29, moment=REGRESSION_MOMENT, innocent=("age", "tenure_months", "plan", "support_calls", "monthly_spend")),
    Case("AE-20", "leakage", "post_outcome_name", "A refund column written after cancellation", lambda rng: _post_outcome_named(rng).rename(
        columns={"final_balance_after": "refund_issued_after"}), 30, leaks=("refund_issued_after",)),
)


def by_id(case_id: str) -> Case:
    return next(c for c in CASES if c.id == case_id)
