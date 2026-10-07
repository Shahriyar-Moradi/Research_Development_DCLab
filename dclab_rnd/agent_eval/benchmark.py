"""A frozen benchmark of 100 or more unseen cases (package 14.3).

The judgment suite (``cases.py``, 20 cases) was used while the prompts and the standard plan were built, so it can no
longer say how a policy does on something new. This benchmark is built the same way (a seeded table, a prediction
moment, at most one planted trap, the decisions a careful data scientist would accept, scored by ``run.score``) but
from base tables no prompt was tuned on, and it is split by source:

- **dev** (3 bases: claims, subscriptions, demand): for building and tuning, freely;
- **test** (6 bases: loans, readmissions, sensors, deliveries, and the diabetes and wine tables that ship with scikit-learn, real
  data and no download): **sealed**. Its fingerprints are committed in ``benchmark_sealed.json`` and a test fails when
  one changes; a run on it must write its result (``--output``), so every look is on record. Never change a prompt, a
  rule or a policy because of a test-set result: that turns it into a dev set (make a new version instead).

The traps are the families of ``docs/SYSTEM_EVALUATION.md`` that live in a table: a post-outcome column under a
telling or a neutral name, a noisy copy of the target (or the target in another unit, or as a sum, for a number), an
identifier assigned in target order, a value missing exactly when the outcome happened, rows of one entity sharing
the label, a time-ordered table split at random, repeated rows; and clean controls (a clean table, constant columns,
a strong honest driver, a harmless "post" name, missingness that is known at the prediction). On the two real
tables only the traps that keep the real target are planted. Column names avoid the R&D's own leakage precedents
(the audit matches those by name), so a case measures the policy, not a lucky name.

    python -m dclab_rnd.agent_eval.benchmark --split dev                        # scripted policies on the dev set
    python -m dclab_rnd.agent_eval.benchmark --split test --output evidence/campaigns/benchmark_v2/results/BEN-002_test_scripted.json
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .cases import Case

VERSION = "benchmark_v2"
SEALED = Path(__file__).with_name("benchmark_sealed.json")
ROWS = 1200


@dataclass(frozen=True)
class Base:
    key: str
    split: str  # "dev" or "test"
    task: str  # "binary" or "regression"
    moment: str
    build: Callable[[np.random.Generator], pd.DataFrame]  # honest inputs and a "target"
    names: dict[str, str] = field(default_factory=dict)  # the planted columns' names, chosen per base
    real: bool = False  # a real table: only the traps that keep its target


def _logistic(rng: np.random.Generator, logit: np.ndarray) -> np.ndarray:
    return (rng.random(len(logit)) < 1 / (1 + np.exp(-logit))).astype(int)


# ---------------------------------------------------------------------- base tables (dev)

def _claims(rng):
    n = ROWS
    f = pd.DataFrame({"policy_age_months": rng.integers(1, 180, n), "vehicle_age_years": rng.integers(0, 20, n),
                      "region": rng.choice(["north", "south", "east", "west"], n), "annual_premium": np.round(rng.gamma(3.0, 250.0, n), 2),
                      "prior_claims": rng.poisson(0.6, n)})
    f["target"] = _logistic(rng, -1.6 + 0.5 * f["prior_claims"] - 0.006 * f["policy_age_months"] + 0.04 * f["vehicle_age_years"])
    return f


def _subscriptions(rng):
    n = ROWS
    f = pd.DataFrame({"months_active": rng.integers(0, 60, n), "weekly_sessions": np.round(rng.gamma(2.0, 2.0, n), 1),
                      "tier": rng.choice(["free", "team", "business"], n), "seats": rng.integers(1, 50, n), "support_tickets": rng.poisson(1.5, n)})
    f["target"] = _logistic(rng, -0.4 - 0.18 * f["weekly_sessions"] - 0.02 * f["months_active"] + 0.2 * f["support_tickets"])
    return f


def _demand(rng):
    n = ROWS
    f = pd.DataFrame({"unit_price": np.round(rng.uniform(1.0, 9.0, n), 2), "on_promotion": rng.integers(0, 2, n),
                      "weekday": rng.choice(["mon", "tue", "wed", "thu", "fri", "sat", "sun"], n), "shelf_metres": np.round(rng.uniform(0.5, 4.0, n), 1),
                      "forecast_temp_c": np.round(rng.normal(15, 7, n), 1)})
    f["target"] = np.round(120 - 9 * f["unit_price"] + 25 * f["on_promotion"] + 12 * f["shelf_metres"] + 0.8 * f["forecast_temp_c"] + rng.normal(0, 10, n), 1)
    return f


# ---------------------------------------------------------------------- base tables (sealed test)

def _loans(rng):
    n = ROWS
    f = pd.DataFrame({"income_k": np.round(rng.gamma(4.0, 12.0, n), 1), "amount_k": np.round(rng.gamma(2.0, 8.0, n), 1),
                      "term_months": rng.choice([12, 24, 36, 60], n), "bureau_score": rng.integers(450, 850, n),
                      "years_employed": rng.integers(0, 35, n), "purpose": rng.choice(["car", "home", "study", "other"], n)})
    f["target"] = _logistic(rng, -1.0 + 0.04 * (f["amount_k"] - f["income_k"] / 3) - 0.008 * (f["bureau_score"] - 650))
    return f


def _readmissions(rng):
    n = ROWS
    f = pd.DataFrame({"patient_age": rng.integers(18, 95, n), "nights_admitted": rng.integers(1, 21, n), "prior_admissions": rng.poisson(1.0, n),
                      "condition_group": rng.choice(["cardio", "resp", "renal", "other"], n), "medications": rng.integers(0, 15, n)})
    f["target"] = _logistic(rng, -2.0 + 0.35 * f["prior_admissions"] + 0.06 * f["nights_admitted"] + 0.012 * (f["patient_age"] - 60))
    return f


def _sensors(rng):
    n = ROWS
    f = pd.DataFrame({"mean_temp_c": np.round(rng.normal(60, 8, n), 1), "vibration_rms": np.round(rng.gamma(2.0, 0.6, n), 3),
                      "hours_since_service": rng.integers(0, 4000, n), "machine_model": rng.choice(["m1", "m2", "m3"], n), "load_pct": rng.integers(20, 100, n)})
    f["target"] = _logistic(rng, -3.0 + 0.9 * f["vibration_rms"] + 0.0005 * f["hours_since_service"] + 0.03 * (f["mean_temp_c"] - 60))
    return f


def _deliveries(rng):
    n = ROWS
    f = pd.DataFrame({"distance_km": np.round(rng.gamma(2.0, 6.0, n), 1), "parcel_kg": np.round(rng.gamma(1.5, 2.0, n), 2),
                      "dispatch_weekday": rng.choice(["mon", "tue", "wed", "thu", "fri", "sat"], n), "carrier": rng.choice(["c1", "c2", "c3"], n),
                      "stops_on_route": rng.integers(3, 40, n)})
    f["target"] = _logistic(rng, -2.2 + 0.05 * f["distance_km"] + 0.05 * f["stops_on_route"] + (f["carrier"] == "c3") * 0.5)
    return f


def _diabetes(rng):
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)  # 442 patients, bundled with scikit-learn: no download
    return data.frame.rename(columns={"s1": "tc", "s2": "ldl", "s3": "hdl", "s4": "tch", "s5": "ltg", "s6": "glu"})


def _wine(rng):
    from sklearn.datasets import load_wine

    data = load_wine(as_frame=True)  # 178 wines, bundled with scikit-learn: no download
    f = data.frame.rename(columns=lambda c: c.replace("/", "_per_").replace(" ", "_"))
    f["target"] = (f["target"] == 0).astype(int)  # the first cultivar or not
    return f


BASES: tuple[Base, ...] = (
    Base("claims", "dev", "binary", "When a claim is filed, before any investigation; the outcome is whether it proves fraudulent.", _claims,
         {"post": "investigation_cost_after", "neutral": "z_rk4", "copy": "adjuster_flag", "id": "claim_no", "missing": "recovery_score",
          "entity": "policy_holder", "date": "filed_on", "benign": "post_code_zone", "strong": "prior_fraud_score"}),
    Base("subscriptions", "dev", "binary", "On the first day of each month, from the account as it stands; the outcome is cancelling within the month.", _subscriptions,
         {"post": "refund_paid_after", "neutral": "m_q9", "copy": "cancel_reason_code", "id": "account_no", "missing": "exit_rating",
          "entity": "company_ref", "date": "snapshot_day", "benign": "postal_region", "strong": "seats_unused_share"}),
    Base("demand", "dev", "regression", "Each evening, for the next day's sales of the item, before the store opens.", _demand,
         {"post": "units_returned_next_week", "neutral": "v_77", "copy": "sold_dozens", "parts": ("sold_morning", "sold_afternoon"), "id": "row_seq",
          "entity": "store_ref", "date": "sales_day", "benign": "post_holiday_week", "strong": "planned_display_units"}),
    Base("loans", "test", "binary", "When the application is submitted, before the decision; the outcome is default within two years.", _loans,
         {"post": "collections_fee_charged", "neutral": "w_c2", "copy": "watchlist_status", "id": "application_no", "missing": "repayment_survey",
          "entity": "borrower_ref", "date": "applied_on", "benign": "post_box_used", "strong": "debt_to_income"}),
    Base("readmissions", "test", "binary", "On the day of discharge, before the patient leaves; the outcome is readmission within 30 days.", _readmissions,
         {"post": "followup_bed_days", "neutral": "q_h5", "copy": "return_visit_code", "id": "admission_no", "missing": "home_call_score",
          "entity": "patient_no", "date": "discharged_on", "benign": "post_op_ward", "strong": "frailty_index"}),
    Base("sensors", "test", "binary", "At the end of each shift, from that shift's readings; the outcome is a breakdown in the next 72 hours.", _sensors,
         {"post": "repair_parts_cost", "neutral": "s_x3", "copy": "fault_ticket", "id": "reading_no", "missing": "inspection_grade",
          "entity": "machine_ref", "date": "shift_date", "benign": "post_maintenance_mode", "strong": "bearing_wear_index"}),
    Base("deliveries", "test", "binary", "When the parcel is handed to the carrier; the outcome is arriving later than promised.", _deliveries,
         {"post": "late_compensation_paid", "neutral": "d_p8", "copy": "complaint_code", "id": "parcel_no", "missing": "delivery_rating_given",
          "entity": "customer_ref", "date": "dispatched_on", "benign": "post_office_pickup", "strong": "route_congestion_index"}),
    Base("diabetes", "test", "regression", "At the baseline visit; the outcome is the disease's progression measured one year later.", _diabetes,
         {"post": "progression_followup_score", "neutral": "k_31", "copy": "progression_pct", "parts": ("prog_part_a", "prog_part_b"), "id": "visit_seq",
          "benign": "post_fast_glucose_flag"}, real=True),
    Base("wine", "test", "binary", "When the sample is analysed in the lab, before its cultivar is recorded.", _wine,
         {"post": "cellar_label_assigned", "neutral": "r_19", "copy": "label_code", "id": "sample_no", "missing": "tasting_note_score",
          "benign": "post_ferment_check"}, real=True),
)


# ---------------------------------------------------------------------- the traps, on any base

def _flip(rng, y, share):
    return np.where(rng.random(len(y)) < share, 1 - y, y)


def _post_outcome(base: Base, rng, f):
    y = f["target"].to_numpy(dtype=float)
    scale = f.drop(columns="target").select_dtypes("number").iloc[:, 0].abs().to_numpy() + 1.0
    if base.task == "binary":
        f[base.names["post"]] = np.round(scale * (0.1 + 0.9 * y) + rng.normal(0, 0.05 * scale.mean(), len(f)), 2)
    else:
        f[base.names["post"]] = np.round(y * rng.uniform(0.05, 0.15, len(f)), 2)  # written after the outcome, in proportion to it
    return f


def _neutral(base: Base, rng, f):
    y = f["target"].to_numpy(dtype=float)
    spread = 1.0 if base.task == "binary" else max(float(np.std(y)), 1.0)
    f[base.names["neutral"]] = np.round(y * 2.0 / spread + rng.normal(0, 0.6, len(f)), 4)
    return f


def _copy(base: Base, rng, f):
    if base.task == "binary":
        f[base.names["copy"]] = _flip(rng, f["target"].to_numpy(), 0.05)
    else:
        f[base.names["copy"]] = np.round(f["target"] / 12.0, 3)  # the target in other units
    return f


def _sum(base: Base, rng, f):
    a, b = base.names["parts"]
    f[a] = np.round(f["target"] * rng.uniform(0.3, 0.7, len(f)), 2)
    f[b] = np.round(f["target"] - f[a], 2)  # the two parts add up to the target
    return f


def _identifier(base: Base, rng, f):
    f = f.sort_values("target", kind="stable").reset_index(drop=True)
    f[base.names["id"]] = np.arange(500000, 500000 + len(f))  # assigned in target order
    return f.sample(frac=1.0, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)


def _missing(base: Base, rng, f):
    value = np.round(rng.normal(60, 12, len(f)), 2)
    f[base.names["missing"]] = np.where(f["target"] == 1, np.nan, value)  # written only when the outcome did not happen
    return f


def _entity(base: Base, rng, f):
    groups = max(len(f) // 5, 20)
    who = rng.integers(0, groups, len(f))
    f[base.names["entity"]] = np.array([f"E{g:05d}" for g in who])
    if base.task == "binary":
        label = rng.random(groups) < float(f["target"].mean())
        f["target"] = label[who].astype(int)  # every row of an entity shares its label
    else:
        f["target"] = np.round(f["target"] + rng.normal(0, 3 * f["target"].std(), groups)[who], 1)  # each entity has its own level
    return f


def _time(base: Base, rng, f):
    days = np.sort(rng.integers(0, 730, len(f)))
    f[base.names["date"]] = (pd.Timestamp("2022-06-01") + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d")
    drift = days / 730.0
    if base.task == "binary":
        f["target"] = (rng.random(len(f)) < 0.12 + 0.4 * drift).astype(int)  # the rate moves with time
    else:
        f["target"] = np.round(f["target"] + 60 * drift, 1)
    return f


def _duplicates(base: Base, rng, f):
    copies = f.sample(n=max(len(f) // 5, 30), random_state=int(rng.integers(1_000_000)))
    return pd.concat([f, copies], ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)


def _constant(base: Base, rng, f):
    f["source_system"] = "erp"
    f["export_version"] = 7
    return f


def _benign(base: Base, rng, f):
    f[base.names["benign"]] = rng.integers(1, 6, len(f))  # known before the prediction, whatever the name says
    return f


def _strong(base: Base, rng, f):
    honest = rng.normal(0, 1, len(f))
    f[base.names["strong"]] = np.round(honest, 3)
    if base.task == "binary":
        f["target"] = _logistic(rng, -0.8 + 2.2 * honest)  # a strong driver that exists at the prediction
    else:
        f["target"] = np.round(f["target"] + 25 * honest, 1)
    return f


def _informative_missing(base: Base, rng, f):
    value = np.round(rng.normal(40, 9, len(f)), 1)
    gap = rng.random(len(f)) < np.where(f["target"] == 1, 0.55, 0.4)  # missing more often for one class: a real signal, known at the time
    f["declared_extra"] = np.where(gap, np.nan, value)
    return f


@dataclass(frozen=True)
class Trap:
    key: str
    suite: str
    title: str
    plant: Callable
    tasks: tuple[str, ...] = ("binary", "regression")
    real_ok: bool = True  # keeps the real target
    leaks: Callable[[Base], tuple[str, ...]] = lambda b: ()
    innocent: Callable[[Base], tuple[str, ...]] = lambda b: ()
    time: bool = False
    group: bool = False
    dups: bool = False


TRAPS: tuple[Trap, ...] = (
    Trap("post_outcome_name", "leakage", "A column written after the outcome, with a telling name", _post_outcome, leaks=lambda b: (b.names["post"],)),
    Trap("post_outcome_neutral", "leakage", "A post-outcome measurement with a neutral name", _neutral, leaks=lambda b: (b.names["neutral"],)),
    Trap("noisy_target_copy", "leakage", "A copy of the target with 5% of labels flipped", _copy, tasks=("binary",), leaks=lambda b: (b.names["copy"],)),
    Trap("unit_conversion", "leakage", "The target in other units", _copy, tasks=("regression",), leaks=lambda b: (b.names["copy"],)),
    Trap("sum_identity", "leakage", "Two columns that add up to the target", _sum, tasks=("regression",), leaks=lambda b: b.names["parts"]),
    Trap("identifier", "leakage", "A number assigned in target order", _identifier, leaks=lambda b: (b.names["id"],)),
    Trap("missing_is_outcome", "leakage", "A value missing exactly when the outcome happened", _missing, tasks=("binary",), leaks=lambda b: (b.names["missing"],)),
    Trap("group_label", "split", "Rows of one entity share the outcome", _entity, real_ok=False, group=True),
    Trap("time_random_split", "split", "Rows over two years whose outcome drifts, split at random", _time, real_ok=False, time=True),
    Trap("duplicates", "split", "A fifth of the rows repeated", _duplicates, dups=True),
    Trap("clean", "control", "A clean table", lambda b, rng, f: f),
    Trap("constant_column", "control", "Two columns with one value each", _constant, innocent=lambda b: ("source_system", "export_version")),
    Trap("benign_post_name", "control", "A value known at the prediction, with post in its name", _benign, innocent=lambda b: (b.names["benign"],)),
    Trap("strong_honest_driver", "control", "A strong driver that exists at the prediction", _strong, real_ok=False, innocent=lambda b: (b.names["strong"],)),
    Trap("informative_missing", "control", "Missing more often for one outcome, known at the prediction", _informative_missing, tasks=("binary",),
         real_ok=False, innocent=lambda b: ("declared_extra",)),
)


NEEDS = {"post_outcome_name": ("post",), "post_outcome_neutral": ("neutral",), "noisy_target_copy": ("copy",), "unit_conversion": ("copy",),
         "sum_identity": ("parts",), "identifier": ("id",), "missing_is_outcome": ("missing",), "group_label": ("entity",), "time_random_split": ("date",),
         "benign_post_name": ("benign",), "strong_honest_driver": ("strong",)}  # the names a trap plants


def _seed(case_id: str) -> int:
    return int(hashlib.sha256(f"{VERSION}:{case_id}".encode()).hexdigest()[:8], 16)


def _builder(base: Base, trap: Trap) -> Callable[[np.random.Generator], pd.DataFrame]:
    def build(rng: np.random.Generator) -> pd.DataFrame:
        return trap.plant(base, rng, base.build(rng).copy())
    return build


def _honest(base: Base) -> tuple[str, ...]:
    return tuple(c for c in base.build(np.random.default_rng(0)).columns if c != "target")


def build_cases() -> tuple[Case, ...]:
    out = []
    for base in BASES:
        honest = _honest(base)
        for trap in TRAPS:
            if base.task not in trap.tasks or (base.real and not trap.real_ok):
                continue
            if any(k not in base.names for k in NEEDS.get(trap.key, ())):
                continue  # a trap this base has no column name for (the real tables have no entity or date)
            cid = f"BV2-{base.key[:4].upper()}-{trap.key}"
            entity = base.names.get("entity") if trap.group else None
            date = base.names.get("date") if trap.time else None
            out.append(Case(cid, trap.suite, trap.key, f"{base.key}: {trap.title}", _builder(base, trap), _seed(cid), moment=base.moment,
                            leaks=tuple(trap.leaks(base)), innocent=tuple(dict.fromkeys([*trap.innocent(base), *honest])),
                            time_column=date, group_column=entity, flag_duplicates=trap.dups, tags=(base.split, base.key, base.task)))
    return tuple(out)


CASES: tuple[Case, ...] = build_cases()


def split(name: str) -> tuple[Case, ...]:
    return tuple(c for c in CASES if c.tags[0] == name)


def sealed_fingerprints() -> dict[str, str]:
    return {c.id: c.fingerprint() for c in split("test")}


def write_sealed() -> None:
    """Freeze the test set: only when the benchmark is made or versioned, never to make a failing check pass."""
    SEALED.write_text(json.dumps({"version": VERSION, "cases": sealed_fingerprints()}, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def run(split_name: str, policies: tuple[str, ...] = ("standard", "audit", "bad")) -> dict:
    """Score the policies on one split. Only the command line (``main``, ``make benchmark``) records a run on the
    sealed set; call this on "test" only from there."""
    import importlib

    suite = importlib.import_module("dclab_rnd.agent_eval.run")  # the package exports a function named run

    cases = split(split_name)
    report = suite.run(policies, cases)
    report.update({"campaign_id": VERSION, "kind": "benchmark_scripted", "suite": f"{VERSION}:{split_name}", "prompt_hashes": None, "split": split_name,
                   "sealed": split_name == "test",
                   "question": "On unseen seeded tables with at most one planted trap each, which policies keep the leak out of the model and keep clean columns usable?",
                   "by_family": suite.by_trap_families(report["results"], {c.id: c.trap for c in cases}),
                   "limitations": [
                       "Scripted policies on seeded, planted traps: evidence about these policies on these cases, not about any model's readiness for production.",
                       "The labels are the benchmark author's; package 14.4 has them reviewed by two people who did not write them.",
                       "A leak counts as caught when the column is kept out of the model; the reason given is not checked.",
                       "Cases built from one base share its columns and its generator: they are not independent of each other.",
                       "In the missing_is_outcome family two columns carry a telling name (exit_rating, delivery_rating_given): the audit "
                       "keeps them out by that name, not by the missingness, which it does not detect. Renaming them would be tuning on a "
                       "set that has been scored, so they stay until a new version.",
                   ]})
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    import re

    import importlib

    suite = importlib.import_module("dclab_rnd.agent_eval.run")  # the package exports a function named run

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.benchmark", description="Run the frozen benchmark (14.3) with scripted policies.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--policy", action="append", choices=sorted(suite.POLICIES))
    parser.add_argument("--output", type=Path, help="write the report as JSON to this new file (never overwritten); required for the sealed test set")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():  # checked before the run: a look at the sealed set always leaves a record
        parser.error(f"{args.output} exists; results are never overwritten, choose a new BEN-<n> file")
    if args.split == "test" and not args.output:
        parser.error("a run on the sealed test set is always recorded: give --output evidence/campaigns/benchmark_v2/results/BEN-<n>_test_….json")
    report = run(args.split, tuple(args.policy or ("standard", "audit", "bad")))
    for policy, s in report["summary"].items():
        print(f"{policy:9s} leaks caught {s['leaks_caught'][0]}/{s['leaks_caught'][1]} (95% {s['leaks_caught_ci95']})  "
              f"false alarms {s['false_alarms'][0]}/{s['false_alarms'][1]} (95% {s['false_alarm_ci95']})  errors {s['errors']}")
    print(report["limitations"][0])
    if args.output:
        found = re.match(r"([A-Z]+-\d+)_", args.output.name)
        report = {"experiment_id": found.group(1) if found else args.output.stem, **report}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
