"""Benchmark v3: the frozen benchmark with rare-event bases and traps (package 15.9).

Adding a trap changes the sealed set, so this is a new version beside ``benchmark.py`` (v2), which stays as it is with
its fingerprints and its results. v3 keeps v2's generated bases and traps on new seeds (every table differs; the two
real scikit-learn tables are left out, since reseeding cannot change them), adds rare-event bases (a positive class
of about 1 to 3 percent) and four rare-event traps, plus a control:

- **label_by_key**: a column that flags every row of a key once any row of that key is a case (labels spread over a
  coarse key, as a label file joined on the account rather than the payment does): a leak;
- **partial_labels**: only the cases someone found are marked, and the moment says so; the solution must call the
  labels partial (an unlabelled row is unknown, not a negative);
- **rare_entity_cluster**: the rare cases sit on a few entities; the entity must be declared as the group, so one
  entity never sits on both sides of the split (keeping it out does not count here: see the control below);
- **oversampled_positives**: the positive rows were copied several times before the table was saved (oversampling
  before a split or a detector): the data stage must say that rows repeat;
- **unclustered_entity** (a control): a many-valued entity column with no link to the outcome must stay usable (on a
  rare target any many-valued column separates the few positives, so keeping it out would be a false alarm).

The partial-label sentence differs per base (each written once, before the freeze, never checked against a rule), and
the two v2 traps that overwrite the target (a time drift, a strong driver) are not planted on rare bases.

Split by source, like v2: **dev** = v2's dev bases and the ``payments`` rare base; **test** (sealed) = v2's test bases
on new seeds and three rare bases no prompt or rule has seen (``returns``, ``warranty``, ``promotions``). The test
fingerprints are committed in ``benchmark_v3_sealed.json``; a run on the test set is always recorded; never change a
prompt, a rule or a policy because of a test-set result (make benchmark_v4 instead). v2's test generators were scored
before (BEN-002 to BEN-005), so v3 is blindest on its rare bases: results are reported per base as well.

    python -m dclab_rnd.agent_eval.benchmark_v3 --split dev
    python -m dclab_rnd.agent_eval.benchmark_v3 --split test --output evidence/campaigns/benchmark_v3/results/BEN3-002_test_scripted.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import benchmark as v2
from .benchmark import Base, Trap, _logistic
from .cases import Case

VERSION = "benchmark_v3"
SALT = "frozen-2026-10-13"  # added before the freeze, after a reviewer looked at the audit's answers on six earlier tables
SEALED = Path(__file__).with_name("benchmark_v3_sealed.json")
ROWS = 1500  # rare cases need a few more rows to hold a handful of positives


# ---------------------------------------------------------------------- rare-event base tables

def _rare(rng, logit_shift: float, f: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    f["target"] = _logistic(rng, logit_shift + score)
    return f


def _payments(rng):  # dev
    n = ROWS
    f = pd.DataFrame({"amount": np.round(rng.gamma(2.0, 40.0, n), 2), "hour_of_day": rng.integers(0, 24, n),
                      "card_age_days": rng.integers(1, 2000, n), "merchant_group": rng.choice(["food", "travel", "games", "retail"], n),
                      "foreign_card": rng.integers(0, 2, n)})
    return _rare(rng, -5.2, f, 0.008 * f["amount"] + 0.9 * f["foreign_card"] - 0.0008 * f["card_age_days"] + (f["merchant_group"] == "games") * 0.8)


def _returns(rng):  # sealed
    n = ROWS
    f = pd.DataFrame({"order_value": np.round(rng.gamma(2.5, 30.0, n), 2), "days_to_return": rng.integers(0, 60, n),
                      "items_in_order": rng.integers(1, 12, n), "category": rng.choice(["shoes", "phones", "toys", "home"], n),
                      "account_age_days": rng.integers(1, 3000, n)})
    return _rare(rng, -3.9, f, 0.01 * f["order_value"] - 0.0007 * f["account_age_days"] + (f["category"] == "phones") * 0.9 - 0.02 * f["days_to_return"])


def _warranty(rng):  # sealed
    n = ROWS
    f = pd.DataFrame({"product_age_months": rng.integers(0, 36, n), "repair_cost_estimate": np.round(rng.gamma(2.0, 60.0, n), 2),
                      "prior_claims_on_serial": rng.poisson(0.3, n), "channel": rng.choice(["store", "online", "partner"], n),
                      "photos_attached": rng.integers(0, 6, n)})
    return _rare(rng, -4.4, f, 0.004 * f["repair_cost_estimate"] + 0.7 * f["prior_claims_on_serial"] - 0.25 * f["photos_attached"]
                 + (f["channel"] == "partner") * 0.6)


def _promotions(rng):  # sealed
    n = ROWS
    f = pd.DataFrame({"coupon_value": np.round(rng.uniform(2, 50, n), 1), "signup_age_hours": rng.integers(0, 2000, n),
                      "devices_seen": rng.integers(1, 6, n), "payment_type": rng.choice(["card", "wallet", "voucher"], n),
                      "basket_items": rng.integers(1, 15, n)})
    return _rare(rng, -4.6, f, 0.03 * f["coupon_value"] - 0.002 * f["signup_age_hours"] + 0.5 * f["devices_seen"])


RARE_BASES: tuple[Base, ...] = (
    Base("payments", "dev", "binary", "When the payment is authorised, before anyone reviews it; the outcome is a confirmed fraud.", _payments,
         {"post": "chargeback_fee_billed", "neutral": "p_t6", "copy": "dispute_code", "id": "payment_no", "missing": "cardholder_reply_score",
          "entity": "card_ref", "date": "authorised_on", "benign": "post_auth_hold_hours", "strong": "device_risk_index",
          "keyflag": "card_flagged", "free_entity": "terminal_ref",
          "partial": "The labels are the cases the review team confirmed; most rows were never checked, so an unmarked row may be a case nobody found."}),
    Base("returns", "test", "binary", "When the return is requested, before the parcel comes back; the outcome is a confirmed abusive return.", _returns,
         {"post": "restocking_loss_booked", "neutral": "r_b4", "copy": "abuse_code", "id": "return_no", "missing": "inspection_score",
          "entity": "customer_ref", "date": "requested_on", "benign": "post_purchase_survey_done", "strong": "return_rate_90d",
          "keyflag": "customer_flagged", "free_entity": "courier_ref",
          "partial": "Inspectors open only a few parcels, so the abuse label exists only for those; the rest were simply not inspected."}),
    Base("warranty", "test", "binary", "When the warranty claim is filed, before the technician sees the product; the outcome is a confirmed false claim.",
         _warranty,
         {"post": "technician_hours_billed", "neutral": "w_k1", "copy": "denial_code", "id": "claim_ref_no", "missing": "technician_report_score",
          "entity": "serial_ref", "date": "filed_on_day", "benign": "post_sale_registration", "strong": "serial_claim_rate",
          "keyflag": "serial_flagged", "free_entity": "repair_shop_ref",
          "partial": "Fraud is marked only where the technician raised a dispute, which most claims never get; a claim without a mark is not known to be honest."}),
    Base("promotions", "test", "binary", "When the coupon is redeemed, before the order ships; the outcome is a confirmed promotion abuse.", _promotions,
         {"post": "clawback_amount", "neutral": "m_z2", "copy": "abuse_flag_code", "id": "redemption_no", "missing": "kyc_check_score",
          "entity": "household_ref", "date": "redeemed_on", "benign": "post_signup_email_opened", "strong": "linked_accounts",
          "keyflag": "household_flagged", "free_entity": "store_ref",
          "partial": "The abuse flags come from a sample audit of redemptions, so the labels are incomplete: most redemptions were not looked at."}),
)
BASES: tuple[Base, ...] = (*(b for b in v2.BASES if not b.real), *RARE_BASES)
RARE_KEYS = tuple(b.key for b in RARE_BASES)


# ---------------------------------------------------------------------- rare-event traps

def _keys(f: pd.DataFrame, rng, per_key: int = 6) -> np.ndarray:
    return rng.integers(0, max(len(f) // per_key, 20), len(f))


def _label_by_key(base: Base, rng, f):
    key = _keys(f, rng)
    hit = pd.Series(f["target"].to_numpy()).groupby(key).transform("max").to_numpy()
    f[base.names["keyflag"]] = hit.astype(int)  # 1 on every row of a key with any case: the labels, spread over the key
    return f


def _partial_labels(base: Base, rng, f):
    y = f["target"].to_numpy()
    found = rng.random(len(f)) < 0.4
    f["target"] = np.where((y == 1) & found, 1, 0)  # only the cases someone found are marked
    return f


def _rare_entity_cluster(base: Base, rng, f):
    entities = max(len(f) // 6, 40)
    who = rng.integers(0, entities, len(f))
    risky = rng.random(entities) < 0.04  # a few entities hold the cases
    f[base.names["entity"]] = np.array([f"K{e:05d}" for e in who])
    f["target"] = (rng.random(len(f)) < np.where(risky[who], 0.35, 0.003)).astype(int)
    return f


def _unclustered_entity(base: Base, rng, f):
    f[base.names["free_entity"]] = np.array([f"T{e:05d}" for e in rng.integers(0, max(len(f) // 6, 40), len(f))])  # no link to the outcome
    return f


def _oversampled(base: Base, rng, f):
    positives = f[f["target"] == 1]
    copies = pd.concat([positives] * 6, ignore_index=True) if len(positives) else positives
    return pd.concat([f, copies], ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)


RARE_TRAPS: tuple[Trap, ...] = (
    Trap("label_by_key", "leakage", "A flag set on every row of a key once any of its rows is a case", _label_by_key, tasks=("binary",),
         real_ok=False, leaks=lambda b: (b.names["keyflag"],)),
    Trap("partial_labels", "labels", "Only the cases someone found are marked", _partial_labels, tasks=("binary",), real_ok=False),
    Trap("rare_entity_cluster", "split", "The rare cases sit on a few entities", _rare_entity_cluster, tasks=("binary",), real_ok=False, group=True),
    Trap("unclustered_entity", "control", "A many-valued entity column with no link to the outcome", _unclustered_entity, tasks=("binary",),
         real_ok=False, innocent=lambda b: (b.names["free_entity"],)),
    Trap("oversampled_positives", "split", "The positive rows were copied before the table was saved", _oversampled, tasks=("binary",),
         real_ok=False, dups=True),
)
NEEDS = {**v2.NEEDS, "label_by_key": ("keyflag",), "rare_entity_cluster": ("entity",), "unclustered_entity": ("free_entity",),
         "partial_labels": ("partial",)}
NOT_ON_RARE = ("time_random_split", "strong_honest_driver")  # they overwrite the target: the case would not be a rare event


def _seed(case_id: str) -> int:
    return int(hashlib.sha256(f"{VERSION}:{SALT}:{case_id}".encode()).hexdigest()[:8], 16)


def _builder(base: Base, trap: Trap) -> Callable[[np.random.Generator], pd.DataFrame]:
    def build(rng: np.random.Generator) -> pd.DataFrame:
        return trap.plant(base, rng, base.build(rng).copy())
    return build


def build_cases() -> tuple[Case, ...]:
    out = []
    for base in BASES:
        honest = v2._honest(base)
        rare = base.key in RARE_KEYS
        for trap in (*v2.TRAPS, *(RARE_TRAPS if rare else ())):
            if base.task not in trap.tasks or (base.real and not trap.real_ok) or (rare and trap.key in NOT_ON_RARE):
                continue
            if any(k not in base.names for k in NEEDS.get(trap.key, ())):
                continue
            cid = f"BV3-{base.key[:4].upper()}-{trap.key}"
            moment = base.moment + (" " + base.names["partial"] if trap.key == "partial_labels" else "")
            out.append(Case(cid, trap.suite, trap.key, f"{base.key}: {trap.title}", _builder(base, trap), _seed(cid), moment=moment,
                            leaks=tuple(trap.leaks(base)), innocent=tuple(dict.fromkeys([*trap.innocent(base), *honest])),
                            time_column=base.names.get("date") if trap.time else None,
                            group_column=base.names.get("entity") if trap.group else None, flag_duplicates=trap.dups,
                            labels="partial" if trap.key == "partial_labels" else None, strict_group=trap.key == "rare_entity_cluster",
                            tags=(base.split, base.key, base.task, "rare" if rare else "common")))
    return tuple(out)


CASES: tuple[Case, ...] = build_cases()


def split(name: str) -> tuple[Case, ...]:
    return tuple(c for c in CASES if c.tags[0] == name)


def sealed_fingerprints() -> dict[str, str]:
    return {c.id: c.fingerprint() for c in split("test")}


def write_sealed() -> None:
    """Freeze the test set: once, when v3 is made; never to make a failing check pass."""
    SEALED.write_text(json.dumps({"version": VERSION, "cases": sealed_fingerprints()}, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def run(split_name: str, policies: tuple[str, ...] = ("standard", "audit", "bad")) -> dict:
    """Score the policies on one split; only the command line records a run on the sealed set."""
    import importlib

    suite = importlib.import_module("dclab_rnd.agent_eval.run")
    cases = split(split_name)
    report = suite.run(policies, cases)
    by_base: dict[str, dict] = {}
    alarms_by_base: dict[str, dict] = {}
    for policy in policies:
        for case in cases:
            row = next((r for r in report["results"] if r["case"] == case.id and r["policy"] == policy), None)
            if row is None:
                continue
            if row.get("leak_caught") is not None:
                cell = by_base.setdefault(case.tags[1], {}).setdefault(policy, [0, 0])
                cell[0] += int(bool(row["leak_caught"]))
                cell[1] += 1
            if row.get("false_alarm") is not None:
                cell = alarms_by_base.setdefault(case.tags[1], {}).setdefault(policy, [0, 0])
                cell[0] += int(bool(row["false_alarm"]))
                cell[1] += 1
    report.update({"campaign_id": VERSION, "kind": "benchmark_scripted", "suite": f"{VERSION}:{split_name}", "prompt_hashes": None,
                   "split": split_name, "sealed": split_name == "test",
                   "question": "On unseen seeded tables, rare-event ones included, which policies keep the leak out, declare the entity, "
                               "call partial labels partial and keep clean columns usable?",
                   "by_family": suite.by_trap_families(report["results"], {c.id: c.trap for c in cases}),
                   "by_base_traps_caught": by_base, "by_base_false_alarms": alarms_by_base, "code": _provenance(),
                   "limitations": [
                       "Scripted policies on seeded, planted traps: evidence about these policies on these cases, not about any model's readiness for production.",
                       "v2's test generators were scored before (BEN-002 to BEN-005); v3 reseeds them, but only the rare bases are new to every rule.",
                       "A leak counts as caught when the column is kept out of the model; the reason given is not checked. Partial labels count "
                       "as caught when the solution says labels are partial.",
                       "Cases built from one base share its columns and its generator: they are not independent of each other.",
                   ]})
    return report


GUARDED = ("dclab_rnd/agent_eval/benchmark_v3_sealed.json", "dclab_rnd/agent_eval/benchmark_v3.py", "dclab_rnd/agent_eval/benchmark.py",
           "dclab_rnd/agent_eval/cases.py", "dclab_rnd/agent_eval/run.py")


def _git(*args: str) -> tuple[int, str]:
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return done.returncode, done.stdout.strip()


def _provenance() -> dict:
    """What the run ran on: the commit, whether the tree was clean, and the prompt and tool fingerprints."""
    from dclab_rnd import prompts

    code, head = _git("rev-parse", "HEAD")
    dirty, _ = _git("diff", "--quiet", "HEAD", "--")
    return {"commit": head if code == 0 else None, "clean_tree": dirty == 0, "prompt_hashes": prompts.fingerprints()}


def frozen_problem() -> str | None:
    """Why the sealed set may not run now, or None: it must be frozen as committed (the fingerprints file tracked, and
    no uncommitted change to it, to the generators, to the cases or to the scorer)."""
    frozen = json.loads(SEALED.read_text(encoding="utf-8")).get("cases") if SEALED.exists() else None
    if frozen != sealed_fingerprints():
        return "the sealed set does not match benchmark_v3_sealed.json (missing, or a case changed)"
    code, _ = _git("ls-files", "--error-unmatch", GUARDED[0])
    if code != 0:
        return "benchmark_v3_sealed.json is not committed"
    code, _ = _git("diff", "--quiet", "HEAD", "--", *GUARDED)
    if code != 0:
        return "the generators, the cases, the scorer or the fingerprints have uncommitted changes"
    return None


def main(argv: list[str] | None = None) -> int:
    import argparse
    import importlib
    import re

    suite = importlib.import_module("dclab_rnd.agent_eval.run")
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.benchmark_v3", description="Run benchmark v3 (15.9) with scripted policies.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--policy", action="append", choices=sorted(suite.POLICIES))
    parser.add_argument("--output", type=Path, help="write the report to this new file (never overwritten); required for the sealed test set")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error(f"{args.output} exists; results are never overwritten")
    if args.split == "test" and not args.output:
        parser.error("a run on the sealed test set is always recorded: give --output evidence/campaigns/benchmark_v3/results/BEN3-<n>_test_….json")
    if args.split == "test":
        problem = frozen_problem()
        if problem:
            parser.error(f"refusing to run the sealed set: {problem}")
    if args.output:  # the look is on record before it happens: a run that stops halfway still leaves its file
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump({"status": "started", "split": args.split, "campaign_id": VERSION, "code": _provenance()}, handle, indent=2)
            handle.write("\n")
    report = run(args.split, tuple(args.policy or ("standard", "audit", "bad")))
    for policy, s in report["summary"].items():
        print(f"{policy:9s} traps caught {s['leaks_caught'][0]}/{s['leaks_caught'][1]} (95% {s['leaks_caught_ci95']})  "
              f"false alarms {s['false_alarms'][0]}/{s['false_alarms'][1]} (95% {s['false_alarm_ci95']})  errors {s['errors']}")
    if args.output:
        found = re.match(r"([A-Z0-9]+-\d+)_", args.output.name)
        report = {"experiment_id": found.group(1) if found else args.output.stem, "status": "completed", **report}
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")  # the file this run created
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
