"""
problem_to_plan_samples.py
==========================

SFT samples for the front of the DCLab workflow, connected end to end:

    problem text ──► problem_understanding ──► ProblemBrief        (the problem contract)
    contract + data ──► data_analysis ──► DataFindings              (data understanding, checked against the contract)
    contract + findings ──► solution_design ──► SolutionPlan + WF-01…WF-10 workflow
    and one connected, multi-turn sample per scenario doing all three in a row.

Where the right answers come from
---------------------------------
* problem_understanding: the description is ASSEMBLED from sentences whose meaning is known, so the gold contract
  is known by construction. It is validated with the repository's own rules: ``dclab_core.contracts.ProblemBrief``
  (a stated fact needs its quote, an inferred one its reason) and ``dclab_core.understand.quoted / supports`` (the
  quote is the person's own words and supports the value). Open questions come from ``understand._questions``,
  the same rule the product uses, so the model learns to ask exactly what the product asks.
* data_analysis and solution_design: the labels are the library's own deterministic output,
  ``dclab_core.analyse.analyse`` (on the training partition only, WF-04) and ``dclab_core.plan.plan``, run with the
  GOLD contract. Every plan passes ``dclab_core.plan.validate``. The workflow mapping uses
  ``dclab_rnd.studio.graph`` (WF-01…WF-10, its rules and gates).

Data: six seeded replicas from ``dclab_core.replicas`` (invented data with planted traps, never anyone's private
data) and the ten public datasets in ``data/public`` with written problem statements.

Benchmark safety: ``dclab_rnd/agent_eval/analyst`` holds the analyst's dev and SEALED briefs. Nothing here reads
their answers; every generated description is compared with every benchmark description and dropped if it is
similar (difflib ratio >= 0.5), so training on these samples cannot leak the benchmark.

Usage
-----
    python research/llm-fine-tuning/experiments/sft_factory/problem_to_plan_samples.py
    python research/llm-fine-tuning/experiments/sft_factory/problem_to_plan_samples.py --scenarios subscription_churn --variants 2
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from dclab_core import analyse as A  # noqa: E402
from dclab_core import plan as P  # noqa: E402
from dclab_core import understand as U  # noqa: E402
from dclab_core.contracts import (Constraint, CoverageFact, DataFindings, Finding, LabelSourceFact, NamesFact,  # noqa: E402
                                  ProblemBrief, SolutionPlan, TaskFact, TextFact)
from dclab_core.replicas import (city_prices, delivery_time, product_reviews, store_demand,  # noqa: E402
                                 subscription_churn, ticket_routing)
from dclab_rnd.studio import graph as G  # noqa: E402
from leakage_audit_samples import SYSTEM_PROMPT, TARGETS, load  # noqa: E402

FIELDS = ["goal", "action", "unit", "task", "target_column", "label_source", "label_coverage", "prediction_moment",
          "time_column", "horizon", "entities", "primary_metric"]
FACT_TYPES = {"task": TaskFact, "label_source": LabelSourceFact, "label_coverage": CoverageFact, "entities": NamesFact}

PU_CARD = (
    "Problem understanding: read the problem description and the file headers (names only, never values) and write "
    "the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, "
    "'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong "
    "value is worse than an unknown one. Output JSON."
)
DA_CARD = (
    "Data analysis: the measurements below were computed on the training partition only, checked against the problem "
    "contract (target, prediction moment, time column, entities). For each measurement, write the finding: its "
    "severity (info, warning or critical), what it shows, what it implies for the solution, and which plan decisions "
    "it affects. Output JSON. "
    f"Rules used by the measurements: a class is tiny under {A.TINY_SHARE:.0%} of rows or {A.TINY_ROWS} rows; "
    f"a positive class under {A.RARE_SHARE:.0%} of rows is rare."
)
SD_CARD = (
    "Solution design: from the problem contract and the data findings, decide the solution plan (one decision per key: "
    "task framing, labels, split, metrics, baseline, forbidden features, identifiers, feature direction, cleaning, "
    "model families, threshold), each with its reason, the findings it cites and the alternatives set aside. Then lay "
    "the decisions on the DCLab workflow WF-01…WF-10 with its gates. Output JSON."
)

# Which workflow step carries each plan decision (dclab_rnd/studio/graph.py NODES).
DECISION_STEP = {
    "task.framing": "WF-01", "labels.coverage": "WF-01", "labels.join": "WF-02", "metric.primary": "WF-01",
    "metric.secondary": "WF-01", "baseline": "WF-01", "threshold.policy": "WF-01", "split.strategy": "WF-03",
    "data.cleaning": "WF-04", "features.forbidden": "WF-05", "features.to_confirm": "WF-05",
    "features.identifiers": "WF-05", "features.direction": "WF-06", "model.families": "WF-07",
}
STEP_DOES = {
    "WF-02": "Record each file's SHA-256 and where it came from; nothing is analysed before lineage is written.",
    "WF-04": "Profile the training partition only; every statistic that steers a choice is fitted there.",
    "WF-08": "Tune conservatively inside training folds, against the paired baseline.",
    "WF-09": "Open the locked holdout once, after every choice is fixed, and challenge the result by slices.",
    "WF-10": "Capture the notebook, the report and the training examples.",
}
GATE_AT = {"WF-04": "solution", "WF-09": "holdout"}


# ----------------------------------------------------------------------------------------------- scenarios
# Each sentence option is (quote, value). The quote is what the person writes; the value is what the contract stores.

def _replica_table(mod, attr, **kw):
    def make(seed):
        return getattr(mod.generate(seed=seed, **kw), attr)
    return make


SCENARIOS = {
    "subscription_churn": dict(
        source="replica", make=_replica_table(subscription_churn, "subscribers"), file="subscribers.csv",
        target="churned", task="binary", time="signup_time", unit_noun="subscriber",
        task_reason="churned holds yes or no for each subscriber, so the answer is binary",
        context=["We run a video subscription with three plans.", "Our streaming product renews every month or year."],
        goal=[("We want to know which subscribers will cancel.", "find subscribers who will cancel"),
              ("Our aim is to keep subscribers who are about to cancel.", "keep subscribers about to cancel")],
        action=[("The retention team will offer them a discount before they decide.", "retention team offers a discount"),
                ("Customer care will call the riskiest ones with a personal offer.", "customer care calls with an offer")],
        unit=[("Each row is one subscriber.", "subscriber")],
        target_s=[("The churned column says yes when the subscriber cancelled.", "churned"),
                ("The label is churned: yes if they left, no if they stayed.", "churned")],
        moment=[("We predict at renewal time, before the subscriber decides.", "at renewal time, before the subscriber decides"),
                ("The score is needed at renewal time; refunds are paid only after a cancellation.", "at renewal time")],
        horizon=[], entities=[],
        metric=[("We will judge it by average precision.", "average precision"), ("We care most about AUC.", "AUC")],
        constraints=[("The team can call the top 200 each week.", "review_capacity", "top 200 per week"),
                     ("A lost subscriber costs about 120 euros, a wasted discount about 10.", "budget", "lost subscriber 120 euros, wasted discount 10 euros")],
    ),
    "ticket_routing": dict(
        source="replica", make=_replica_table(ticket_routing, "tickets", n=3000), file="tickets.csv",
        target="assigned_team", task="multiclass", time="created_at", unit_noun="ticket",
        task_reason="assigned_team names one of several teams, so the answer is one class among many",
        context=["Our support desk receives tickets from web, email and chat.", "We sell a business software suite with a support desk."],
        goal=[("We want to send each new ticket straight to the right team.", "send each ticket to the right team"),
              ("The goal is to stop tickets bouncing between teams.", "stop tickets bouncing between teams")],
        action=[("The router will place the ticket in that team's queue.", "place the ticket in the team's queue")],
        unit=[("Every row is one support ticket.", "support ticket")],
        target_s=[("The target is assigned_team, the team that finally handled it.", "assigned_team"),
                ("assigned_team is the label we want to predict.", "assigned_team")],
        moment=[("Routing happens when the ticket is created, before anyone reads it.", "when the ticket is created, before anyone reads it")],
        horizon=[],
        entities=[("Customers often open many tickets, so customer_id repeats.", ["customer_id"])],
        metric=[("We will judge it by macro F1, since small teams matter too.", "macro F1")],
        constraints=[("Routing must answer within 200 milliseconds.", "latency", "answer within 200 milliseconds")],
    ),
    "delivery_time": dict(
        source="replica", make=_replica_table(delivery_time, "orders", n=3000), file="orders.csv",
        target="actual_minutes", task="regression", time="order_time", unit_noun="order",
        task_reason="actual_minutes is a number of minutes, so the answer is a quantity",
        context=["We run a food delivery app in one city.", "Restaurants hand meals to our couriers."],
        goal=[("We want to show customers an honest delivery estimate.", "show an honest delivery estimate"),
              ("Our goal is to predict how long each delivery will take.", "predict how long each delivery takes")],
        action=[("The estimate is shown to the customer in the app.", "estimate shown to the customer")],
        unit=[("Each row is one order.", "order")],
        target_s=[("The target is actual_minutes, the minutes from order to delivery.", "actual_minutes")],
        moment=[("We estimate when the order is placed, before a courier is assigned.", "when the order is placed, before a courier is assigned")],
        horizon=[],
        entities=[("Many orders come from the same restaurant, identified by restaurant_id.", ["restaurant_id"])],
        metric=[("We will judge it by MAE in minutes.", "MAE"), ("We will judge it by RMSE.", "RMSE")],
        constraints=[("An estimate within five minutes counts as good.", "other", "within five minutes counts as good")],
    ),
    "city_prices": dict(
        source="replica", make=_replica_table(city_prices, "flats"), file="flats.csv",
        target="price", task="regression", time=None, unit_noun="flat",
        task_reason="price is an amount of money, so the answer is a quantity",
        context=["We are a property website in one city.", "Owners list flats for sale on our site."],
        goal=[("We want to suggest a fair asking price for each new listing.", "suggest a fair asking price"),
              ("Our aim is to value flats in districts we have little data on.", "value flats in thinly covered districts")],
        action=[("The suggested price appears while the owner fills in the listing.", "suggested price shown during listing")],
        unit=[("Each row is one flat.", "flat")],
        target_s=[("price is the target, in euros.", "price"), ("The column to predict is price.", "price")],
        moment=[("We price a flat when it is listed: its place, size, rooms and age are known.", "when a flat is listed")],
        horizon=[], entities=[],
        metric=[("We will judge it by MAE.", "MAE")],
        constraints=[],
    ),
    "store_demand": dict(
        source="replica", make=_replica_table(store_demand, "sales"), file="sales.csv",
        target="units", task="regression", time="date", unit_noun="store, product and day",
        task_reason="units is a count of items sold, so the answer is a quantity",
        context=["We run a chain of grocery stores.", "Our stores reorder stock every week."],
        goal=[("We want to forecast how much each store will sell of each product.", "forecast sales per store and product")],
        action=[("The forecast drives the stock order.", "forecast drives the stock order")],
        unit=[("Each row is one store, product and day.", "store, product and day")],
        target_s=[("The target is units, the units sold that day.", "units")],
        moment=[("We forecast each Sunday evening, for the next 7 days of each store and product.", "each Sunday evening"),
                ("Every Sunday evening we forecast; promotions are planned weeks ahead, but the day's weather is measured on the day.", "every Sunday evening")],
        horizon=[("The forecast covers the next 7 days.", "the next 7 days")],
        entities=[("Each store and product pair is its own series, named in the series column.", ["series"])],
        metric=[("We will judge it by MAE.", "MAE")],
        constraints=[],
    ),
    "product_reviews": dict(
        source="replica", make=_replica_table(product_reviews, "reviews"), file="reviews.csv",
        target="negative", task="binary", time=None, unit_noun="review",
        task_reason="negative is 1 or 0 for each review, so the answer is binary",
        context=["We run an online marketplace.", "Shoppers leave reviews on the products they bought."],
        goal=[("We want to spot negative reviews early.", "spot negative reviews early"),
              ("Our aim is to answer unhappy shoppers before they leave.", "answer unhappy shoppers quickly")],
        action=[("A care agent will reply to each flagged review the same day.", "care agent replies the same day")],
        unit=[("Each row is one review.", "review")],
        target_s=[("The label is negative: 1 when the review is negative.", "negative")],
        moment=[("We score a review when it is posted, before anyone answers it.", "when the review is posted, before anyone answers it")],
        horizon=[], entities=[],
        metric=[("We will judge it by average precision.", "average precision")],
        constraints=[("Agents can handle the top 50 reviews a day.", "review_capacity", "top 50 per day")],
    ),
}

PUBLIC = {
    # key: (target column name, task, unit noun, context, goal, action, target sentence, moment sentence, metric, constraint)
    "adult": ("high_income", "a person", "We study census records.", ("We want to estimate which people earn more than 50K a year.", "estimate who earns more than 50K"),
              ("The estimate guides a survey's follow-up sampling.", "estimate guides survey follow-up"),
              ("The high_income column is 1 when income is above 50K.", "high_income"),
              ("We score each person from their census record, once the record is filed.", "once the census record is filed"), None),
    "bank_marketing": ("subscribed", "a client", "We are a retail bank calling clients about a term deposit.",
              ("We want to call the clients most likely to subscribe.", "call clients most likely to subscribe"),
              ("Agents call the top clients on the list first.", "agents call the top clients first"),
              ("The subscribed column says 1 when the client took the deposit.", "subscribed"),
              ("We score each client immediately before the call; the call's duration is known only after it ends.", "immediately before the call"),
              ("The team can make 300 calls a day.", "review_capacity", "300 calls a day")),
    "breast_cancer": ("malignant", "a tumour", "We support a pathology lab's research.",
              ("We want to separate malignant from benign tumours in a research set.", "separate malignant from benign tumours"), None,
              ("malignant is the label: 1 for a malignant tumour.", "malignant"),
              ("We classify after all cell-nucleus measurements are taken.", "after all cell-nucleus measurements are taken"), None),
    "credit_default": ("defaulted_next_month", "a card holder", "We issue credit cards.",
              ("We want to know which card holders will default.", "find card holders who will default"),
              ("Collections will contact them before the due date.", "collections contacts them before the due date"),
              ("The target is defaulted_next_month, 1 when the payment was missed.", "defaulted_next_month"),
              ("We predict at the end of each billing period.", "at the end of each billing period"),
              ("A missed payment costs about 900, a needless reminder about 5.", "budget", "missed payment 900, needless reminder 5")),
    "german_credit": ("bad_risk", "a loan applicant", "We are a lender reviewing loan applications.",
              ("We want to flag applicants who will turn out to be bad risks.", "flag applicants who will be bad risks"),
              ("Flagged applications go to a senior underwriter.", "flagged applications go to a senior underwriter"),
              ("bad_risk is the label: 1 for a bad credit risk.", "bad_risk"),
              ("We score each application at underwriting time.", "at underwriting time"), None),
    "heart_disease": ("disease", "a patient", "We assist a cardiology research group.",
              ("We want to classify whether heart disease is present.", "classify whether heart disease is present"), None,
              ("The disease column is 1 when heart disease is present.", "disease"),
              ("We classify after all listed tests, including ca and thal, are done.", "after all listed tests are done"), None),
    "mushroom": ("poisonous", "a mushroom", "We catalogue mushrooms for a field guide.",
              ("We want to tell poisonous mushrooms from edible ones.", "tell poisonous from edible mushrooms"), None,
              ("The poisonous column is 1 for a poisonous mushroom.", "poisonous"),
              ("We classify once the mushroom's physical traits are recorded.", "once physical traits are recorded"), None),
    "online_shoppers": ("purchased", "a website session", "We run an online shop.",
              ("We want to know which visitors will buy in this session.", "know which visitors will buy"),
              ("Likely buyers get a chat invitation.", "likely buyers get a chat invitation"),
              ("purchased is the label: 1 when the session ended in a purchase.", "purchased"),
              ("We predict at the start of each session; page counts, durations, bounce and exit rates and page values describe the finished session.", "at the start of each session"),
              ("The chat team can talk to 100 visitors an hour.", "review_capacity", "100 visitors an hour")),
    "spambase": ("spam", "an email", "We run a mail service.",
              ("We want to keep spam out of the inbox.", "keep spam out of the inbox"),
              ("Emails flagged as spam go to the junk folder.", "flagged emails go to the junk folder"),
              ("The spam column is 1 for a spam email.", "spam"),
              ("We classify each email when it is received.", "when the email is received"), None),
    "wine_quality": ("good_quality", "a wine", "We advise a wine producer.",
              ("We want to predict which wines the panel will rate 6 or higher.", "predict wines rated 6 or higher"), None,
              ("The good_quality column is 1 when the rating is 6 or higher.", "good_quality"),
              ("We predict after the physicochemical tests, before the tasting panel.", "after the tests, before the tasting panel"), None),
}

PUBLIC_HORIZON = {"credit_default": [("The prediction looks one month ahead.", "one month ahead")]}

for _k, (_tgt, _unit, _ctx, _goal, _act, _ts, _mom, _con) in PUBLIC.items():
    SCENARIOS[_k] = dict(
        source="public", file=f"{_k}.csv", target=_tgt, task="binary", time=None, unit_noun=_unit,
        task_reason=f"{_tgt} is 1 or 0, so the answer is binary",
        context=[_ctx, f"This project uses the {_k.replace('_', ' ')} data."],
        goal=[_goal, (f"We need to predict {TARGETS[_k]}.", TARGETS[_k].replace("whether ", ""))],
        action=[_act] if _act else [],
        unit=[(f"Each row is {_unit}.", _unit.split(" ", 1)[1]), (f"Every row describes {_unit}.", _unit.split(" ", 1)[1])],
        target_s=[_ts, (f"{_tgt} is the label.", _tgt), (f"The target column is {_tgt}.", _tgt)], moment=[_mom], horizon=PUBLIC_HORIZON.get(_k, []), entities=[],
        metric=[("We will judge it by average precision.", "average precision"), ("We care most about AUC.", "AUC")],
        constraints=[_con] if _con else [],
    )

from splits import SPLIT_OF  # noqa: E402  (one split map for every generator)


# ----------------------------------------------------------------------------------------------- the contract

def _stated(cls, value, quote):
    return cls(value=value, status="stated", quote=quote, source="text")


def _inferred(cls, value, reason):
    return cls(value=value, status="inferred", reason=reason, source="rule")


def build_case(key, spec, rng, columns, full: bool):
    """Assemble one description and its gold contract. ``full`` keeps target and moment (needed downstream)."""
    take = lambda p: rng.random() < p  # noqa: E731
    pick = lambda opts: opts[rng.integers(len(opts))]  # noqa: E731
    parts, fields, constraints = [pick(spec["context"])], {}, []

    def use(name, p, field=None, cls=TextFact):
        opts = spec[name]
        if opts and take(p):
            q, v = pick(opts)
            parts.append(q)
            fields[field or name] = _stated(cls, v, q)
            return True
        return False

    has_goal = use("goal", 0.9)
    use("action", 0.6)
    stated_unit = use("unit", 0.6)
    has_target = use("target_s", 1.0 if full else 0.6, "target_column")
    has_moment = use("moment", 1.0 if full else 0.6, "prediction_moment")
    use("horizon", 0.8)
    use("entities", 0.7, cls=NamesFact)
    use("metric", 0.5, "primary_metric")
    if spec["constraints"] and take(0.6):
        q, kind, text = pick(spec["constraints"])
        parts.append(q)
        constraints.append(Constraint(kind=kind, text=text, quote=q))

    file = spec["file"]
    if not stated_unit:
        fields["unit"] = _inferred(TextFact, spec["unit_noun"], f"each row of {file} describes one {spec['unit_noun']}")
    if has_target:
        fields["task"] = _inferred(TaskFact, spec["task"], spec["task_reason"])
        fields["label_source"] = _inferred(LabelSourceFact, "column", f"the label is a column of {file}")
        fields["label_coverage"] = _inferred(CoverageFact, "complete", "the label is a column of the main table, so every row is labelled")
    elif has_goal:
        kind = {"binary": "a yes or no for each case", "multiclass": "one of several classes",
                "regression": "a quantity"}[spec["task"]]
        fields["task"] = _inferred(TaskFact, spec["task"], f'the goal "{fields["goal"].quote}" asks for {kind}')
    # invariants: labels are known exactly when the target is stated
    assert ("label_source" in fields) == has_target and ("label_coverage" in fields) == has_target
    if spec["time"]:
        fields["time_column"] = _inferred(TextFact, spec["time"], f"{spec['time']} is the only date or time column in {file}")

    description = " ".join(parts)
    all_fields = {f: fields.get(f, FACT_TYPES.get(f, TextFact)()) for f in FIELDS}
    all_fields["target_definition"] = TextFact()
    questions = U._questions(all_fields, constraints)
    brief = ProblemBrief(constraints=constraints, open_questions=questions,
                         **{n: f.model_dump() for n, f in all_fields.items()})
    return description, brief, has_target and has_moment


def compact(brief: ProblemBrief) -> dict:
    out = {}
    for f in FIELDS:
        fact = getattr(brief, f)
        if fact.known:
            out[f] = {"value": fact.value, "status": fact.status,
                      **({"quote": fact.quote} if fact.status == "stated" else {"reason": fact.reason})}
    out["unknown"] = [f for f in FIELDS if not getattr(brief, f).known]
    out["constraints"] = [{"kind": c.kind, "text": c.text, "quote": c.quote} for c in brief.constraints]
    out["open_questions"] = [{"field": q.field, "text": q.text, "impact": q.impact} for q in brief.open_questions]
    return out


def from_compact(d: dict) -> ProblemBrief:
    kw = {}
    for f in FIELDS:
        if f in d:
            x = d[f]
            kw[f] = {"value": x["value"], "status": x["status"],
                     **({"quote": x["quote"], "source": "text"} if x["status"] == "stated" else {"reason": x["reason"], "source": "rule"})}
    return ProblemBrief(constraints=[Constraint(**c) for c in d.get("constraints", [])],
                        open_questions=d.get("open_questions", []), **kw)


def check_contract(description: str, brief: ProblemBrief) -> list[str]:
    problems = []
    for f in FIELDS:
        fact = getattr(brief, f)
        if fact.status == "stated":
            if not U.quoted(fact.quote, description):
                problems.append(f"{f}: quote not in description")
            if not U.supports(fact.quote, " ".join(fact.value) if isinstance(fact.value, list) else fact.value):
                problems.append(f"{f}: quote does not support value")
    for c in brief.constraints:
        if not U.quoted(c.quote, description):
            problems.append("constraint quote not in description")
    return problems


# ----------------------------------------------------------------------------------------------- data and plan

def table_for(key, spec, seed):
    if spec["source"] == "replica":
        df = spec["make"](seed).reset_index(drop=True)
    else:
        X, y, _ = load(key, 3000, seed)
        df = X.copy()
        df[spec["target"]] = y.to_numpy()
    return df, Path(spec["file"]).stem


def training_index(df, spec, rng):
    n = int(len(df) * 0.75)
    if spec["time"] and spec["time"] in df.columns:
        order = pd.to_datetime(df[spec["time"]], errors="coerce").sort_values(kind="stable").index
        return list(order[:n]), f"the earliest 75% of rows by {spec['time']} ({n:,} rows); the rest is the locked holdout"
    idx = rng.permutation(len(df))[:n]
    return sorted(int(i) for i in idx), f"a random 75% of rows ({n:,} rows); the rest is the locked holdout"


def workflow(plan: SolutionPlan) -> list[dict]:
    """Each plan decision placed on the WF-01…WF-10 step that carries it, with the two gates. Compact on purpose:
    the steps' fixed rules are the workflow's, not the plan's, so the plan does not repeat them."""
    by_step = defaultdict(list)
    for d in plan.decisions:
        by_step[DECISION_STEP.get(d.key, "WF-01")].append(d.key)
    steps = []
    for node in G.NODES:
        item = {"step": f"{node['id']} {node['name']}"}
        if by_step.get(node["id"]):
            item["decisions"] = by_step[node["id"]]
        if node["id"] in GATE_AT:
            item["gate"] = GATE_AT[node["id"]]
        steps.append(item)
    return steps


def plan_json(plan: SolutionPlan) -> dict:
    return {
        "task": plan.task,
        "decisions": [{"key": d.key, "choice": d.choice, "cites": d.cites, "reason": d.reason,
                       "alternatives": [{"choice": a.choice, "why_not": a.why_not} for a in d.alternatives]}
                      for d in plan.decisions],
        "risks": [{"statement": r.statement, "check": r.check, "cites": r.cites} for r in plan.risks],
        "would_change": plan.would_change,
        "workflow": workflow(plan),
    }


def evidence_view(ev: dict) -> str:
    """The evidence as measured, plus the length of every list, so no answer has to count."""
    view = dict(ev)
    for k, v in ev.items():
        if isinstance(v, (list, dict)) and v:
            view[f"{k}_count"] = len(v)
            if isinstance(v, list) and ev.get("column") in v:  # "and so do N other columns"
                view[f"{k}_other_than_{ev['column']}_count"] = len(v) - 1
    return json.dumps(view, ensure_ascii=False, default=str)


def findings_in(f: DataFindings) -> str:
    return "\n".join(f"- {x.id} [{x.topic}] evidence: {evidence_view(x.evidence)}" for x in f.findings)


def findings_out(f: DataFindings) -> list[dict]:
    return [{"id": x.id, "severity": x.severity, "statement": x.statement, "implication": x.implication,
             "affects": x.affects} for x in f.findings]


def findings_brief(f: DataFindings) -> str:
    return "\n".join(f"- {x.id} ({x.severity}, {x.topic}): {x.statement} Implication: {x.implication} "
                     f"Evidence: {evidence_view(x.evidence)}" for x in f.findings)


J = lambda o: json.dumps(o, ensure_ascii=False, default=str)  # noqa: E731


# ----------------------------------------------------------------------------------------------- number check

NUM = re.compile(r"(?<![\w.\-])[+-]?\d[\d,]*(?:\.\d+)?%?(?![\w-])")  # skips ids such as WF-02, DCLAB-R04, SHA-256
POLICY_CONSTANTS = {"0.5"}  # the plan's standard "probability threshold" alternative: a convention, not data


SCI = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?e[+-]\d+")


def _shown(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def numbers_ok(output: str, inputs: str) -> list[str]:
    """Every number in the output appears in the input: directly, rounded as displayed, as a percentage of an input
    decimal, or in scientific notation to its shown precision."""
    inputs_values = [float(x) for x in SCI.findall(inputs)]
    sci_bad = []
    for tok in SCI.findall(output):
        v, mant = float(tok), tok.lower().split("e")[0]
        rel = 0.5 * 10 ** (-_shown(mant)) / max(abs(float(mant)), 1e-12)
        pool = inputs_values + [float(x.replace(",", "")) for x in NUM.findall(SCI.sub(" ", inputs)) if not x.endswith("%")]
        if not any(abs(d - v) <= rel * abs(v) + 1e-12 for d in pool):
            sci_bad.append(tok)
    output, inputs = SCI.sub(" ", output), SCI.sub(" ", inputs)
    have = set(NUM.findall(inputs))
    decimals = []
    for n in have:
        try:
            decimals.append(float(n.rstrip("%").replace(",", "")))
        except ValueError:
            pass
    bad = []
    for n in NUM.findall(output):
        if n in have or n.lstrip("+") in have or len(n.strip("+-")) <= 1 or n in POLICY_CONSTANTS:
            continue
        if n.endswith("%"):
            v = float(n.rstrip("%").replace(",", ""))
            if any(round(d * 100, _shown(n.rstrip("%"))) == v for d in decimals):  # the same value, rounded as displayed
                continue
        try:
            v = float(n.replace(",", "").rstrip("%"))
            if any(round(d, _shown(n)) == v for d in decimals):  # the same value, rounded as displayed
                continue
        except ValueError:
            pass
        bad.append(n)
    return sci_bad + bad


# ----------------------------------------------------------------------------------------------- benchmark guard

def benchmark_texts():
    out = []
    for name in ("briefs_dev.json", "briefs_sealed.json"):
        p = ROOT / "dclab_rnd" / "agent_eval" / "analyst" / name
        if p.exists():
            out += [b["description"] for b in json.loads(p.read_text())]
    return out


def too_close(text, bench):
    t = text.lower()
    return max((difflib.SequenceMatcher(None, t, b.lower()).ratio() for b in bench), default=0.0)


# ----------------------------------------------------------------------------------------------- build

def files_line(spec, df):
    return f"{spec['file']}: {', '.join(map(str, df.columns))}"


def build(keys, variants, seeds, partial, seed0):
    bench = benchmark_texts()
    samples, report = [], {"dropped": Counter(), "max_benchmark_similarity": 0.0, "library_vs_gold": Counter(),
                           "plan_validation_failures": 0}
    for key in keys:
        spec = SCENARIOS[key]
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) + seed0)  # stable across runs
        t0, n0 = time.time(), len(samples)
        seed_list = seeds if spec["source"] == "replica" else [1]
        per_seed = variants if spec["source"] == "replica" else variants * 2
        for seed in seed_list:
            df, table = table_for(key, spec, seed)
            files = {spec["file"]: list(map(str, df.columns))}
            idx, partition = training_index(df, spec, rng)
            cache = {}
            for v in range(per_seed + partial):
                full = v < per_seed
                description, brief, connected = build_case(key, spec, rng, list(df.columns), full)
                sim = too_close(description, bench)
                report["max_benchmark_similarity"] = max(report["max_benchmark_similarity"], round(sim, 3))
                if sim >= 0.5:
                    report["dropped"]["too_close_to_benchmark"] += 1
                    continue
                problems = check_contract(description, brief)
                if problems:
                    report["dropped"]["contract_check"] += 1
                    continue
                # library comparison (diagnostic only, never a label)
                lib = U.understand(description, files)
                for f in FIELDS:
                    g, l = getattr(brief, f), getattr(lib, f)
                    report["library_vs_gold"][
                        "agree" if (g.known == l.known and (not g.known or g.value == l.value)) else
                        ("library_unknown" if g.known and not l.known else
                         "library_extra" if l.known and not g.known else "library_differs")] += 1
                meta = {"scenario": key, "dataset": key, "seed": seed, "source": spec["source"]}
                pu_user = (f"STAGE: problem_understanding\n{PU_CARD}\n\nDESCRIPTION:\n{description}\n\n"
                           f"FILES (headers only):\n{files_line(spec, df)}")
                pu_out = J(compact(brief))
                assert from_compact(json.loads(pu_out)).model_dump() == brief.model_dump()
                samples.append(_sample([("user", pu_user), ("assistant", pu_out)],
                                       {**meta, "stage": "problem_understanding", "complete": connected}))
                if not connected:
                    continue
                ck = J(compact(brief))
                if ck not in cache:
                    findings = A.analyse({table: df}, target=spec["target"], brief=brief, train_index=idx, main=table)
                    plan = P.plan(brief, findings)
                    issues = P.validate(plan, findings)
                    cache[ck] = (findings, plan, issues)
                findings, plan, issues = cache[ck]
                if issues:
                    report["plan_validation_failures"] += 1
                    report["dropped"]["plan_validation"] += 1
                    continue
                contract = J(compact(brief))
                table_line = f"TABLE: {table} — {len(df):,} rows, {df.shape[1]} columns ({', '.join(map(str, df.columns))})"
                da_body = (f"PROBLEM CONTRACT:\n{contract}\n\n{table_line}\nTRAINING PARTITION: {partition}\n\n"
                           f"MEASUREMENTS:\n{findings_in(findings)}")
                da_user = f"STAGE: data_analysis\n{DA_CARD}\n\n{da_body}"
                da_out = J({"task": findings.task, "target": findings.target, "training_rows": findings.training_rows,
                            "findings": findings_out(findings)})
                sd_user = (f"STAGE: solution_design\n{SD_CARD}\n\nPROBLEM CONTRACT:\n{contract}\n\n"
                           f"DATA FINDINGS:\n{findings_brief(findings)}")
                sd_out = J(plan_json(plan))
                bad = numbers_ok(da_out, da_user) + numbers_ok(sd_out, sd_user)
                if bad:
                    report["dropped"]["number_not_in_input"] += 1
                    report.setdefault("number_examples", []).append({"scenario": key, "numbers": bad[:3]})
                    continue
                samples.append(_sample([("user", da_user), ("assistant", da_out)], {**meta, "stage": "data_analysis"}))
                samples.append(_sample([("user", sd_user), ("assistant", sd_out)], {**meta, "stage": "solution_design"}))
                chain = [
                    ("user", pu_user), ("assistant", pu_out),
                    ("user", f"STAGE: data_analysis\n{DA_CARD}\n\n{table_line}\nTRAINING PARTITION: {partition}\n\n"
                             f"MEASUREMENTS:\n{findings_in(findings)}"),
                    ("assistant", da_out),
                    ("user", f"STAGE: solution_design\n{SD_CARD}\n\nUse the contract and the findings above."),
                    ("assistant", sd_out),
                ]
                samples.append(_sample(chain, {**meta, "stage": "connected_problem_to_plan"}))
        print(f"{key:20s} {len(samples) - n0:4d} samples  {time.time() - t0:6.1f}s", flush=True)
    return samples, report


def _sample(turns, meta):
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + [{"role": r, "content": c} for r, c in turns],
            "metadata": {**meta, "action": meta["stage"]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--variants", type=int, default=5, help="complete descriptions per replica seed (public: twice this)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--partial", type=int, default=3, help="extra incomplete descriptions per seed (problem understanding only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()

    samples, report = build(args.scenarios, args.variants, args.seeds, args.partial, args.seed)
    seen, stages = set(), defaultdict(lambda: {"train": [], "val": [], "test": []})
    for s in samples:
        h = hashlib.sha256(J(s["messages"][1:]).encode()).hexdigest()
        if h in seen:
            report["dropped"]["duplicate"] += 1
            continue
        seen.add(h)
        stages[s["metadata"]["stage"]][SPLIT_OF.get(s["metadata"]["scenario"], "train")].append(s)
    summary = {}
    for stage, splits in stages.items():
        random.Random(args.seed).shuffle(splits["train"])
        out = args.out / stage
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in splits.items():
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(J(r) + "\n" for r in rows)
        summary[stage] = {k: len(v) for k, v in splits.items()}
        (out / "manifest.json").write_text(json.dumps({"stage": stage, "splits": summary[stage],
                                                       "by_scenario": dict(Counter(r["metadata"]["scenario"] for v in splits.values() for r in v)),
                                                       "split_by_scenario": SPLIT_OF}, indent=2))
    report["dropped"] = dict(report["dropped"])
    report["library_vs_gold"] = dict(report["library_vs_gold"])
    report["stages"] = summary
    (args.out / "problem_to_plan_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
