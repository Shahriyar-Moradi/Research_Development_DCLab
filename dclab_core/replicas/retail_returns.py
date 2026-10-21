"""Retail returns fraud, invented: point-of-sale bills and the returns filed against them, with staff return fraud planted (a rare event replica).

A retail group wants to find the fake returns its staff file: a cashier copies a real bill and, late in the evening, files a return against the copy to
take the refund. It has the sales table (one row per bill) and the returns table (one row per return). Its labels grew over time: at first only its
business rules said which returns look fake; later the loss-prevention team began to review a fixed number of returns a day, the rule's hits first and
a random sample of the rest, and clustering spread their confirmed cases to similar ones. A common loss-prevention set-up, invented here with
the traps below.

The traps a careful data scientist must see:

* **Rare positives.** About 3% of returns are fake: accuracy flatters any model that flags nothing; precision at the review capacity and the area
  under the precision-recall curve say how well it ranks.
* **Labels from rules.** Before the team reviewed anything, the late-evening rule labelled every return filed after 22:30 as fraud
  (`label_source` = rule): most of them are honest. A model trained or judged on those labels learns the rule, not the fraud; only the reviewed labels
  (`expert`, and the random part of each day's batch, `expert_audit`) are evidence, and the clustered ones (`cluster`) spread mistakes.
* **A rule computed with future data.** `cashier_late_share` is each cashier's share of late returns over the whole log, the months after each return
  included: at the moment a return is filed nobody knows it. It must be kept out or recomputed from earlier returns only.
* **Written after the review.** `review_status` is the loss-prevention team's verdict: it is the label itself, written after the moment.
* **Copied bills.** A fake return points at a copy: a new bill a dishonest cashier issued from a real one (the same items, the amount edited a
  little, minutes after it, sometimes from another store), or at a real bill nobody returned. The copy and its original are near-duplicates that may
  fall on both sides of a time split; honest customers also buy the same things twice, so a match alone is not fraud. Only matching across the sales
  table finds the copies, and a copy must stay on the side of its original.
* **Staff, not customers.** A few cashiers file all the fake returns, more of them in the last months: a model that learns who they are has learnt
  nothing about the next one.
* **Drift.** In the last three months the dishonest cashiers learn the rule and file earlier in the evening, and new ones start: the latest period is
  the honest holdout, and the rule's recall falls there.
* **Review capacity.** The team reviews five returns a day, fewer than are filed on almost every day: the threshold is the fifth score of a day,
  and the random part of each batch is how what the queue misses is estimated.

``truth`` says which returns were planted, how, and which bill a copy copies, and is for tests only. Invented data: a score on it says whether the
decisions are sound, never how a model would do on real returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dclab_core.replicas import LABELS

SALES_FILE, RETURNS_FILE, INFO_FILE = "sales.csv", "returns.csv", "replica.json"
LATE = 22.5            # the business rule: a return filed after 22:30 is suspicious
CAPACITY = 5           # returns the loss-prevention team reviews a day, the rule's hits first, then a random sample
REVIEW_FROM = 120      # the day the team began to review: before it, only the rule labelled
DRIFT_DAYS = 90        # the last three months: the dishonest cashiers adapt; the holdout
OPEN, CLOSE = 10.0, 23.4
COPIED, OTHER_STORE, REUSED = "copied_bill", "copied_from_another_store", "reused_bill"


@dataclass(frozen=True)
class Replica:
    """What a person would upload, plus what only the tests may see."""

    sales: pd.DataFrame     # one row per bill (the copies among them)
    returns: pd.DataFrame   # one row per return, with the labels as the business has them
    truth: pd.DataFrame     # every return: planted fraud or not, how, and the bill a copy copies (tests only)
    facts: dict[str, Any] = field(default_factory=dict)


def _ids(rng: np.random.Generator, n: int, prefix: str) -> np.ndarray:
    digits = np.array(list("0123456789abcdef"))[rng.integers(0, 16, size=(n, 10))]
    return np.array([prefix + "".join(row) for row in digits])


def _in_hours(ts: pd.Series, rng: np.random.Generator) -> pd.Series:
    """A time outside the store's hours moves to the next opening, a little after it."""
    ts = pd.Series(ts).reset_index(drop=True)
    hour = ts.dt.hour + ts.dt.minute / 60
    late, early = hour >= CLOSE, hour < OPEN
    opening = ts.dt.normalize() + pd.to_timedelta(late.astype(int), "D") + pd.to_timedelta(OPEN + rng.uniform(0, 2, len(ts)), "h")
    return ts.where(~(late | early), opening)


def generate(seed: int = 0, n_stores: int = 40, cashiers_per_store: int = 6, days: int = 270, n_sales: int = 60_000,
             return_rate: float = 0.08, n_dishonest: int = 12, fraud_per_day: tuple[float, float] = (0.45, 1.1),
             capacity: int = CAPACITY, repeat_share: float = 0.03) -> Replica:
    """Draw a replica. Deterministic per ``seed``. ``fraud_per_day``: the planted fake returns a day before and during the last
    ``DRIFT_DAYS``; ``capacity``: the returns the team reviews a day from ``REVIEW_FROM``; ``repeat_share``: honest customers who buy
    the same thing again (honest near-duplicates)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-04")
    end = start + pd.Timedelta(days=days)
    stores = np.array([f"ST{100 + i}" for i in range(n_stores)])
    staff = {s: [f"{s}-C{j}" for j in range(cashiers_per_store)] for s in stores}
    cashiers = np.array([c for s in stores for c in staff[s]])
    store_of = {c: c.split("-")[0] for c in cashiers}

    # ---- sales: one row per bill, and honest repeat purchases (the same items and price again, days later)
    day = rng.integers(0, days, n_sales)
    hour = np.clip(rng.normal(16.5, 3.0, n_sales), OPEN, 22.9)
    sales = pd.DataFrame({"sold_at": start + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h"), "cashier_id": rng.choice(cashiers, n_sales),
                          "amount": np.round(np.exp(rng.normal(4.6, 0.7, n_sales)), 2), "items": rng.integers(1, 7, n_sales),
                          "payment": rng.choice(["card", "cash"], n_sales, p=[0.7, 0.3])})
    again = sales.sample(frac=repeat_share, random_state=seed).reset_index(drop=True)
    again["sold_at"] = _in_hours(again["sold_at"] + pd.to_timedelta(rng.uniform(1, 20, len(again)), "D"), rng)
    again["cashier_id"] = [staff[store_of[c]][int(rng.integers(0, cashiers_per_store))] for c in again["cashier_id"]]
    sales = pd.concat([sales, again[again["sold_at"] < end]], ignore_index=True)
    sales["store_id"] = sales["cashier_id"].map(store_of)
    sales["bill_id"] = _ids(rng, len(sales), "B")

    # ---- honest returns: a customer brings back all or part of what they bought, after buying it, in the store's hours
    picked = sales.sample(frac=return_rate, random_state=seed + 1).reset_index(drop=True)
    picked = pd.concat([picked, picked.sample(frac=0.05, random_state=seed + 2)], ignore_index=True)  # some bills are returned twice, in parts
    n = len(picked)
    returned_at = _in_hours(picked["sold_at"] + pd.to_timedelta(rng.uniform(1, 30 * 24, n), "h"), rng)
    r_store = np.where(rng.random(n) < 0.93, picked["store_id"], rng.choice(stores, n))
    share = np.where(rng.random(n) < 0.7, 1.0, rng.uniform(0.2, 0.9, n))
    method = np.where(rng.random(n) < 0.85, picked["payment"], rng.choice(["store_credit", "cash", "card"], n, p=[0.6, 0.2, 0.2]))
    honest = pd.DataFrame({"bill_id": picked["bill_id"], "store_id": r_store,
                           "cashier_id": [staff[s][int(rng.integers(0, cashiers_per_store))] for s in r_store], "returned_at": returned_at,
                           "amount": np.round(picked["amount"] * share, 2), "items": np.maximum(1, np.round(picked["items"] * share)).astype(int),
                           "refund_method": method, "reason_code": rng.choice(["size", "defect", "changed_mind", "gift"], n, p=[0.4, 0.2, 0.3, 0.1]),
                           "_fraud": 0, "_how": "", "_copy_of": ""})

    # ---- fake returns: a few dishonest cashiers copy a recent real bill (a new bill, the amount edited a little) or reuse one, and return it
    dishonest = rng.choice(cashiers, n_dishonest, replace=False)
    early_crew = dishonest[: max(1, n_dishonest * 2 // 3)]  # more of them in the last months
    sold = sales.sort_values("sold_at").reset_index(drop=True)
    times = sold["sold_at"].to_numpy()
    copies, fakes = [], []
    for d in range(days):
        drifted = d >= days - DRIFT_DAYS
        for _ in range(rng.poisson(fraud_per_day[1] if drifted else fraud_per_day[0])):
            c = str(rng.choice(dishonest if drifted else early_crew))
            s = store_of[c]
            how = str(rng.choice([COPIED, OTHER_STORE, REUSED], p=[0.6, 0.2, 0.2]))
            lo = np.searchsorted(times, np.datetime64(start + pd.Timedelta(days=d - 20)))
            hi = np.searchsorted(times, np.datetime64(start + pd.Timedelta(days=d)))
            recent = sold.iloc[lo:hi]
            pool = recent[recent["store_id"] != s] if how == OTHER_STORE else recent[recent["store_id"] == s]
            if pool.empty:
                continue
            bill = pool.iloc[int(rng.integers(0, len(pool)))]
            target, copy_of, amount = bill["bill_id"], "", float(bill["amount"])
            if how != REUSED:  # the copy: a new bill in the dishonest cashier's store, minutes after the original, the amount edited a little
                target, copy_of = str(_ids(rng, 1, "B")[0]), bill["bill_id"]
                amount = round(amount * float(rng.uniform(0.95, 1.05)), 2)
                copies.append({"sold_at": bill["sold_at"] + pd.Timedelta(minutes=float(rng.uniform(2, 90))), "cashier_id": c, "store_id": s,
                               "amount": amount, "items": int(bill["items"]), "payment": "cash" if rng.random() < 0.8 else "card",
                               "bill_id": target, "_copy_of": copy_of})
            h = float(np.clip(rng.normal(19.8, 1.2) if drifted else rng.normal(22.9, 0.45), 17.0, CLOSE - 0.05))
            fakes.append({"bill_id": target, "store_id": s, "cashier_id": c, "returned_at": start + pd.Timedelta(days=d) + pd.to_timedelta(h, "h"),
                          "amount": amount, "items": int(bill["items"]),
                          "refund_method": str(rng.choice(["cash", "store_credit", "card"], p=[0.85, 0.1, 0.05])),
                          "reason_code": str(rng.choice(["changed_mind", "size", "defect"], p=[0.5, 0.35, 0.15])),
                          "_fraud": 1, "_how": how, "_copy_of": copy_of})
    sales = pd.concat([sales, pd.DataFrame(copies).drop(columns=["_copy_of"], errors="ignore")], ignore_index=True)
    sales = sales.sort_values("sold_at").reset_index(drop=True)
    returns = pd.concat([honest, pd.DataFrame(fakes)], ignore_index=True)
    sold_at = returns["bill_id"].map(sales.set_index("bill_id")["sold_at"])
    returns = returns[(returns["returned_at"] < end) & (returns["returned_at"] > sold_at)]  # the log ends with its last day; a return follows its sale
    returns = returns.sort_values("returned_at", kind="stable").reset_index(drop=True)
    returns.insert(0, "return_id", _ids(rng, len(returns), "R"))

    # ---- what the business adds: the rule, a cashier statistic over the whole log, labels as they grew, the review verdict
    hour_of = returns["returned_at"].dt.hour + returns["returned_at"].dt.minute / 60
    returns["rule_flag"] = (hour_of >= LATE).astype(int)
    returns["cashier_late_share"] = returns.groupby("cashier_id")["rule_flag"].transform("mean").round(4)  # the whole log: the future included
    label = pd.Series(np.nan, index=returns.index)
    source = pd.Series("", index=returns.index, dtype=object)
    day_of = (returns["returned_at"] - start).dt.days.to_numpy()
    flagged_all = returns["rule_flag"].to_numpy() == 1
    first = returns.index[(day_of < REVIEW_FROM) & flagged_all]
    label[first], source[first] = 1.0, "rule"  # before the team: every late return is fraud, by the rule
    for d in range(REVIEW_FROM, days):  # the team's day: the rule's hits first, then a random sample of the rest, `capacity` in all
        today = returns.index[day_of == d].to_numpy()
        if not len(today):
            continue
        flagged = today[flagged_all[today]]
        hits = rng.permutation(flagged)[:capacity]
        rest = rng.permutation(np.setdiff1d(today, hits))[: max(0, capacity - len(hits))]
        label[hits], source[hits] = returns.loc[hits, "_fraud"].astype(float), "expert"
        label[rest], source[rest] = returns.loc[rest, "_fraud"].astype(float), "expert_audit"
        missed = np.setdiff1d(flagged, hits)
        label[missed], source[missed] = 1.0, "rule"  # the rule's hits the team had no time for
    confirmed = returns.loc[source.isin(["expert", "expert_audit"]) & (label == 1)]
    near = returns.loc[(source == "") & returns["cashier_id"].isin(confirmed["cashier_id"]) & (returns["refund_method"] == "cash")]
    near = near.sample(min(len(near), max(1, len(confirmed) // 2)), random_state=seed + 3) if len(near) else near
    label[near.index], source[near.index] = 1.0, "cluster"  # "similar to a confirmed case": the same cashier, a cash refund
    returns["label"] = label
    returns["label_source"] = source
    returns["review_status"] = np.where(source.isin(["expert", "expert_audit"]), np.where(returns["_fraud"] == 1, "confirmed", "cleared"), "")

    truth = returns[["return_id", "_fraud", "_how", "_copy_of", "cashier_id", "bill_id"]].rename(
        columns={"_fraud": "fraud", "_how": "how", "_copy_of": "copy_of"})
    returns = returns.drop(columns=["_fraud", "_how", "_copy_of"])
    sales = sales[["bill_id", "store_id", "cashier_id", "sold_at", "amount", "items", "payment"]]
    sales["sold_at"] = sales["sold_at"].dt.strftime("%Y-%m-%d %H:%M")
    returns["returned_at"] = returns["returned_at"].dt.strftime("%Y-%m-%d %H:%M")
    facts = {"target": "label", "positive": 1, "rows": len(returns), "fraud_rate": round(float(truth["fraud"].mean()), 4),
             "time_column": "returned_at", "holdout_days": DRIFT_DAYS, "review_from_day": REVIEW_FROM,
             "split": "time and group: the last 90 days are the holdout; a copy and the bill it copies on one side",
             "group_column": "bill_id", "near_duplicates": "a copy and its original (the same items, the amount within 5%, minutes apart) are one group",
             "entity": "cashier_id", "capacity_per_day": int(capacity), "rule": f"returned at or after {LATE:g}h",
             "join": {"returns.bill_id": "sales.bill_id"}, "dishonest_cashiers": int(n_dishonest),
             "keep_out": {"review_status": "written after the review: it is the label", "cashier_late_share": "computed over the whole log, the future included",
                          "label_source": "how a label was made, not a property of the return"},
             "evaluate_on": ["expert", "expert_audit"], "noisy_sources": ["rule", "cluster"],
             "metric": "precision at the daily capacity, with average precision; never accuracy",
             "links": [COPIED, OTHER_STORE, REUSED]}
    return Replica(sales=sales, returns=returns, truth=truth, facts=facts)


def save(replica: Replica, directory: Path) -> dict[str, Path]:
    """Write the two files a person would upload, and a note that they are synthetic (the truth is never written)."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"sales": directory / SALES_FILE, "returns": directory / RETURNS_FILE, "info": directory / INFO_FILE}
    replica.sales.to_csv(paths["sales"], index=False)
    replica.returns.to_csv(paths["returns"], index=False)
    paths["info"].write_text(json.dumps({**LABELS, "replica": "retail_returns", "files": [SALES_FILE, RETURNS_FILE],
                                         "question": "Staff file fake returns on copied bills, mostly late in the evening. Help us catch them. Our labels come "
                                                     "from our rules, the loss-prevention team's reviews and clustering; the team checks five returns a day."},
                                        indent=2))
    return paths
