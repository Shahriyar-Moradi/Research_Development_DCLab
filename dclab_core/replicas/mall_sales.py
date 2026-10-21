"""Mall sales, invented: each store's daily visitors and its daily sales by category, with the causes of a change planted (the metric_change replica).

A retail group with brands in several malls asks why a number moved: a brand head wants to know why the brand's sales are down in one mall against
last year, in minutes and with the reason. It has two tables: the door counters' visitors (one row per store a day) and the tills' sales (one row per
store, category and day: the paying transactions and their amount). A common "ask your data" question with a root cause:
sales = footfall x conversion x basket, and the answer is which factor moved, where, and by how much.

The causes planted in the question month (the last month of the data, against the same month a year before), one of each per seed:

* **A footfall drop in one place.** Every store of one mall loses about a fifth of its visitors (an entrance closed for works): the brand's sales
  there fall with them, while conversion and basket hold.
* **A conversion drop in one category.** One brand's largest category converts about a third less in every mall (sizes out of stock): visitors
  come, fewer buy, and only that category's rate moved.
* **A basket drop from the mix.** Another brand's average basket falls because a promotion moved its sales toward its cheapest category: each
  category's basket holds; the mix moved, not the rate.
* **A data gap.** One store's door counter failed for part of the month: some days have no visitor row, a few read zero, while the tills sold as
  usual. Summing its visitors says footfall fell; the honest answer is that the month cannot be decomposed until the days are back.
* **No change.** One brand's sales moved only within their usual day-to-day variation.

Beside them, questions the data cannot answer at all (a competitor's opening, a profit margin with no cost table). ``truth`` lists every question
with what a careful analyst finds: the factor that moved first, the segment that moved it, or the refusal and what is missing. Invented data: an
answer on it says whether the analysis is sound, never why any real number changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dclab_core.replicas import LABELS

FOOTFALL_FILE, SALES_FILE, INFO_FILE = "footfall.csv", "sales.csv", "replica.json"
MALLS = ("Harbour", "Riverside", "Hilltop", "Old Town", "Lakeside", "Northfield")
BRANDS = ("Arlo", "Bexa", "Corin", "Dune", "Elva", "Fenn", "Gala", "Hollis")
CATEGORIES = ("clothing", "shoes", "accessories", "home")
CONVERSION = {"clothing": 0.07, "shoes": 0.035, "accessories": 0.05, "home": 0.02}   # buying visits per visitor, by category
BASKET = {"clothing": 62.0, "shoes": 85.0, "accessories": 18.0, "home": 40.0}      # the average paying transaction, by category
START, DAYS = "2024-10-01", 730                                                       # two years: the last month against the same month a year before
FOOTFALL_DROP, CONVERSION_DROP, GAP_DAYS = 0.2, 0.35, 12


@dataclass(frozen=True)
class Replica:
    """What a person would upload, plus what only the tests may see."""

    footfall: pd.DataFrame   # date, mall, brand, visitors (one row per store a day; the gap's days missing or zero)
    sales: pd.DataFrame      # date, mall, brand, category, transactions, sales
    period: tuple[str, str]  # the question month, first and last day
    compare: tuple[str, str]  # the same month a year before
    truth: list[dict[str, Any]] = field(default_factory=list)  # for tests only: each question and what a careful analyst finds
    info: dict[str, Any] = field(default_factory=dict)


def _month(day: pd.Timestamp) -> tuple[str, str]:
    first = day.replace(day=1)
    return str(first.date()), str((first + pd.offsets.MonthEnd(0)).date())


def generate(seed: int = 0) -> Replica:
    """The two tables over two years for every store (each brand in four or five of the six malls), the causes planted in the last month, each
    on brands and stores of its own so no question reads another's cause."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, periods=DAYS, freq="D")
    period = _month(days[-1])
    compare = (str((pd.Timestamp(period[0]) - pd.DateOffset(years=1)).date()), str((pd.Timestamp(period[1]) - pd.DateOffset(years=1)).date()))
    in_period = (days >= period[0]) & (days <= period[1])
    by_brand = {b: [str(m) for m in rng.choice(MALLS, size=int(rng.integers(4, 6)), replace=False)] for b in BRANDS}
    mall_size = dict(zip(MALLS, rng.uniform(0.7, 1.4, len(MALLS))))
    order = [BRANDS[i] for i in rng.permutation(len(BRANDS))]
    drop_mall = str(rng.choice(MALLS))
    footfall_brand = next(b for b in order if drop_mall in by_brand[b])
    others = [b for b in order if b != footfall_brand]
    conversion_brand, mix_brand, steady_brand, gap_brand = others[:4]
    by_brand[steady_brand] = [m for m in by_brand[steady_brand] if m != drop_mall]  # the steady brand stays out of the mall whose footfall fell
    gap_mall = next(m for m in by_brand[gap_brand] if m != drop_mall)  # one store's door counter fails: its own questions only
    conversion_category = "clothing"  # the largest category of every brand: the drop shows in the total
    stores = [(m, b) for b in BRANDS for m in by_brand[b]]
    weekday = np.array([0.85, 0.8, 0.85, 0.95, 1.1, 1.35, 1.2])[days.dayofweek]
    season = 1 + 0.12 * np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 80) / 365.25)
    trend = 1 + 0.03 * np.arange(DAYS) / 365.25  # three per cent a year more visitors
    footfall_rows, sales_rows = [], []
    for mall, brand in stores:
        base = 900 * mall_size[mall] * rng.uniform(0.6, 1.5)
        visitors = base * weekday * season * trend * rng.lognormal(0, 0.08, DAYS)
        if mall == drop_mall:
            visitors = np.where(in_period, visitors * (1 - FOOTFALL_DROP), visitors)
        visitors = np.round(visitors).astype(int)
        lift = rng.uniform(0.8, 1.2)  # the store's own way of converting
        for c in CATEGORIES:
            rate = np.full(DAYS, CONVERSION[c] * lift)
            if brand == conversion_brand and c == conversion_category:
                rate = np.where(in_period, rate * (1 - CONVERSION_DROP), rate)
            if brand == mix_brand:  # the promotion: the cheapest category sells more, the others a little less, in the month
                rate = np.where(in_period, rate * (2.2 if c == "accessories" else 0.9), rate)
            transactions = rng.poisson(visitors * rate)
            basket = BASKET[c] * rng.lognormal(0, 0.05, DAYS)
            amount = np.round(transactions * basket, 2)
            sales_rows.append(pd.DataFrame({"date": days, "mall": mall, "brand": brand, "category": c, "transactions": transactions, "sales": amount}))
        footfall_rows.append(pd.DataFrame({"date": days, "mall": mall, "brand": brand, "visitors": visitors}))
    footfall = pd.concat(footfall_rows, ignore_index=True)
    sales = pd.concat(sales_rows, ignore_index=True)
    gap_days = sorted(rng.choice(np.flatnonzero(in_period), size=GAP_DAYS, replace=False))
    zero_days, missing_days = gap_days[:3], gap_days[3:]
    in_gap = footfall["mall"].eq(gap_mall) & footfall["brand"].eq(gap_brand)
    footfall.loc[in_gap & footfall["date"].isin(days[zero_days]), "visitors"] = 0  # the counters read zero while the tills sold
    footfall = footfall[~(in_gap & footfall["date"].isin(days[missing_days]))].reset_index(drop=True)  # and some days have no row at all
    for frame in (footfall, sales):
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    month = pd.Timestamp(period[0]).strftime("%B %Y")
    truth = [
        {"id": "footfall_drop", "question": f"Why are {footfall_brand}'s sales down in {drop_mall} in {month} against last year?", "template": "why_changed",
         "params": {"metric": "sales", "filters": {"brand": footfall_brand, "mall": drop_mall}, "period": list(period), "compare": "last_year"},
         "first_factor": "footfall", "direction": "down"},
        {"id": "footfall_where", "question": f"Why is footfall down for {footfall_brand} in {month} against last year?", "template": "why_changed",
         "params": {"metric": "footfall", "filters": {"brand": footfall_brand}, "period": list(period), "compare": "last_year"},
         "first_factor": "any", "first_segment": {"dimension": "mall", "value": drop_mall}, "direction": "down"},  # the brand's other malls may hide it
        {"id": "conversion_drop", "question": f"Why did {conversion_brand}'s sales drop in {month} against last year?", "template": "why_changed",
         "params": {"metric": "sales", "filters": {"brand": conversion_brand}, "period": list(period), "compare": "last_year"},
         "first_factor": "conversion", "first_segment": {"dimension": "category", "value": conversion_category}, "direction": "down"},
        {"id": "mix_basket", "question": f"Why did {mix_brand}'s average basket fall in {month} against last year?", "template": "why_changed",
         "params": {"metric": "basket", "filters": {"brand": mix_brand}, "period": list(period), "compare": "last_year"},
         "first_factor": "basket", "first_segment": {"dimension": "category", "value": "accessories"}, "effect": "mix", "direction": "down"},
        {"id": "no_change", "question": f"Why did {steady_brand}'s sales change in {month} against last year?", "template": "why_changed",
         "params": {"metric": "sales", "filters": {"brand": steady_brand}, "period": list(period), "compare": "last_year"},
         "first_factor": None, "direction": "none"},
        {"id": "data_gap", "question": f"Why are {gap_brand}'s sales down in {gap_mall} in {month} against last year?", "template": "why_changed",
         "params": {"metric": "sales", "filters": {"brand": gap_brand, "mall": gap_mall}, "period": list(period), "compare": "last_year"},
         "refused": True, "missing": "footfall", "missing_days": len(gap_days)},
        {"id": "competitor", "question": f"Did the new competitor that opened near {drop_mall} cause the drop in {month}?", "template": None,
         "refused": True, "missing": "competitor"},
        {"id": "margin", "question": f"Why did {conversion_brand}'s profit margin fall in {month}?", "template": None, "refused": True, "missing": "margin"},
    ]
    m_name, year = pd.Timestamp(period[0]).strftime("%B"), pd.Timestamp(period[0]).year
    paraphrases = {  # the same questions in other words, written before any model answered them (a model's arm is measured on these)
        "footfall_drop": f"How come {footfall_brand} took less money at {drop_mall} this {m_name} than the {m_name} before?",
        "footfall_where": f"Where did {footfall_brand} lose people coming through the door in {m_name} {year}, compared with a year earlier?",
        "conversion_drop": f"What explains {conversion_brand}'s weaker takings in {m_name} {year} versus {m_name} {year - 1}?",
        "mix_basket": f"Shoppers at {mix_brand} spent less each time they paid in {m_name} {year} than a year before. What happened?",
        "no_change": f"Did anything really move for {steady_brand}'s takings in {m_name} {year} compared to the year before?",
        "data_gap": f"How come {gap_brand} at {gap_mall} looks weaker in {m_name} {year} than a year ago?",
        "competitor": f"Is the rival shop that opened next to {drop_mall} the reason we did worse in {m_name} {year}?",
        "margin": f"Why are {conversion_brand}'s profits thinner in {m_name} {year}?",
    }
    truth = [{**t, "paraphrase": paraphrases[t["id"]]} for t in truth]
    info = {**LABELS, "kind": "metric_change", "seed": seed, "period": list(period), "compare": list(compare), "stores": len(stores),
            "planted": {"footfall_drop": {"mall": drop_mall, "share": FOOTFALL_DROP}, "conversion_drop": {"brand": conversion_brand, "category": conversion_category,
                                                                                                        "share": CONVERSION_DROP},
                        "mix": {"brand": mix_brand, "category": "accessories"}, "data_gap": {"mall": gap_mall, "brand": gap_brand, "days": len(gap_days),
                                                                                                    "zero_days": len(zero_days)},
                        "steady": steady_brand}}
    return Replica(footfall=footfall, sales=sales, period=period, compare=compare, truth=truth, info=info)


def layer() -> Any:
    """The group's approved definitions: three sums, two ratios, sales = footfall x conversion x basket, the dimensions each table names,
    and what the data does not hold."""
    from dclab_core import metric_change as mc

    return mc.SemanticLayer(
        tables=[mc.Table(name="footfall", date="date"), mc.Table(name="sales", date="date")],
        metrics=[mc.Metric(name="footfall", label="Footfall", kind="sum", table="footfall", column="visitors", unit="visitors",
                           synonyms=["visitors", "traffic", "visits"], approved_by="the group's analytics team"),
                 mc.Metric(name="transactions", label="Paying transactions", kind="sum", table="sales", column="transactions",
                           synonyms=["transactions", "purchases", "receipts"], approved_by="the group's analytics team"),
                 mc.Metric(name="sales", label="Sales", kind="sum", table="sales", column="sales", unit="currency",
                           synonyms=["revenue", "turnover"], approved_by="finance"),
                 mc.Metric(name="conversion", label="Conversion", kind="ratio", numerator="transactions", denominator="footfall",
                           synonyms=["conversion rate"], approved_by="the group's analytics team"),
                 mc.Metric(name="basket", label="Average basket", kind="ratio", numerator="sales", denominator="transactions", unit="currency",
                           synonyms=["basket", "average ticket", "basket size", "average transaction"], approved_by="finance")],
        dimensions=[mc.Dimension(name="mall", label="Mall", columns={"footfall": "mall", "sales": "mall"}, synonyms=["malls", "centre", "site"]),
                    mc.Dimension(name="brand", label="Brand", columns={"footfall": "brand", "sales": "brand"}, synonyms=["brands", "store"]),
                    mc.Dimension(name="category", label="Category", columns={"sales": "category"}, synonyms=["categories", "department"])],
        factorizations=[mc.Factorization(metric="sales", factors=["footfall", "conversion", "basket"])],
        not_held=[mc.NotHeld(name="competitor", synonyms=["competitors", "rival", "rivals"],
                             needs="a table of competitors' openings and promotions by mall and date"),
                  mc.NotHeld(name="margin", synonyms=["profit", "profits", "cost", "costs"], needs="the cost of the goods sold by brand and category"),
                  mc.NotHeld(name="weather", synonyms=["rain", "temperature"], needs="the daily weather at each mall"),
                  mc.NotHeld(name="price", synonyms=["prices", "pricing"], needs="each product's price by date")])


def save(replica: Replica, directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"footfall": directory / FOOTFALL_FILE, "sales": directory / SALES_FILE, "info": directory / INFO_FILE}
    replica.footfall.to_csv(paths["footfall"], index=False)
    replica.sales.to_csv(paths["sales"], index=False)
    paths["info"].write_text(json.dumps(replica.info, indent=1) + "\n", encoding="utf-8")
    return paths


__all__ = ["Replica", "generate", "layer", "save", "MALLS", "BRANDS", "CATEGORIES", "FOOTFALL_DROP", "CONVERSION_DROP", "GAP_DAYS"]
