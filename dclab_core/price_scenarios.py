"""Single-unit offers and fixed-horizon renewals. Only pre-offer inputs enter a response estimate."""

from __future__ import annotations

import math
from bisect import insort
from collections import defaultdict
from decimal import ROUND_CEILING, Decimal
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PriceScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    price: float = Field(gt=0, le=1e12)
    audience: int = Field(ge=1, le=1_000_000, strict=True)
    target_revenue: float = Field(ge=0, le=1e16)
    probabilities: list[float | None] = Field(min_length=3, max_length=3)
    outcome: Literal["purchase", "churn"] = "purchase"
    independent: bool = False

    @model_validator(mode="after")
    def valid_rates(self):
        if any(p is not None and not 0 <= p <= 1 for p in self.probabilities):
            raise ValueError("Each event probability must be between zero and one.")
        return self


MULTIPLES = (1.0, 1.05, 1.10)
MIN_CELL = 10


def downside(counts, rates, required, seed=19):
    """P(total purchases < required) for independent Bernoulli trials grouped by rate."""
    if required <= 0:
        return 0.0
    if required > sum(counts):
        return 1.0
    if len(counts) == 1 and counts[0] <= 200:
        n, p = counts[0], rates[0]
        return min(1.0, math.fsum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(required)))
    rng = np.random.default_rng(seed)
    draws = sum(rng.binomial(n, p, size=20_000) for n, p in zip(counts, rates))
    return float(np.mean(draws < required))


def price_row(price, target, outcome, counts, rates, independent):
    required = int((Decimal(str(target)) / Decimal(str(price))).to_integral_value(rounding=ROUND_CEILING))
    row = {
        "price": price,
        "break_even_units": required,
        "audience": sum(counts),
        "expected_events": None,
        "expected_revenue": None,
        "event_rate": None,
        "downside_probability": None,
    }
    if rates is None:
        return row
    event_count = sum(n * p for n, p in zip(counts, rates))
    sale_rates = rates if outcome == "purchase" else [1 - p for p in rates]
    row.update(
        expected_events=event_count,
        event_rate=event_count / sum(counts),
        expected_revenue=price * sum(n * p for n, p in zip(counts, sale_rates)),
    )
    if independent:
        row["downside_probability"] = downside(counts, sale_rates, required)
    return row


def assumptions(request: PriceScenario):
    rows = []
    for multiple, p in zip(MULTIPLES, request.probabilities):
        price = float(Decimal(str(request.price)) * Decimal(str(multiple)))
        rows.append(
            price_row(
                price,
                request.target_revenue,
                request.outcome,
                [request.audience],
                None if p is None else [p],
                request.independent,
            )
        )
    return {
        "source": "user assumption",
        "outcome": request.outcome,
        "rows": rows,
        "risk_assumption": "independent offers" if request.independent else "dependence unspecified",
    }


def required(frame, fields):
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame[list(fields)].isna().any().any():
        raise ValueError("Missing values in required fields: " + ", ".join(fields))


def histories(offers, events):
    """Sweep by availability, then take the five most recent eligible events by occurrence."""
    required(offers, ["offer_id", "customer_id", "offered_at"])
    required(events, ["event_id", "customer_id", "occurred_at", "available_at", "discounted"])
    if offers.offer_id.duplicated().any():
        raise ValueError("Each offer_id must identify one opportunity.")
    columns = ["event_id", "customer_id", "occurred_at", "available_at", "discounted"]
    events = events[columns + [k for k in ("kind", "paid_price") if k in events]].drop_duplicates().copy()
    if "kind" not in events:
        events["kind"] = "purchase"
    if "paid_price" not in events:
        events["paid_price"] = np.nan
    if events.event_id.duplicated().any() or not events.discounted.isin([0, 1]).all():
        raise ValueError("Conflicting event IDs or invalid discount indicators.")
    for col in ("occurred_at", "available_at"):
        events[col] = pd.to_datetime(events[col], utc=True, errors="raise")
    events["ready"] = events[["occurred_at", "available_at"]].max(axis=1)
    stream = iter(events.sort_values("ready").itertuples())
    pending = next(stream, None)
    histories_by_customer = defaultdict(list)
    result = {}
    ordered = offers.assign(cutoff=pd.to_datetime(offers.offered_at, utc=True)).sort_values("cutoff")
    for offer in ordered.itertuples():
        while pending is not None and pending.ready < offer.cutoff:
            recent = histories_by_customer[pending.customer_id]
            insort(
                recent,
                (
                    pending.occurred_at,
                    str(pending.event_id),
                    pending.kind,
                    int(pending.discounted),
                    pending.paid_price,
                ),
            )
            if len(recent) > 5:
                recent.pop(0)
            pending = next(stream, None)
        recent = histories_by_customer.get(offer.customer_id, ())
        n = len(recent)
        purchases = [item for item in recent if item[2] == "purchase"]
        share = sum(item[3] for item in purchases) / len(purchases) if purchases else None
        result[offer.offer_id] = (n, share, purchases[-1][4] if purchases else None)
    return pd.DataFrame.from_dict(
        result, orient="index", columns=["history_count", "discount_share", "last_paid"]
    )


def study_rows(offers, events, outcomes, horizon_days):
    """Split before fitting, discard immature labels, allowlist fields at the model boundary."""
    required(offers, ["offer_id", "customer_id", "offered_at", "price", "segment", "split", "currency"])
    required(outcomes, ["offer_id"])
    if offers.currency.nunique() != 1 or not offers.split.isin(["train", "score"]).all():
        raise ValueError("Use one currency and explicit train/score partitions.")
    x = offers[["offer_id", "customer_id", "offered_at", "price", "segment", "split"]].copy()
    x["price"] = pd.to_numeric(x.price, errors="raise")
    if not np.isfinite(x.price).all() or (x.price <= 0).any():
        raise ValueError("All observed offer prices must be positive and finite.")
    x["offered_at"] = pd.to_datetime(x.offered_at, utc=True)
    x = x.join(histories(x, events), on="offer_id")
    x["history_band"] = np.where(
        x.discount_share.isna(), "unknown", np.where(x.discount_share >= 0.6, "high", "low")
    )
    train, cohort = x[x.split == "train"], x[x.split == "score"]
    if train.empty or cohort.empty or train.offered_at.max() >= cohort.offered_at.min():
        raise ValueError("Score offers must follow all training offers.")
    if set(train.customer_id) & set(cohort.customer_id) or x.customer_id.duplicated().any():
        raise ValueError("This study requires one offer per person and disjoint train/score customers.")
    if not set(outcomes.offer_id).issubset(set(offers.offer_id)):
        raise ValueError("An outcome references an unknown offer.")
    if not {"outcome", "available_at"}.issubset(outcomes.columns):
        raise ValueError("Outcome records require outcome and available_at.")
    labels = outcomes[outcomes.offer_id.isin(train.offer_id)][["offer_id", "available_at", "outcome"]]
    labels = labels.dropna(subset=["outcome", "available_at"])
    if labels.offer_id.duplicated().any() or not labels.outcome.isin([0, 1]).all():
        raise ValueError("Training outcomes must be unique, observed binary labels.")
    train = train.merge(labels, on="offer_id", validate="one_to_one")
    available = pd.to_datetime(train.available_at, utc=True)
    horizon = train.offered_at + pd.to_timedelta(horizon_days, unit="D")
    cutoff = cohort.offered_at.min()
    train = train[(available >= horizon) & (available < cutoff)]
    return train, cohort


def standardized(train, cohort, prices, fields, minimum=MIN_CELL):
    """Direct standardization to a fixed cohort with exact support in each prespecified stratum."""
    weights = cohort.groupby(fields, sort=True).size() if fields else {(): len(cohort)}
    grouped = train.groupby([*fields, "price"]).outcome.agg(["mean", "size"])
    counts, rates, sizes = [], [[] for _ in prices], [[] for _ in prices]
    for group, count in weights.items():
        key = group if isinstance(group, tuple) else (group,)
        counts.append(int(count))
        for j, price in enumerate(prices):
            index = (*key, price) if fields else price
            if index not in grouped.index or grouped.loc[index, "size"] < minimum:
                return None
            rates[j].append(float(grouped.loc[index, "mean"]))
            sizes[j].append(int(grouped.loc[index, "size"]))
    return counts, rates, sizes


def effect_interval(counts, a, b, na, nb):
    """Independent people in disjoint assigned arms; fixed target-population weights."""
    w = np.array(counts) / sum(counts)
    a, b, na, nb = map(np.asarray, (a, b, na, nb))
    change = float(w @ (b - a))
    # Agresti-Caffo smoothing for uncertainty, including empty outcome classes.
    pa, pb = (a * na + 1) / (na + 2), (b * nb + 1) / (nb + 2)
    variance = np.sum(w * w * (pa * (1 - pa) / (na + 2) + pb * (1 - pb) / (nb + 2)))
    width = 1.96 * math.sqrt(variance)
    return {
        "change_pp": 100 * change,
        "interval95_pp": [max(-100, 100 * (change - width)), min(100, 100 * (change + width))],
    }


class StudySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    base_price: float = Field(gt=0, le=1e12)
    horizon_days: int = Field(ge=1, le=3650)
    outcome: Literal["purchase", "churn"]
    design: Literal["randomized", "observational", "unknown"]
    design_basis: str = Field(min_length=1, max_length=400)


def fit_study(offers, events, outcomes, spec):
    train, cohort = study_rows(offers, events, outcomes, spec.horizon_days)
    prices = [float(Decimal(str(spec.base_price)) * Decimal(str(m))) for m in MULTIPLES]
    fields = ["segment", "history_band"]
    supported = [standardized(train, cohort, [price], fields) for price in prices]
    result = {
        "prices": prices,
        "outcome": spec.outcome,
        "design": spec.design,
        "design_basis": spec.design_basis,
        "horizon_days": spec.horizon_days,
        "audience": len(cohort),
        "mature_training": len(train),
        "excluded_training": int((offers.split == "train").sum()) - len(train),
        "currency": str(offers.currency.iloc[0]),
    }
    last_paid = pd.to_numeric(cohort.last_paid, errors="coerce")
    valid = last_paid[np.isfinite(last_paid) & (last_paid > 0)]
    result["prior_price_count"] = len(valid)
    result["mean_change_from_last_paid"] = [
        float((price / valid - 1).mean()) if len(valid) else None for price in prices
    ]
    if not any(supported):
        return dict(
            result,
            source="insufficient evidence",
            reason="Fewer than ten matured offers per price in one or more cohort strata.",
        )
    counts = next(part[0] for part in supported if part is not None)
    result.update(
        source="model prediction",
        counts=counts,
        rates=[part[1][0] if part is not None else None for part in supported],
    )
    if not all(supported):
        result["reason"] = "Some prices lack ten mature offers in every cohort stratum."
    if spec.design != "unknown" and supported[0] and any(supported[1:]):
        result["effect_source"] = (
            "randomized estimate"
            if spec.design == "randomized"
            else "adjusted observational estimate assuming no unmeasured confounding"
        )
        parts = (
            [
                standardized(train, cohort, [price], []) if part else None
                for price, part in zip(prices, supported)
            ]
            if spec.design == "randomized"
            else supported
        )
        weights, base_rates, base_sizes = parts[0]
        result["effects"] = [
            effect_interval(weights, base_rates[0], part[1][0], base_sizes[0], part[2][0]) if part else None
            for part in parts[1:]
        ]
    return result


def score_study(study, target, independent):
    rows = []
    for j, price in enumerate(study["prices"]):
        counts = study.get("counts", [study["audience"]])
        rates = study["rates"][j] if "rates" in study else None
        rows.append(price_row(price, target, study["outcome"], counts, rates, independent))
    return {k: v for k, v in study.items() if k not in ("counts", "rates")} | {"rows": rows}
