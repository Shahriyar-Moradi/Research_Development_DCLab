"""What to collect next (34.14): which column groups the score leans on, which usual kinds of data are missing, and where
more rows still help.

Three readings, each said for what it is:

- **Adds beyond the others** (measured on this data): the training folds' out-of-fold predictions with every column, and
  with each group's columns left out (made constant) in turn; a group's reading is how much the score falls without it,
  with a paired bootstrap interval on the same rows (over whole groups or entities when the split keeps them together).
  A group is named only when its interval is above zero; otherwise it is "cannot tell". What two groups both hold (a fee
  and the plan it is priced by) reads as little for each: the reading is what a group adds beyond the others, never all
  it carries. The host passes ``predict_without(columns)``, out-of-fold on its training folds; the holdout is never read.
- **Usually informative, not here** (prior knowledge, never measured): each business domain's kinds of data that carried
  such problems before (churn: usage, support contacts...), matched to the table's column names by whole word parts; a
  kind no column matches is named as missing, in the domain's order. It says what is worth asking for, never that it would
  help. A problem in no known domain names nothing missing, and its columns are weighed one by one.
- **Where more rows still help** (34.13's segments whose score still rises with more rows) and the open data questions of
  the data-need agent (``mapping.coverage``), passed through as they are.

Active learning stays an option and never the default: both of its gates failed (ALQ-001, ALQ-002).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Optional

import numpy as np
from pydantic import Field

from .contracts import CONTRACTS, Contract, Versioned, _version

# (kind, the words a column's name holds when it is of that kind (whole word parts; "a_b" is two parts in a row), what it is),
# in the domain's usual order of value. A name's parts: split at _, -, spaces, digits and lower-to-upper case changes.
SIGNALS: dict[str, list[tuple[str, tuple[str, ...], str]]] = {
    "churn": [
        ("tenure", ("tenure", "signup", "joined", "customer_since", "member_since", "months_as_customer"), "how long the customer has been one"),
        ("contract", ("contract", "plan", "tier", "subscription", "package", "tech_support", "online_security", "online_backup",
                      "streaming", "device_protection", "add_on", "addon"), "the plan, the contract and the services on it"),
        ("billing", ("fee", "charge", "price", "bill", "billing", "amount", "spend", "revenue", "payment_method", "paperless"), "what they pay and how"),
        ("payments", ("late", "overdue", "arrears", "missed_payment", "failed_payment", "dunning", "late_payment"), "late or failed payments"),
        ("support", ("support_call", "support_ticket", "support_contact", "ticket", "complaint", "call_center", "contact"),
         "support contacts and complaints"),
        ("usage", ("usage", "login", "session", "active", "minute", "data_gb", "visit", "last_seen", "last_login", "streams_watched"),
         "how much they use the product"),
        ("satisfaction", ("nps", "satisfaction", "rating", "survey", "csat"), "what they said about it"),
    ],
    "fraud": [
        ("velocity", ("per_hour", "per_day", "velocity", "last_1h", "last_24h", "count_1h", "count_24h", "orders_today", "attempt"),
         "how many actions in a short time"),
        ("amount", ("amount", "value", "fare", "price", "total"), "the amount against the account's usual"),
        ("account_age", ("account_age", "created", "signup", "days_since_signup", "tenure"), "how new the account is"),
        ("device", ("device", "ip", "browser", "fingerprint", "user_agent"), "the device and network used"),
        ("location", ("lat", "lon", "latitude", "longitude", "distance", "country", "city", "geo", "zip"), "where it happened against where it usually does"),
        ("identity_links", ("shared", "linked", "accounts_per", "cards_per", "same_phone", "same_card"), "links to other accounts (a shared phone or card)"),
    ],
    "credit": [
        ("income", ("income", "salary", "earnings"), "income"),
        ("debt", ("debt", "loan", "balance", "utilization", "limit", "bill_amt", "mortgage"), "existing debt and how much of it is used"),
        ("history", ("late", "default", "delinquent", "delinquency", "history", "missed", "pay", "repayment"), "repayment history"),
        ("employment", ("employ", "employment", "job", "occupation", "years_at"), "employment"),
        ("age", ("age",), "age"),
    ],
    "delivery": [
        ("distance", ("distance", "km", "lat", "lon"), "the distance"),
        ("time_of_day", ("hour", "weekday", "created_at", "order_time"), "when it was ordered"),
        ("load", ("orders_open", "queue", "busy", "drivers", "couriers", "load"), "how busy the fleet was"),
        ("weather", ("weather", "rain", "temp", "temperature", "storm"), "the weather"),
        ("courier", ("courier", "driver", "biker", "rider"), "who delivered it"),
    ],
    "demand": [
        ("calendar", ("date", "day", "week", "month", "holiday", "weekday"), "the calendar and holidays"),
        ("price", ("price", "discount", "promo", "promotion"), "price and promotions"),
        ("history", ("lag", "previous", "last_week", "rolling", "sales", "units_sold"), "past demand"),
        ("weather", ("weather", "rain", "temp", "temperature"), "the weather"),
        ("stock", ("stock", "inventory", "available"), "what was in stock"),
    ],
    # 0.117.0: four more domains (the plan's item 13), each the kinds a careful data scientist would ask for first
    "maintenance": [
        ("sensors", ("vibration", "temperature", "temp", "pressure", "voltage", "current", "rpm", "sensor", "reading"),
         "sensor readings (vibration, temperature, pressure) against the machine's own normal"),
        ("operating_age", ("runtime", "run_hours", "operating_hours", "cycles", "installed", "age"), "how long and how hard it has run"),
        ("maintenance_history", ("maintenance", "service", "repair", "replaced", "overhaul", "last_service"), "past maintenance and repairs"),
        ("errors", ("error", "alarm", "fault", "warning", "alert", "event_code"), "error codes and alarms"),
        ("load", ("load", "throughput", "speed", "utilization", "duty"), "the load it runs under"),
        ("environment", ("humidity", "ambient", "dust", "weather"), "where it runs"),
    ],
    "conversion": [
        ("recency_frequency", ("last_purchase", "recency", "frequency", "past_purchases", "purchases", "orders", "days_since"),
         "how recently and how often they bought"),
        ("engagement", ("visit", "session", "page", "pageviews", "click", "open", "email_open", "time_on_site"), "how they engaged before"),
        ("offer", ("offer", "discount", "coupon", "promo", "campaign", "treatment"), "the offer and the campaign they got"),
        ("channel", ("channel", "source", "referrer", "utm", "medium", "device"), "where they came from"),
        ("value", ("spend", "revenue", "basket", "order_value", "monetary", "amount"), "how much they spend"),
    ],
    "claims": [
        ("policy", ("policy", "premium", "coverage", "deductible", "sum_insured"), "the policy and its cover"),
        ("claim_history", ("prior_claim", "previous_claim", "past_claim", "claim_count", "claims"), "past claims"),
        ("insured", ("vehicle", "property", "driver", "occupation", "age"), "what or who is insured"),
        ("incident", ("incident", "accident", "damage", "injury", "severity", "witness"), "the incident itself"),
        ("timing", ("policy_start", "days_to_claim", "report_delay", "reported"), "how soon after the policy began, and how fast it was reported"),
    ],
    "returns": [
        ("item", ("category", "size", "product", "sku", "item", "color", "brand"), "the item (its category and size)"),
        ("customer_history", ("return_rate", "past_returns", "previous_returns", "returns_count"), "the customer's past returns"),
        ("order", ("amount", "discount", "price", "payment", "quantity", "basket"), "the order and how it was paid"),
        ("timing", ("days_since", "delivered", "returned_at", "purchase_date", "hours_since"), "how long after the purchase"),
        ("channel", ("store", "online", "channel", "cashier", "seller"), "where it was bought and returned"),
    ],
}
DOMAIN_WORDS: dict[str, tuple[str, ...]] = {
    "churn": ("churn", "unsubscribe", "attrition", "retention", "subscriber", "subscription", "renew"),
    "fraud": ("fraud", "fake", "chargeback", "scam", "abuse", "stolen"),
    "credit": ("credit", "loan", "default", "repay", "lend"),
    "delivery": ("delivery", "deliveries", "courier", "arrival", "eta"),
    "demand": ("demand", "forecast", "how many", "units sold", "daily sales", "weekly sales"),
    # after the first five, so a problem they named keeps its domain; words that rarely mean anything else
    "maintenance": ("machine failure", "machine breakdown", "equipment", "sensor", "breakdown", "downtime", "maintenance", "turbine"),
    "conversion": ("conversion", "convert", "lead scoring", "sales lead", "campaign", "respond", "response rate", "uplift", "who buys"),
    "claims": ("claim", "insurance", "insured", "policyholder", "underwriting"),
    "returns": ("refund", "returned", "product return", "item return", "return rate"),
}
ACTIVE_LEARNING = "Active learning stays an option, never the default here: both of its gates failed (ALQ-001, ALQ-002)."
MAX_ONE_BY_ONE = 20  # columns weighed one by one when no domain groups them
BOOTSTRAP = 200


def tokens(name: str) -> list[str]:
    """A column name's word parts: split at _, -, spaces, digits and a lower-to-upper case change ("TechSupport" is tech,
    support; "PAY_0" is pay)."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name))
    return [t for t in re.split(r"[^A-Za-z]+", spaced.lower()) if t]


def _matches(parts: list[str], word: str) -> bool:
    """A word (one part, or several parts in a row) found among a name's parts, a part also matching in the plural."""
    want = word.strip("_").split("_")

    def same(part: str, w: str) -> bool:
        return part == w or part == w + "s" or part == w + "es" or (w.endswith("y") and part == w[:-1] + "ies")

    return any(all(same(parts[i + j], w) for j, w in enumerate(want)) for i in range(len(parts) - len(want) + 1))


def domain_of(text: str) -> Optional[str]:
    """The business domain a problem's words name (the first whose words appear as whole words), or None."""
    low = f" {str(text or '').lower()} "
    return next((d for d, words in DOMAIN_WORDS.items() if any(re.search(rf"\b{re.escape(w)}(s|es)?\b", low) for w in words)), None)


def kind_of(column: str, domain: str) -> Optional[str]:
    """The domain's kind a column's name matches; when several do, the one with the longest matching word."""
    parts = tokens(column)
    best: Optional[tuple[int, int, str]] = None
    for order, (kind, words, _) in enumerate(SIGNALS.get(domain, [])):
        for word in words:
            if _matches(parts, word):
                rank = (len(word), -order)
                if best is None or rank > best[:2]:
                    best = (rank[0], rank[1], kind)
    return best[2] if best else None


def groups(columns: Sequence[str], domain: Optional[str]) -> dict[str, list[str]]:
    """The columns by the domain's kinds (a column of no kind in ``other``); with no domain, each column on its own (the
    first ``MAX_ONE_BY_ONE``)."""
    if domain is None:
        return {str(c): [str(c)] for c in list(columns)[:MAX_ONE_BY_ONE]}
    out: dict[str, list[str]] = {}
    for column in columns:
        out.setdefault(kind_of(column, domain) or "other", []).append(str(column))
    return out


class GroupReading(Contract):
    group: str
    columns: list[str] = Field(default_factory=list)
    score_without: Optional[float] = None
    adds: Optional[float] = Field(default=None, description="How much the score falls without this group, beyond the others (positive: it adds).")
    low: Optional[float] = None
    high: Optional[float] = None
    verdict: Literal["adds", "cannot tell", "adds nothing"] = "cannot tell"


class MissingKind(Contract):
    kind: str
    what: str
    rank: int


class LabelSegment(Contract):
    column: str
    level: str
    rows: int = 0


class CollectAdvice(Versioned):
    """What to collect next (34.14, v1): kept apart by where each part comes from."""

    schema_version: Literal["dclab_core.collect_advice.v1"] = _version("collect_advice")
    domain: Optional[str] = None
    metric: Optional[str] = None
    score_all: Optional[float] = None
    groups: list[GroupReading] = Field(default_factory=list, description="Measured on the training folds: what each group adds beyond the others.")
    matched: dict[str, list[str]] = Field(default_factory=dict, description="The columns each kind was matched to by name, to correct.")
    missing: list[MissingKind] = Field(default_factory=list, description="The domain's usual kinds no column matches: prior knowledge, not measured.")
    label_segments: list[LabelSegment] = Field(default_factory=list, description="Where the score still rose with more rows (34.13).")
    questions: list[str] = Field(default_factory=list, description="The data-need agent's open questions.")
    assumption: str = ("A group's reading is what it adds beyond the other groups: what two groups both hold (a fee and the plan "
                       "it is priced by, a total and the months that make it) reads as little for each.")
    plain: str = Field(default="", max_length=3000, description="The advice in words; it names columns and kinds, never a level of a column.")
    active_learning: str = ACTIVE_LEARNING


CONTRACTS["collect_advice"] = CollectAdvice


def _paired(y: np.ndarray, full: np.ndarray, without: np.ndarray, metric_fn: Callable[[np.ndarray, np.ndarray], float], sign: float,
            members: Optional[list[np.ndarray]], draws: int, rng: np.random.Generator) -> tuple[Optional[float], Optional[float]]:
    values = []
    for _ in range(draws):
        rows = (np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) if members is not None
                else rng.integers(0, len(y), len(y)))
        try:
            a, b = float(metric_fn(y[rows], full[rows])), float(metric_fn(y[rows], without[rows]))
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(a) and np.isfinite(b):
            values.append(sign * (a - b))
    if len(values) < max(20, draws // 2):
        return None, None
    return round(float(np.quantile(values, 0.05)), 6), round(float(np.quantile(values, 0.95)), 6)


def advise(columns: Sequence[str], *, problem: str = "", domain: Optional[str] = None, y: Optional[Sequence[Any]] = None,
           predict_without: Optional[Callable[[list[str]], Any]] = None, metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
           higher_is_better: bool = True, metric: Optional[str] = None, groups_of_rows: Optional[Sequence[Any]] = None,
           data_value: Optional[Mapping[str, Any]] = None, data_needs: Sequence[Mapping[str, Any]] = (), bootstrap: int = BOOTSTRAP,
           seed: int = 0) -> CollectAdvice:
    """The advice for a table's model columns. ``predict_without(columns)``: out-of-fold predictions on the training rows with
    those columns left out (``[]`` for every column kept), always for the same rows as ``y``; ``metric_fn(y, predictions)``
    the measure; ``groups_of_rows``: the entity of each row, when the split keeps entities together; ``data_value``: a
    ``DataValue`` (34.13) as a dict; ``data_needs``: ``mapping.coverage``'s questions."""
    domain = domain if domain is not None else domain_of(problem)
    by_kind = groups(columns, domain)
    readings: list[GroupReading] = []
    score_all = None
    if predict_without is not None:
        if y is None or metric_fn is None:
            raise ValueError("Weighing the groups needs the rows' outcomes and the measure")
        target = np.asarray(y, dtype=float)
        full = np.asarray(predict_without([]), dtype=float)
        score_all = float(metric_fn(target, full))
        sign = 1.0 if higher_is_better else -1.0
        members = None
        if groups_of_rows is not None:
            codes = np.unique(np.asarray(groups_of_rows, dtype=object).astype(str), return_inverse=True)[1]
            order = np.argsort(codes, kind="stable")
            members = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
        rng = np.random.default_rng(seed)
        for kind, cols in by_kind.items():
            if kind == "other" or len(cols) == len(columns):
                continue
            without = np.asarray(predict_without(cols), dtype=float)
            score = float(metric_fn(target, without))
            low, high = _paired(target, full, without, metric_fn, sign, members, bootstrap, rng)
            verdict = "cannot tell" if low is None else "adds" if low > 0 else "adds nothing" if high is not None and high <= 0 else "cannot tell"
            readings.append(GroupReading(group=kind, columns=cols[:20], score_without=round(score, 6), adds=round(sign * (score_all - score), 6),
                                         low=low, high=high, verdict=verdict))
        readings.sort(key=lambda r: (r.verdict != "adds", -(r.adds or 0)))
    missing = [MissingKind(kind=k, what=what, rank=i + 1) for i, (k, _, what) in enumerate(SIGNALS.get(domain or "", [])) if k not in by_kind]
    segments = [LabelSegment(column=str(s.get("column")), level=str(s.get("level")), rows=int(s.get("rows") or 0))
                for s in (data_value or {}).get("segments") or [] if s.get("answer") == "likely_help" and s.get("column")][:5]
    questions = [str(q.get("question")) for q in data_needs if q.get("question")][:10]
    parts = []
    named = [r for r in readings if r.verdict == "adds"][:3]
    if named:
        parts.append("On the training folds, the score falls most without " + "; ".join(
            f"{r.group} (the {metric or 'score'} falls by {r.adds:.4f}, 90% interval {r.low} to {r.high})" for r in named)
            + ": what each adds beyond the other groups.")
    elif readings:
        parts.append("On the training folds, no group's own share of the score could be told from noise: what two groups both hold "
                     "reads as little for each.")
    if missing:
        parts.append(f"Usually informative for {domain} and not in this table: "
                     + "; ".join(f"{m.kind} ({m.what})" for m in missing[:4]) + ". Worth asking for; not measured here.")
    elif domain is None:
        parts.append("The problem's words name no domain DCLab knows usual signals for: no missing kind is named.")
    if segments:
        columns_named = sorted({s.column for s in segments})
        parts.append("In some segments of " + ", ".join(columns_named) + " the score still rose with more rows (34.13): more labelled rows "
                     "there may help.")
    parts.append(ACTIVE_LEARNING)
    return CollectAdvice(domain=domain, metric=metric, score_all=None if score_all is None else round(score_all, 6), groups=readings,
                         matched={k: v for k, v in by_kind.items() if domain is not None}, missing=missing, label_segments=segments,
                         questions=questions, plain=" ".join(parts)[:3000])


__all__ = ["SIGNALS", "DOMAIN_WORDS", "tokens", "domain_of", "kind_of", "groups", "GroupReading", "MissingKind", "LabelSegment",
           "CollectAdvice", "advise"]
