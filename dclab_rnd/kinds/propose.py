"""A kind's fields read from the table (S9.1 of the 100-students plan, 2026-10-25; the live first conversation: a
forecast the card read right ran as a regression, because no kind proposed its own fields and the older written
people typed them in).

``propose_fields(kind, frame, target)`` reads what the table shows (a forecast's series and dates, a ranking's users
and items, a time-to-event's duration, event and start) and returns a :class:`FieldProposal`: the fields, each with
its reason, the ones assumed (said as assumptions, changeable on the Solution page) and the ones only the person can
give (``missing``: asked, never guessed). A kind whose data lives in other tables (entity tables, streams' incidents)
names them as missing. Names and values are read, never sent anywhere; nothing here decides the kind: the proposal
follows the card's family (S9.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd


@dataclass
class FieldProposal:
    kind: str
    fields: dict[str, Any] = field(default_factory=dict)
    time_column: Optional[str] = None
    why: list[str] = field(default_factory=list)
    assumed: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "kind_fields": self.fields, "time_column": self.time_column, "why": self.why,
                "assumed": self.assumed, "missing": self.missing}


def _tokens(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", str(name).lower())) - {""}


def _named(frame: pd.DataFrame, words: set[str], exclude: set[str] = frozenset()) -> list[str]:
    return [c for c in frame.columns if c not in exclude and _tokens(c) & words]


def _dates(frame: pd.DataFrame, exclude: set[str] = frozenset()) -> list[str]:
    """Columns whose values read as dates (most of them), the time-named first."""
    out = []
    for c in frame.columns:
        if c in exclude:
            continue
        s = frame[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
            continue
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(200)
            if len(sample) and pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.9 and sample.str.contains(r"\d{4}|\d{1,2}[/-]\d{1,2}").mean() > 0.9:
                out.append(c)
    timed = {"date", "time", "day", "timestamp", "at", "datetime", "week", "month"}
    return sorted(out, key=lambda c: (not (_tokens(c) & timed), list(frame.columns).index(c)))


def _binary(frame: pd.DataFrame, c: str) -> bool:
    values = frame[c].dropna().unique()
    return len(values) == 2


def _repeating_category(frame: pd.DataFrame, c: str) -> bool:
    s = frame[c].dropna()
    return len(s) > 0 and 1 < s.nunique() <= max(2, int(0.5 * len(s)))


# ------------------------------------------------------------------------------------------------ each kind


OUTCOME_WORDS = {"renewed", "churned", "bought", "converted", "cancelled", "canceled", "purchased", "responded", "clicked", "left",
                 "outcome", "label", "target", "retained", "returned"}


def _segmentation(frame: pd.DataFrame, target: Optional[str]) -> FieldProposal:
    p = FieldProposal("segmentation", why=["the segments are found from the columns that describe each row"])
    outcomes = [c for c in _named(frame, OUTCOME_WORDS) if _binary(frame, c)]
    if outcomes:  # an outcome in the table describes the segments afterwards and never forms them (it would be a leak)
        p.fields["outcome_column"] = outcomes[0]
        p.why.append(f"`{outcomes[0]}` is an outcome: it describes each segment afterwards, never forms them")
    return p


TEXT_WORDS = {"text", "body", "message", "comment", "review", "note", "notes", "description", "content", "subject"}
SOURCE_WORDS = {"author", "source", "writer", "sender", "user", "customer", "patient", "account", "channel", "agent", "doctor"}


def _embedded(frame: pd.DataFrame, target: Optional[str]) -> FieldProposal:
    p = FieldProposal("embedded")
    texts = [c for c in frame.columns if c != target and frame[c].dtype == object and frame[c].dropna().astype(str).str.len().mean() > 40]
    texts.sort(key=lambda c: (not (_tokens(c) & TEXT_WORDS), -frame[c].dropna().astype(str).str.len().mean()))
    if texts:
        p.fields.update(item_kind="text", text_column=texts[0])
        p.why.append(f"`{texts[0]}` holds the long texts each row is about")
    else:
        p.missing.append("text_column")
    sources = [c for c in _named(frame, SOURCE_WORDS, {target, *texts}) if _repeating_category(frame, c)]
    if sources:
        p.fields["source_column"] = sources[0]
        p.why.append(f"`{sources[0]}` repeats: rows of one source stay on one side of the split")
    else:
        p.missing.append("source_column")
    return p


SERIES_WORDS = {"series", "store", "shop", "product", "sku", "item", "region", "branch", "site", "location", "category"}


def _forecasting(frame: pd.DataFrame, target: Optional[str]) -> FieldProposal:
    p = FieldProposal("forecasting")
    dates = _dates(frame, {target} if target else set())
    if not dates:
        p.missing.append("time_column")
        return p
    date = dates[0]
    p.time_column = date
    p.why.append(f"`{date}` holds the dates the forecast moves along")
    stamp = pd.to_datetime(frame[date].astype(str), errors="coerce", format="mixed")
    candidates = [c for c in frame.columns if c not in (target, date) and (frame[c].dtype == object or bool(_tokens(c) & SERIES_WORDS))
                  and 1 < frame[c].nunique(dropna=True) < max(3, len(frame) // 3)]
    one_per_date = [c for c in candidates if not pd.DataFrame({"s": frame[c], "d": stamp}).duplicated().any()]
    one_per_date.sort(key=lambda c: (not (_tokens(c) & {"series"}), frame[c].nunique()))
    if one_per_date:
        p.fields["series_column"] = one_per_date[0]
        p.why.append(f"`{one_per_date[0]}` names one series: each of its values has one row per date")
    else:
        p.missing.append("series_column")  # no column names the series (or one series only): the person says which
    step = stamp.dropna().sort_values().diff().dropna()
    days = float(step.dt.total_seconds().median() / 86400) if len(step) else None
    season = 7 if days is not None and 0.5 <= days <= 1.5 else 12 if days is not None and 27 <= days <= 32 else 52 if days is not None and 6 <= days <= 8 else None
    if season:
        p.fields["season"] = season
        p.why.append(f"the dates are {'daily' if season == 7 else 'monthly' if season == 12 else 'weekly'}: a season of {season}")
    p.fields["horizon"] = season or 7
    p.assumed["horizon"] = (f"assumed: {p.fields['horizon']} steps ahead; how far ahead the forecast is needed is the person's to say, "
                            "on the Solution page")
    return p


DURATION_WORDS = {"days", "duration", "tenure", "months", "weeks", "lifetime", "survival", "time"}
EVENT_WORDS = {"cancelled", "canceled", "churned", "died", "failed", "event", "ended", "left", "closed", "defaulted", "status"}
START_WORDS = {"signup", "start", "started", "joined", "opened", "begin", "created", "since"}


def _time_to_event(frame: pd.DataFrame, target: Optional[str]) -> FieldProposal:
    p = FieldProposal("time_to_event")
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and not _binary(frame, c)]
    durations = ([target] if target in numeric and _tokens(target) & DURATION_WORDS else []) + [c for c in _named(frame, DURATION_WORDS) if c in numeric and c != target]
    if durations:
        p.fields["duration_column"] = durations[0]
        p.why.append(f"`{durations[0]}` counts the time until the event (or until it was last seen)")
    else:
        p.missing.append("duration_column")
    events = [c for c in _named(frame, EVENT_WORDS) if _binary(frame, c)]
    if events:
        p.fields["event_column"] = events[0]
        p.why.append(f"`{events[0]}` says whether the event happened (the rest are still open: censored)")
    else:
        p.missing.append("event_column")
    starts = [c for c in _dates(frame) if _tokens(c) & START_WORDS] or _dates(frame)
    if starts:
        p.fields["start_column"] = starts[0]
        p.why.append(f"`{starts[0]}` is when each one started")
    else:
        p.missing.append("start_column")
    return p


def _ranking(frame: pd.DataFrame, target: Optional[str]) -> FieldProposal:
    p = FieldProposal("ranking")
    for name, words in (("user_column", {"user", "customer", "visitor", "member", "account", "buyer"}),
                        ("item_column", {"item", "product", "sku", "article", "movie", "song", "listing"})):
        found = [c for c in _named(frame, words) if _repeating_category(frame, c) or pd.api.types.is_numeric_dtype(frame[c])]
        if found:
            p.fields[name] = found[0]
            p.why.append(f"`{found[0]}` names the {name.split('_')[0]}s")
        else:
            p.missing.append(name)
    dates = _dates(frame)
    if dates:
        p.fields["time_column"] = dates[0]
        p.why.append(f"`{dates[0]}` orders the interactions")
    else:
        p.missing.append("time_column")
    events = [c for c in _named(frame, {"event", "action", "type", "interaction", "kind"}) if frame[c].nunique(dropna=True) <= 20]
    if events:
        p.fields["event_column"] = events[0]
        counted = [v for v in frame[events[0]].dropna().astype(str).unique() if v.lower() in ("purchase", "buy", "order", "bought", "conversion", "convert")]
        p.why.append(f"`{events[0]}` says what each interaction was")
        if counted:
            p.fields["relevant_event"] = counted[0]
            p.assumed["relevant_event"] = f"assumed: a `{counted[0]}` is the interaction that counts; the person confirms it"
        else:
            p.missing.append("relevant_event")
    else:
        p.missing += ["event_column", "relevant_event"]
    if dates:
        stamp = pd.to_datetime(frame[dates[0]].astype(str), errors="coerce", format="mixed").dropna()
        if len(stamp):
            moment = stamp.sort_values().iloc[int(0.8 * (len(stamp) - 1))].date().isoformat()
            p.fields["moment"] = moment
            p.assumed["moment"] = f"assumed: the ranking is made on {moment} (80% of the log before it); the person can move it"
    if "moment" not in p.fields:
        p.missing.append("moment")
    return p


TREATMENT_WORDS = {"offer", "treated", "treatment", "campaign", "coupon", "voucher", "exposed", "promo", "promotion", "discount", "contacted", "got"}


def _uplift(frame: pd.DataFrame, target: Optional[str]) -> FieldProposal:
    p = FieldProposal("uplift")
    treatments = [c for c in _named(frame, TREATMENT_WORDS, {target} if target else set()) if _binary(frame, c)]
    if treatments:
        p.fields["treatment_column"] = treatments[0]
        p.why.append(f"`{treatments[0]}` says who got the action")
    else:
        p.missing.append("treatment_column")
    p.missing.append("randomised")  # who got it at random or by choice decides what can be measured: only the person knows
    return p


def _entity_tables(frame: pd.DataFrame, target: Optional[str]) -> FieldProposal:
    return FieldProposal("entity_tables", missing=["target_table", "horizon_days", "cutoffs"],
                         why=["its outcome is counted in another table of events (orders, visits): the tables are attached first"])


def _streams(frame: pd.DataFrame, target: Optional[str]) -> FieldProposal:
    p = FieldProposal("streams")
    dates = _dates(frame)
    units = [c for c in _named(frame, {"machine", "device", "sensor", "unit", "pump", "turbine", "asset", "vehicle", "engine"}) if c not in dates]
    if units:
        p.fields["unit_column"] = units[0]
        p.why.append(f"`{units[0]}` names the units whose readings come in")
    else:
        p.missing.append("unit_column")
    if dates:
        p.fields["time_column"] = dates[0]
    else:
        p.missing.append("time_column")
    sensors = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in units and not _binary(frame, c)]
    if sensors:
        p.fields["sensor_columns"] = sensors
        p.why.append(f"{len(sensors)} readings: {', '.join(sensors[:5])}")
    else:
        p.missing.append("sensor_columns")
    p.missing.append("incident_time_column")  # the incidents come in their own table: attached first
    return p


PROPOSERS: dict[str, Callable[[pd.DataFrame, Optional[str]], FieldProposal]] = {
    "segmentation": _segmentation, "embedded": _embedded, "forecasting": _forecasting, "time_to_event": _time_to_event,
    "ranking": _ranking, "uplift": _uplift, "entity_tables": _entity_tables, "streams": _streams,
}


# a card's family that means one kind with fields of its own (classification, regression, rare events and anomalies run as
# their tasks and packs; text_or_images is the embedded kind only when the table names its sources)
FAMILY_KINDS = {"forecasting": "forecasting", "time_to_event": "time_to_event", "ranking": "ranking", "segmentation": "segmentation",
                "uplift": "uplift", "entity_tables": "entity_tables", "streams": "streams", "text_or_images": "embedded"}
NO_OUTCOME = frozenset({"segmentation", "ranking", "streams", "entity_tables"})  # kinds that read no outcome column


def for_family(family: Optional[str], frame: pd.DataFrame, target: Optional[str] = None) -> Optional[FieldProposal]:
    """The kind a card's family means, with its fields from the table; None when the family is a plain task's."""
    kind = FAMILY_KINDS.get(str(family or ""))
    return propose_fields(kind, frame, target) if kind else None


def propose_fields(kind: str, frame: pd.DataFrame, target: Optional[str] = None) -> Optional[FieldProposal]:
    """The kind's fields the table shows, its assumptions and what only the person can give; None for a kind with no
    fields of its own (the plain tasks)."""
    fn = PROPOSERS.get(kind)
    if fn is None or frame is None:
        return None
    try:
        return fn(frame, target)
    except Exception as exc:  # noqa: BLE001 - a table that cannot be read leaves the kind unproposed, with why
        return FieldProposal(kind, missing=["fields"], why=[f"the table could not be read for its fields: {type(exc).__name__}"])


__all__ = ["FieldProposal", "PROPOSERS", "FAMILY_KINDS", "NO_OUTCOME", "for_family", "propose_fields"]
