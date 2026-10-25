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
from typing import Any, Callable, Iterable, Optional

import pandas as pd


@dataclass
class FieldProposal:
    kind: str
    fields: dict[str, Any] = field(default_factory=dict)
    time_column: Optional[str] = None
    why: list[str] = field(default_factory=list)
    assumed: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    declined: dict[str, str] = field(default_factory=dict)  # S9.3: a missing field the person did not know, with what follows

    @property
    def complete(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "kind_fields": self.fields, "time_column": self.time_column, "why": self.why,
                "assumed": self.assumed, "missing": self.missing, **({"declined": self.declined} if self.declined else {})}


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


OUTCOME_WORDS = {"renewed", "churned", "churn", "bought", "converted", "conversion", "cancelled", "canceled", "cancel", "purchased",
                 "responded", "response", "clicked", "left", "outcome", "label", "target", "retained", "returned", "exited",
                 "attrition", "default", "defaulted", "fraud", "won", "approved", "subscribed"}  # specific words only: one outcome
# column is left out of the segments, and a broad word ("is" in is_male) would take its place from the real outcome


def _segmentation(frame: pd.DataFrame, target: Optional[str]) -> FieldProposal:
    p = FieldProposal("segmentation", why=["the segments are found from the columns that describe each row"])
    outcomes = ([target] if target in frame.columns else []) + [c for c in _named(frame, OUTCOME_WORDS) if _binary(frame, c) and c != target]
    if outcomes:  # an outcome in the table describes the segments afterwards and never forms them (it would be a leak; the review of S9:
        # Telco's `Churn` was missed by the words and named as the target, and would have formed the segments)
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
    # a series column only where dates repeat (several series): on one series every date is unique, and any column with few
    # values (a weekday) would pass as "one row per date" (the review of S9)
    one_per_date = [c for c in candidates if not pd.DataFrame({"s": frame[c], "d": stamp}).duplicated().any()] if stamp.dropna().duplicated().any() else []
    one_per_date.sort(key=lambda c: (not (_tokens(c) & {"series"}), frame[c].nunique()))
    if one_per_date:
        p.fields["series_column"] = one_per_date[0]
        p.why.append(f"`{one_per_date[0]}` names one series: each of its values has one row per date")
    else:
        p.missing.append("series_column")  # no column names the series (or one series only): the person says which
    step = stamp.dropna().drop_duplicates().sort_values().diff().dropna()  # the distinct dates: repeated ones (many series) read 0
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
    if durations and target in frame.columns and target != p.fields.get("duration_column"):  # the kind's target is its duration (the review of S9:
        # a proposal on the event `churned` read complete and no save could pass): suggested, with the change said
        p.missing.append("duration_column")
        p.fields.pop("duration_column")  # read by its name only, which may be another count (months billed): never named as the outcome
        p.declined["duration_column"] = ("its outcome is the duration, how long each one lasted: choose that column as the outcome on "
                                         "the Solution page" + (f", with `{target}` as its event" if _binary(frame, target) else ""))
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
    p = FieldProposal("ranking", why=["the ranking machine reads the items in their own table, attached by the name `items`"])
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


def for_family(family: Optional[str], frame: pd.DataFrame, target: Optional[str] = None, said: Optional[dict[str, Optional[str]]] = None,
               tables: Optional[Iterable[str]] = None) -> Optional[FieldProposal]:
    """The kind a card's family means, with its fields from the table; None when the family is a plain task's."""
    kind = FAMILY_KINDS.get(str(family or ""))
    return propose_fields(kind, frame, target, said, tables) if kind else None


def propose_fields(kind: str, frame: pd.DataFrame, target: Optional[str] = None, said: Optional[dict[str, Optional[str]]] = None,
                   tables: Optional[Iterable[str]] = None) -> Optional[FieldProposal]:
    """The kind's fields the table shows, its assumptions and what only the person can give; None for a kind with no
    fields of its own (the plain tasks). ``said`` (S9.3): the person's answers to the kind's questions, in their words
    (None: they did not know); ``tables``: the related tables attached, by name, when known."""
    fn = PROPOSERS.get(kind)
    if fn is None or frame is None:
        return None
    try:
        p = fn(frame, target)
    except Exception as exc:  # noqa: BLE001 - a table that cannot be read leaves the kind unproposed, with why
        return FieldProposal(kind, missing=["fields"], why=[f"the table could not be read for its fields: {type(exc).__name__}"])
    if kind == "ranking" and tables is not None and "items" not in set(tables):
        p.missing.append("items_table")
    return _with_said(p, frame, said or {})


# ------------------------------------------------------------------------------------------------ S9.3: what the person says


@dataclass(frozen=True)
class KindQuestion:
    """A kind's field only the person can give, asked in the kind's words. ``columns``: the answer names a column (the
    table's columns that could be it are offered); ``assume``: the value taken when the person does not know (said as
    an assumption); without one the field stays missing and ``unknown`` says what runs instead."""

    kind: str
    field: str
    text: str
    why: str
    options: tuple[str, ...] = ()
    columns: bool = False
    unknown: str = ""
    assume: Any = None

    @property
    def key(self) -> str:  # the Home question's field: one per kind field
        return f"kind_{self.field}"


def _column_q(kind: str, name: str, text: str, why: str, unknown: str) -> KindQuestion:
    return KindQuestion(kind, name, text, why, columns=True, unknown=unknown)


PLAIN = "until it is said, the plain task runs on the outcome, and the Solution page keeps the kind suggested"
_QUESTIONS = (
    KindQuestion("uplift", "randomised",
                 "Was the action (the offer, the call) given at random, as an experiment with a group left out, or did someone choose who got it?",
                 "Only a random choice lets the report say what the action changes; otherwise it says who responded with it.",
                 ("At random: an experiment", "Someone chose who got it", "I don't know"),
                 unknown="assumed: not at random, the cautious reading: the report says who responded with the action, never what it caused",
                 assume=False),
    _column_q("uplift", "treatment_column", "Which column says who got the action (the offer, the call)?",
              "The effect is read by comparing those who got it with those who did not.", PLAIN),
    _column_q("forecasting", "series_column", "Which column names one series (a store, a product, a region)?",
              "Each series is forecast along its own dates; rows of two series are never mixed.", PLAIN),
    _column_q("forecasting", "time_column", "Which column holds the date each row is for?", "The forecast moves along the dates.", PLAIN),
    _column_q("time_to_event", "duration_column", "Which column says how long each one lasted, or has lasted so far?",
              "The time until the event is what is learned.", PLAIN),
    _column_q("time_to_event", "event_column", "Which column says whether the event (the cancellation, the failure) has happened yet?",
              "The ones still open are counted as open, never as no event.", PLAIN),
    _column_q("time_to_event", "start_column", "Which column says when each one started?", "The start orders the rows for the split.", PLAIN),
    _column_q("ranking", "user_column", "Which column names the user (the customer, the visitor)?", "Each user gets their own list.", PLAIN),
    _column_q("ranking", "item_column", "Which column names the item (the product, the article)?", "The items are what is ranked.", PLAIN),
    _column_q("ranking", "event_column", "Which column says what each interaction was (a view, a purchase)?",
              "One kind of interaction is the one that counts.", PLAIN),
    KindQuestion("ranking", "relevant_event", "Which interaction counts as a success: a purchase, a click, a booking?",
                 "The ranking is judged on the interactions that count.", unknown=PLAIN),
    _column_q("embedded", "text_column", "Which column holds the texts each row is about?", "The model reads the texts.", PLAIN),
    _column_q("embedded", "source_column", "Which column says where each text comes from (an author, a customer, a channel)?",
              "Texts of one source stay on one side of the split, or the score would be too good.", PLAIN),
)
QUESTIONS: dict[tuple[str, str], KindQuestion] = {(q.kind, q.field): q for q in _QUESTIONS}
NEEDS_TABLE = {  # a field a sentence cannot give: the table it comes from is attached on the project's Data page
    ("ranking", "items_table"): "the items in their own table, attached by the name `items` on the project's Data page",
    ("streams", "incident_time_column"): "the incidents in their own table, attached on the project's Data page",
    ("entity_tables", "target_table"): "the table of events that decides the outcome, attached on the project's Data page",
}
RANDOM = re.compile(r"\b(random\w*|experiment\w*|a/?b\b|control group|left out|holdout|hold-out|lottery|coin)", re.I)
CHOSEN = re.compile(r"\b(chose|chosen|choose|picked|selected|targeted|decided|by hand|hand-?picked)\b", re.I)
NEGATIONS = frozenset({"no", "not", "never", "without", "nobody", "none", "nothing", "neither", "nor", "non", "nope"})


def _negated(text: str, at: int) -> bool:
    """Whether a negation stands in the three words before position ``at``, in the same clause ("we did not run an A/B
    test", "no control group", "nobody chose", "non-random"; not "no randomisation, we targeted": two clauses)."""
    clause = re.split(r"[,.;:!?]|\bbut\b|\band\b", text[:at].lower())[-1]  # a negation reaches no further than its clause
    words = re.findall(r"[a-z']+", clause)[-3:]
    return any(w in NEGATIONS or w.endswith("n't") for w in words) or clause.endswith("non-")


def _randomised(text: str) -> Optional[bool]:
    """Random, not random, or unread: each word of randomness or of choosing, negated or not, is evidence; both kinds
    of evidence at once (\"random, but the manager picked some\") leave the answer unread, never guessed."""
    random_ = chosen = False
    for m in RANDOM.finditer(text):
        if _negated(text, m.start()):
            chosen = True
        else:
            random_ = True
    for m in CHOSEN.finditer(text):
        if _negated(text, m.start()):
            random_ = True
        else:
            chosen = True
    return None if random_ == chosen else random_


def read_said(q: KindQuestion, words: str, frame: pd.DataFrame, fields: dict[str, Any]) -> tuple[bool, Any]:
    """What the person's words give for the field: (read, value). A column must be named exactly once; an interaction
    must be one of the event column's values; "random" or "chosen" answers the randomisation."""
    from dclab_core.conversation import named_columns

    text = str(words or "")
    if q.field == "randomised":  # "random, not chosen" is random; "we did not run an A/B test" is not (the review of S9)
        value = _randomised(text)
        return (False, None) if value is None else (True, value)
    if q.columns:
        named = named_columns(text, [str(c) for c in frame.columns])
        return (True, named[0]) if len(named) == 1 else (False, None)
    if q.field == "relevant_event" and fields.get("event_column") in frame.columns:
        values = [str(v) for v in frame[fields["event_column"]].dropna().astype(str).unique()[:50]]
        hits = [v for v in values if re.search(rf"(?<![\w]){re.escape(v)}(?![\w])", text, re.I)]
        return (True, hits[0]) if len(hits) == 1 else (False, None)
    return False, None


def _with_said(p: FieldProposal, frame: pd.DataFrame, said: dict[str, Optional[str]]) -> FieldProposal:
    """A missing field the person answered: read from their words (with why), or, when they did not know, assumed and
    said, or left missing with what runs instead (never a silent default)."""
    for name in list(p.missing):
        q = QUESTIONS.get((p.kind, name))
        if q is None or name not in said:
            continue
        if said[name] is None:  # "I don't know"
            if q.assume is not None:
                p.fields[name] = q.assume
                p.assumed[name] = q.unknown
                p.missing.remove(name)
            else:
                p.declined[name] = q.unknown
            continue
        read, value = read_said(q, said[name], frame, p.fields)
        if read:
            p.fields[name] = value
            p.missing.remove(name)
            p.why.append(f"{name.replace('_', ' ')}: as the person said ({value if name != 'randomised' else ('at random' if value else 'not at random')})")
    if p.kind == "uplift" and p.fields.get("randomised") is False and "randomised" not in p.assumed:
        p.why.append("not at random, as the person said: the report says who responded with the action, never what it caused")
    return p


def question_for(p: Optional[FieldProposal], asked: Iterable[str] = ()) -> Optional[KindQuestion]:
    """The first missing field of the proposal the person can answer in words, not asked before (S9.3)."""
    if p is None:
        return None
    asked = set(asked)
    for name in p.missing:
        q = QUESTIONS.get((p.kind, name))
        if q is not None and q.key not in asked and name not in p.declined:
            return q
    return None


def options_for(q: KindQuestion, frame: pd.DataFrame, p: FieldProposal) -> list[str]:
    """A column question's options: the columns that could be it, the kind's own words first; names only, never values."""
    if not q.columns:
        return list(q.options)
    taken = {v for v in p.fields.values() if isinstance(v, str)}
    words = {"series_column": SERIES_WORDS, "duration_column": DURATION_WORDS, "event_column": EVENT_WORDS | {"action", "type", "event"},
             "start_column": START_WORDS, "treatment_column": TREATMENT_WORDS, "text_column": TEXT_WORDS, "source_column": SOURCE_WORDS,
             "user_column": {"user", "customer", "visitor", "member", "account"}, "item_column": {"item", "product", "sku", "article"},
             "time_column": {"date", "time", "day", "timestamp"}}.get(q.field, set())
    cols = [str(c) for c in frame.columns if str(c) not in taken]
    return (sorted(cols, key=lambda c: (not (_tokens(c) & words), cols.index(c)))[:3] + ["I don't know"])


def still_needed(p: Optional[FieldProposal]) -> list[str]:
    """What the kind still needs, in words for the person: each declined field with what follows, each other table."""
    if p is None:
        return []
    out = [f"{name.replace('_', ' ')}: {why}" for name, why in p.declined.items()]
    out += [f"{NEEDS_TABLE[(p.kind, name)]}" for name in p.missing if (p.kind, name) in NEEDS_TABLE]
    return out


def said_of(understanding: dict[str, Any]) -> dict[str, Optional[str]]:
    """The person's answers to the kind's questions in a draft's understanding: their words, or None for "I don't know"."""
    out: dict[str, Optional[str]] = {k[len("kind_"):]: str(v) for k, v in (understanding or {}).items()
                                     if k.startswith("kind_") and isinstance(v, str)}
    for k in (understanding or {}).get("not_known") or []:
        if str(k).startswith("kind_"):
            out[str(k)[len("kind_"):]] = None
    return out


__all__ = ["FieldProposal", "PROPOSERS", "FAMILY_KINDS", "NO_OUTCOME", "QUESTIONS", "KindQuestion", "for_family", "propose_fields",
           "question_for", "options_for", "read_said", "said_of", "still_needed"]
