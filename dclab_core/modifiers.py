"""Modifiers (package 30.11): conditions on the data that change how a kind is handled, each a recipe part with its rule.

The owner's task recipes name them; this package builds the first three:

* **rare positives:** the minority class is under 5% of the training rows (one cut-off for the whole studio, replacing the
  1%, 5% and 10% that grew in different places). The primary metric becomes average precision; accuracy is shown only
  with a warning.
* **groups:** an identifier repeats across rows (at least two rows per value at the median, at least twenty values). The
  split keeps each group on one side, on the coarsest such key (a finer one nests inside it); with a time column, a key
  whose values recur over time is not grouped (the forward split already keeps the future out, and grouping too would
  leave nothing to test on): the person is asked instead.
* **time order:** the rows have a time column. The split is forward in time (DCLAB-R02); features are built only from
  data before each row's cutoff. The owner's rule adds "the target rate changes over time": a moving target makes the
  evidence stronger and is said, but a time column alone already applies it, because a random split of ordered rows reads
  the future whether or not the rate moves.

30.12 adds two, to finish the fraud case:

* **noisy labels:** a label-source column holds a noisy source (a rule, a cluster of look-alikes) beside at least twenty
  confirmed rows (a person's review, a random audit). A noisy source's rows are neither trained nor judged on (weighing
  them by their agreement instead is left for later); the rule that built a noisy source's labels is flagged, since a
  model would learn it back; each rule's agreement with the confirmed labels is reported. Under noisy labels the
  rare-positives test reads the confirmed rows' rate, the audit's when there is one (a rule's labels make the positives
  look common).
* **review capacity:** a team checks K cases a period. Precision at K is the measure the holdout's queue is judged by
  (selection inside training keeps the primary metric: selection by precision at K within folds is left), recall is
  estimated only from the random audit, and the operating threshold is the K-th highest score of an average period. K is
  asked on every screening (binary) problem, never guessed (storing it from Home and the wizard is left).

``detect(facts)`` reads aggregate facts about the training rows (never a row, never a value) and says, for each modifier,
whether it applies, why, and how strong the evidence is. Rules decide; a model may only propose a modifier to a person.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from dclab_core.contracts import Contract
from dclab_core.recipes import Detection, RecipePart

RARE_BELOW = 0.05       # the minority share under which positives are rare
ROWS_PER_KEY = 2.0      # rows per value at the median from which an identifier groups rows
DRIFT_RATIO = 1.5       # the positive rate's change between the first and last third (or its inverse) that counts as drift
CLASS_SHIFT = 0.1       # the class mix's total variation distance between the first and last third
LEVEL_SHIFT = 0.3       # the target's level shift between the first and last third, in standard deviations
MIN_GROUPS = 20         # a key with fewer distinct values cannot hold a group split's holdout
RECURS = 0.85           # the share of the latest rows whose key was seen earlier from which a forward split cannot also group them
MIN_CONFIRMED = 20      # confirmed rows below which noisy labels cannot be judged on confirmed ones alone
CONFIRMED = ("expert", "expert_audit", "audit", "review", "reviewed", "confirmed", "person")  # a person looked: the labels judged on
AUDIT = ("expert_audit", "audit")  # drawn at random: the only rows recall can be estimated from
NOISY = ("rule", "cluster", "heuristic", "weak", "proxy", "model")  # labels a rule or a likeness wrote
PERIODS = ("day", "week", "month")
VERSION = "1"

PARTS: dict[str, RecipePart] = {
    "rare_positives": RecipePart(
        name="rare_positives", part_type="modifier", version=VERSION, level="experimental",
        detection=[Detection(signal="minority_rate", rule=f"< {RARE_BELOW} of the training rows")],
        primary_metric="average_precision", guard_metrics=["roc_auc"],
        report_items=["accuracy only with a warning: a model that flags nothing scores high"],
        leakage_checks=["class shares read on the training rows only"]),
    "groups": RecipePart(
        name="groups", part_type="modifier", version=VERSION, level="experimental",
        detection=[Detection(signal="rows_per_key", rule=f">= {ROWS_PER_KEY:g} rows per value at the median, for an identifier-like column")],
        required_inputs=["group_column"], split="group", leakage_checks=["no group on both sides of the split"],
        artifact_needs=["the group key's column name"]),
    "time_order": RecipePart(
        name="time_order", part_type="modifier", version=VERSION, level="experimental",
        detection=[Detection(signal="time_column", rule="a column orders the rows in time (DCLAB-R02); a moving target strengthens it: "
                                                         f"the positive rate x{DRIFT_RATIO:g} or /{DRIFT_RATIO:g}, the class mix over {CLASS_SHIFT:g}, "
                                                         f"the level over {LEVEL_SHIFT:g} sd")],
        required_inputs=["time_column"], split="time", leakage_checks=["features only from data before each row's cutoff"],
        artifact_needs=["the cutoff rule for features", "the training date range"], report_items=["the holdout is the latest period"]),
    "noisy_labels": RecipePart(
        name="noisy_labels", part_type="modifier", version=VERSION, level="experimental",
        detection=[Detection(signal="label_source", rule=f"a label-source column with a noisy source ({', '.join(NOISY)}) and at least "
                                                         f"{MIN_CONFIRMED} confirmed rows")],
        required_inputs=["label_source_column"],
        leakage_checks=["the rule that built a noisy source's labels flagged: a model would learn it back",
                        "cross-validation and the holdout judged on confirmed labels only"],
        report_items=["each noisy source's rule and its agreement with the confirmed labels", "trained and evaluated on confirmed labels only"]),
    "review_capacity": RecipePart(
        name="review_capacity", part_type="modifier", version=VERSION, level="experimental",
        detection=[Detection(signal="the person's words", rule="K cases a period, stated by the person", asks_user=True)],
        required_inputs=["capacity_k", "capacity_period"], primary_metric="precision_at_k", guard_metrics=["audit_recall"],  # the holdout queue's measure
        threshold_rule="the K-th highest score of an average period",
        artifact_needs=["the threshold: the K-th score of an average period"],
        report_items=["precision at K on confirmed labels", "recall estimated only from the random audit"]),
}


class KeyFact(Contract):
    column: str = Field(max_length=200)
    rows_per_key: float = Field(ge=0, description="Rows per value at the median, on the training rows.")
    groups: int = Field(default=0, ge=0, description="How many distinct values it holds.")
    recurring: Optional[float] = Field(default=None, ge=0, le=1, description="The share of the latest third's rows whose value "
                                       "appeared earlier (None without a time column): a key that recurs cannot also split forward.")


class Facts(Contract):
    """What the detection reads: aggregates of the training rows, never a row."""

    task: str = Field(max_length=40)
    minority_rate: Optional[float] = Field(default=None, ge=0, le=1, description="The smallest class's share (binary, multiclass).")
    positive_rate: Optional[float] = Field(default=None, ge=0, le=1, description="The positive class's share (binary).")
    keys: list[KeyFact] = Field(default_factory=list, max_length=40, description="Identifier-like columns that could group rows.")
    time_column: Optional[str] = Field(default=None, max_length=200)
    rate_change: Optional[float] = Field(default=None, ge=0, description="Last third's positive rate over the first third's.")
    class_shift: Optional[float] = Field(default=None, ge=0, le=1)
    level_shift: Optional[float] = Field(default=None, description="In standard deviations.")
    # 30.12: where the labels came from, and how many cases a team can check
    label_sources: dict[str, int] = Field(default_factory=dict, description="Rows per label source among the training rows with an outcome.")
    confirmed_rate: Optional[float] = Field(default=None, ge=0, le=1, description="The positive rate on the confirmed rows.")
    audit_rate: Optional[float] = Field(default=None, ge=0, le=1, description="The positive rate on the rows drawn at random for review.")
    capacity_k: Optional[int] = Field(default=None, ge=1, description="The cases a team checks a period, as the person stated it.")
    capacity_period: Optional[str] = Field(default=None, max_length=20)


class Detected(Contract):
    name: str = Field(max_length=60)
    applies: bool
    reason: str = Field(max_length=600)
    strength: float = Field(ge=0, le=1, description="How far past its threshold the evidence is: 0 at none, 1 far past.")
    value: Optional[str] = Field(default=None, max_length=200, description="The column it applies with (a group key, the time column).")
    asks_user: bool = False


def _drift(f: Facts) -> Optional[str]:
    if f.rate_change is not None and (f.rate_change > DRIFT_RATIO or (0 < f.rate_change < 1 / DRIFT_RATIO)):
        return f"the positive rate changes {f.rate_change:.2g} times from the first to the last third"
    if f.class_shift is not None and f.class_shift > CLASS_SHIFT:
        return f"the class mix moves by {f.class_shift:.2g} from the first to the last third"
    if f.level_shift is not None and abs(f.level_shift) > LEVEL_SHIFT:
        return f"the target's level moves {f.level_shift:+.2g} sd from the first to the last third"
    return None


def _noisy(facts: Facts) -> Detected:
    sources = {str(k).lower(): int(v) for k, v in (facts.label_sources or {}).items() if v}
    noisy = sorted(s for s in sources if s in NOISY)
    confirmed = sum(v for s, v in sources.items() if s in CONFIRMED)
    if noisy and confirmed >= MIN_CONFIRMED:
        share = sum(sources[s] for s in noisy) / max(1, sum(sources.values()))
        return Detected(name="noisy_labels", applies=True, strength=round(min(1.0, share * 2), 3),
                        reason=(f"{share:.0%} of the labels come from {', '.join(noisy)}, beside {confirmed} confirmed rows: the model is "
                                "judged on the confirmed rows only, and the rule behind each noisy source is flagged")[:600])
    if noisy:
        return Detected(name="noisy_labels", applies=False, strength=0.0, asks_user=True,
                        reason=f"the labels come from {', '.join(noisy)} with {confirmed} confirmed rows (fewer than {MIN_CONFIRMED}): which can be trusted?")
    return Detected(name="noisy_labels", applies=False, strength=0.0, reason="no noisy label source among the training rows")


def _capacity(facts: Facts) -> Detected:
    if facts.task != "binary":
        return Detected(name="review_capacity", applies=False, strength=0.0, reason="not a screening task: no queue of cases to check")
    if facts.capacity_k:
        period = facts.capacity_period or "period"
        return Detected(name="review_capacity", applies=True, strength=1.0, asks_user=True, value=str(facts.capacity_k),
                        reason=(f"the team checks {facts.capacity_k} cases a {period}: precision at {facts.capacity_k} decides, the threshold is "
                                f"the {facts.capacity_k}-th score of an average {period}, and recall is estimated only from the random audit"))
    return Detected(name="review_capacity", applies=False, strength=0.0, asks_user=True,
                    reason="how many cases can the team check a day, a week or a month? (always asked on a screening problem)")


def detect(facts: Facts) -> list[Detected]:
    """Each modifier with whether it applies, why, how strong the evidence is, and the column it applies with."""
    out = []
    noisy = _noisy(facts)
    rate = facts.positive_rate if facts.task == "binary" and facts.positive_rate is not None else facts.minority_rate
    honest = facts.audit_rate if facts.audit_rate is not None else facts.confirmed_rate
    where = "of the training rows"
    if noisy.applies and facts.task == "binary" and honest is not None:  # a rule's labels make the positives look common
        rate = honest
        where = "of the rows drawn at random for review" if facts.audit_rate is not None else "of the confirmed rows"
    if rate is not None and facts.task == "binary":
        if rate > 1 - RARE_BELOW:  # the positive class is the larger: the rare class is the other (the review of 30.11)
            out.append(Detected(name="rare_positives", applies=False, strength=0.0, asks_user=True,
                                reason=f"the positive class is {rate:.2%} of the training rows: the rare class is the other one; which is the event?"))
        else:
            rare = rate < RARE_BELOW
            out.append(Detected(name="rare_positives", applies=rare, strength=round(min(1.0, max(0.0, (RARE_BELOW - rate) / RARE_BELOW)), 3),
                                reason=f"the positive class is {rate:.2%} {where} ({'under' if rare else 'not under'} {RARE_BELOW:.0%})"))
    elif rate is not None and facts.task == "multiclass":  # macro F1 already weighs a small class: no change to the metric
        out.append(Detected(name="rare_positives", applies=False, strength=0.0,
                            reason=f"the smallest class is {rate:.2%} of the training rows; macro F1 already weighs it"))
    else:
        out.append(Detected(name="rare_positives", applies=False, strength=0.0, reason="no class shares for this task"))
    keys = sorted((k for k in facts.keys if k.rows_per_key >= ROWS_PER_KEY and k.groups >= MIN_GROUPS),
                  key=lambda k: (-k.rows_per_key, k.column))
    if keys:  # the coarsest key with enough groups: a finer one nests inside it, and grouping only the finer would leave the coarser
        best = keys[0]  # entity on both sides (the review of 30.11)
        others = f"; other keys that repeat: {', '.join(f'`{k.column}`' for k in keys[1:4])}" if len(keys) > 1 else ""
        if facts.time_column and best.recurring is not None and best.recurring >= RECURS:
            out.append(Detected(name="groups", applies=False, value=best.column, strength=0.0, asks_user=True,
                                reason=(f"`{best.column}` repeats but recurs over time ({best.recurring:.0%} of the latest rows' values were seen earlier): "
                                        "the forward split keeps the future out, and grouping too would leave almost nothing to test on; "
                                        f"group by it only if the model will face new {best.column} values{others}")[:600]))
        else:
            out.append(Detected(name="groups", applies=True, value=best.column, strength=round(min(1.0, (best.rows_per_key - 1) / 9), 3),
                                asks_user=len(keys) > 1,
                                reason=f"`{best.column}` repeats ({best.rows_per_key:g} rows per value at the median, {best.groups} values): each value on one side of the split{others}"[:600]))
    else:
        out.append(Detected(name="groups", applies=False, strength=0.0, reason="no identifier repeats across rows with enough values to group by"))
    if facts.time_column:
        moved = _drift(facts)
        out.append(Detected(name="time_order", applies=True, value=facts.time_column, strength=1.0 if moved else 0.6,
                            reason=f"`{facts.time_column}` orders the rows: the holdout is the latest period" + (f"; {moved}" if moved else "")))
    else:
        out.append(Detected(name="time_order", applies=False, strength=0.0, reason="no column orders the rows in time"))
    out += [noisy, _capacity(facts)]
    return out


# ------------------------------------------------------------------------------------------- the facts, from rows (30.13)

PROVISIONAL_SHARE = 0.75  # at most 1 - the largest holdout an engine split takes (20 % tabular, 25 % time to event)
MIN_TIMED_ROWS = 30  # fewer rows with a valid time than this: the rows cannot be ordered, a seeded share is read


def provisional_reading(index: "pd.Index", times: Any = None) -> tuple[Any, bool]:
    """75 % of the rows: what may be read before the engine locks its holdout (the rest stays unseen), and whether they
    were chosen by time. With ``times`` (the rows' time column), the earliest 75 % of the rows with a valid time and no
    row past the earliest 75 % of its distinct times, so a later holdout (the latest 20 % of a tabular split, the latest
    25 % of starts, a forecast's last dates) is never read; a row without a valid time is never read (the engine drops it,
    or orders it last). Otherwise a seeded random 75 %, which a random holdout drawn later overlaps (``rows_read_note``
    says so). Pass the rows that have a target."""
    import numpy as np
    import pandas as pd

    if times is not None:
        stamps = pd.Series(list(times), index=index)
        stamps = pd.to_numeric(stamps, errors="coerce") if pd.api.types.is_numeric_dtype(stamps) else pd.to_datetime(stamps, errors="coerce")
        valid = stamps.dropna()
        if len(valid) >= MIN_TIMED_ROWS:
            distinct = np.sort(valid.unique())
            cutoff = distinct[max(int(PROVISIONAL_SHARE * len(distinct)) - 1, 0)]
            keep = (valid.rank(method="first") <= int(PROVISIONAL_SHARE * len(valid))) & (valid <= cutoff)
            return valid.index[keep.to_numpy()], True
    keep = np.random.default_rng(42).random(len(index)) < PROVISIONAL_SHARE
    return index[keep], False



def label_source_columns(X: pd.DataFrame) -> list[str]:
    """30.12: a column whose values name where each label came from: nearly all of them sources DCLab knows (a rule, a
    cluster, an expert, an audit), whichever appear in the rows read (the earliest may hold only a rule's)."""
    import pandas as pd

    known, out = set(NOISY) | set(CONFIRMED), []
    for c in X.columns:
        if not (X[c].dtype == object or isinstance(X[c].dtype, pd.StringDtype)):
            continue
        v = X[c].dropna().astype(str).str.strip().str.lower()
        v = v[v != ""]
        if len(v) >= 20 and v.isin(known).mean() >= 0.9:
            out.append(str(c))
    return out


def facts_of(X: pd.DataFrame, y: pd.Series, task: str, time_column: Optional[str], group_candidates: list[str],
             positive_rate: Optional[float], positive_label: Any = None, label_source: Optional[str] = None,
             capacity_k: Optional[int] = None, capacity_period: Optional[str] = None) -> "Facts":
    """30.11: what the modifiers' rules read, from the rows read (the provisional training rows): the minority share, how
    many rows each identifier-like value holds, and whether the target moves between the first and the last third in time."""
    import pandas as pd

    minority = None
    if task == "binary" and positive_rate is not None:
        minority = min(positive_rate, 1 - positive_rate)
    elif task == "multiclass" and len(y):
        minority = float(y.astype(str).value_counts(normalize=True).min())
    t = pd.to_datetime(X[time_column], errors="coerce") if time_column and time_column in X else None
    ordered = t is not None and len(X) >= 60 and t.notna().mean() >= 0.8
    order = t.rank(method="first") if ordered else None
    keys = []
    for c in group_candidates:
        if c not in X or not X[c].notna().any():
            continue
        values = X[c].astype(str)
        recurring = None
        if ordered:  # the latest third's rows whose key appeared in the earlier rows: a key that recurs cannot also split forward
            last_rows = order > 2 * len(order) / 3
            recurring = float(values[last_rows].isin(set(values[~last_rows])).mean()) if last_rows.any() else None
        keys.append(KeyFact(column=str(c)[:200], rows_per_key=float(values.value_counts().median()), groups=int(values.nunique()),
                              recurring=recurring))
    change = shift = level = None
    if ordered:
        first, last = order <= len(order) / 3, order > 2 * len(order) / 3
        if task == "binary" and positive_label is not None:  # the positive class's rate, first third against last
            hits = y.astype(str) == str(positive_label)
            a, b = float(hits[first].mean()), float(hits[last].mean())
            change = (b / a) if a > 0 else (2.0 if b > 0 else None)
        elif task == "multiclass":
            pa, pb = y[first].astype(str).value_counts(normalize=True), y[last].astype(str).value_counts(normalize=True)
            shift = min(1.0, 0.5 * float(pa.subtract(pb, fill_value=0).abs().sum()))  # rounding can pass 1 (the review of 30.11)
        elif task == "regression":
            v = pd.to_numeric(y, errors="coerce")
            sd = float(v.std())
            level = float((v[last].mean() - v[first].mean()) / sd) if sd and sd > 0 else None
    sources, confirmed_rate, audit_rate = {}, None, None
    if label_source and label_source in X:  # 30.12: where the labels came from, and the positive rate where a person looked
        src = X[label_source].fillna("").astype(str).str.strip().str.lower()
        sources = {str(k): int(v) for k, v in src.value_counts().items() if k}
        if task == "binary" and positive_label is not None:
            hits = y.astype(str) == str(positive_label)
            for names, put in ((CONFIRMED, "confirmed"), (AUDIT, "audit")):
                rows = src.isin(names)
                if rows.sum() >= MIN_CONFIRMED:
                    rate = float(hits[rows].mean())
                    confirmed_rate, audit_rate = (rate, audit_rate) if put == "confirmed" else (confirmed_rate, rate)
    return Facts(task=task, minority_rate=minority, positive_rate=positive_rate if task == "binary" else None, keys=keys[:40],
                   time_column=str(time_column)[:200] if time_column else None,
                   rate_change=change, class_shift=shift, level_shift=level if level is None or abs(level) < 1e6 else None,
                   label_sources=sources, confirmed_rate=confirmed_rate, audit_rate=audit_rate, capacity_k=capacity_k,
                   capacity_period=capacity_period)


def applied(found: list[Detected]) -> list[RecipePart]:
    """The parts of the modifiers that apply."""
    return [PARTS[d.name] for d in found if d.applies and d.name in PARTS]


__all__ = ["RARE_BELOW", "ROWS_PER_KEY", "DRIFT_RATIO", "CLASS_SHIFT", "LEVEL_SHIFT", "MIN_GROUPS", "RECURS", "MIN_CONFIRMED", "CONFIRMED",
           "AUDIT", "NOISY", "PERIODS", "PARTS", "KeyFact", "Facts", "Detected", "detect", "applied", "label_source_columns", "facts_of",
           "PROVISIONAL_SHARE", "MIN_TIMED_ROWS", "provisional_reading"]
