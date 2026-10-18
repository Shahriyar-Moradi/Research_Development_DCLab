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

``detect(facts)`` reads aggregate facts about the training rows (never a row, never a value) and says, for each modifier,
whether it applies, why, and how strong the evidence is. Rules decide; a model may only propose a modifier to a person.
"""

from __future__ import annotations

from typing import Optional

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


def detect(facts: Facts) -> list[Detected]:
    """Each modifier with whether it applies, why, how strong the evidence is, and the column it applies with."""
    out = []
    rate = facts.positive_rate if facts.task == "binary" and facts.positive_rate is not None else facts.minority_rate
    if rate is not None and facts.task == "binary":
        if rate > 1 - RARE_BELOW:  # the positive class is the larger: the rare class is the other (the review of 30.11)
            out.append(Detected(name="rare_positives", applies=False, strength=0.0, asks_user=True,
                                reason=f"the positive class is {rate:.2%} of the training rows: the rare class is the other one; which is the event?"))
        else:
            rare = rate < RARE_BELOW
            out.append(Detected(name="rare_positives", applies=rare, strength=round(min(1.0, max(0.0, (RARE_BELOW - rate) / RARE_BELOW)), 3),
                                reason=f"the positive class is {rate:.2%} of the training rows ({'under' if rare else 'not under'} {RARE_BELOW:.0%})"))
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
    return out


def applied(found: list[Detected]) -> list[RecipePart]:
    """The parts of the modifiers that apply."""
    return [PARTS[d.name] for d in found if d.applies and d.name in PARTS]


__all__ = ["RARE_BELOW", "ROWS_PER_KEY", "DRIFT_RATIO", "CLASS_SHIFT", "LEVEL_SHIFT", "MIN_GROUPS", "RECURS", "PARTS", "KeyFact", "Facts",
           "Detected", "detect", "applied"]
