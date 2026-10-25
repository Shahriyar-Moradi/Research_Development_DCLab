"""A student of one point answering another whose truth follows from its answer (step 5 of the 100-students plan, the
owner, 2026-10-25).

On planted tables today's rules are weak for "do rows repeat an entity?" (``modifier.groups``, right 61%), "do the rows'
times order the split?" (``modifier.time_order``, 72%) and "is this cleaning step kept?" (``data.cleaning_keep``, 55%).
Two trained students already hold that skill: the split student (``split.strategy``: 77% against the rule's 50% on dev)
and the cleaning-step student (``data.cleaning_step``: 92% against 73%). A new student per point would learn the same
thing from the same tables; instead the existing one answers, and its answer is mapped:

- ``modifier.groups``: the split keeps each entity on one side (``group`` or ``time_group``);
- ``modifier.time_order``: the split holds the latest period out (``time`` or ``time_group``);
- ``data.cleaning_keep``: the step applied is the one the column needs (numbers after missing markers too: a column of
  numbers written with "N/A" needs both).

The record's state carries the base student's fields (built one way: ``ledger.states``), so the base reads exactly what
it was trained on. A mapped answerer is saved like any student (a card and a fitted file whose bytes ``fitted.json``
hashes) and whispers in shadow beside the rule; it acts only after a sealed exam of its own point and a person's switch.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

SPLIT_FIELDS = ("rows", "columns", "has_time", "has_groups", "date_columns", "id_columns", "repeat_share")  # ledger.states.split_state


def _groups(value: Any, state: Mapping[str, Any]) -> Optional[bool]:
    return None if value is None else str(value) in ("group", "time_group")


def _time_order(value: Any, state: Mapping[str, Any]) -> Optional[bool]:
    return None if value is None else str(value) in ("time", "time_group")


def _keep(value: Any, state: Mapping[str, Any]) -> Optional[bool]:
    step = state.get("cleaning")
    if value is None or not step:
        return None
    return str(value) == str(step) or (str(value) == "missing_markers" and str(step) == "numbers")


COLUMN_STEPS = ("missing_markers", "numbers", "dates", "booleans", "trim")  # the cleaning-step student's own steps
COLUMN_BANDS = ("dtype", "uniqueness", "value_pattern")


def ready(point: str, state: Mapping[str, Any]) -> bool:
    """Whether a record holds what the base student reads: every split field (an older record or another table's lacks
    them), or a column step with the column's bands (a table-wide step, duplicates or names, has no column to judge)."""
    if point == "data.cleaning_keep":
        return state.get("cleaning") in COLUMN_STEPS and all(state.get(k) is not None for k in COLUMN_BANDS)
    return all(state.get(k) is not None for k in SPLIT_FIELDS)


def _split_view(state: Mapping[str, Any]) -> dict[str, Any]:
    return {k: state[k] for k in SPLIT_FIELDS if k in state}


def _cleaning_view(state: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if k != "cleaning"}


# each mapped point: the base point, the base's own options, how its state is read, how its answer becomes this point's
MAPS: dict[str, tuple[str, tuple[str, ...], Any, Any]] = {
    "modifier.groups": ("split.strategy", ("stratified", "time", "group", "time_group"), _split_view, _groups),
    "modifier.time_order": ("split.strategy", ("stratified", "time", "group", "time_group"), _split_view, _time_order),
    "data.cleaning_keep": ("data.cleaning_step", ("missing_markers", "numbers", "dates", "booleans", "trim", "none"), _cleaning_view, _keep),
}


class MappedAnswerer:
    """``base`` (a fitted ``ClassifierAnswerer`` of ``MAPS[point][0]``) answering ``point``. Picklable: the map is looked up
    by the point's name, never stored as a function."""

    by = "classifier"

    def __init__(self, base: Any, point: str, name: Optional[str] = None):
        if point not in MAPS:
            raise ValueError(f"no map for {point!r}: one of {', '.join(MAPS)}")
        if getattr(base, "point", None) != MAPS[point][0]:
            raise ValueError(f"{point} is answered by a {MAPS[point][0]} student, not {getattr(base, 'point', None)!r}")
        self.base, self.point, self.points = base, point, frozenset({point})
        self.name = name or f"{point}.from_{MAPS[point][0]}"
        self.version = f"{base.version}+map1"
        self.explain = False

    def answer(self, point: str, state: Mapping[str, Any], options: list[Any]) -> Optional[Any]:
        from dclab_core.ledger import Answer

        if point != self.point:
            return None
        base_point, base_options, view, value_of = MAPS[point]
        if not ready(point, state):
            return None  # the review of step 5: without its own fields the base student reads them as zeros: it abstains
        got = self.base.answer(base_point, view(state), list(base_options))
        value = value_of(getattr(got, "value", None), state)
        if got is None or value is None:
            return None
        return Answer(by="classifier", answerer=f"{self.name}@{self.version}", value=value, raw=got.raw, confidence=got.confidence,
                      reason=f"{base_point}: {got.value}")

    def answer_many(self, point: str, states: list[Mapping[str, Any]], options: list[list[Any]]) -> list[Optional[Any]]:
        return [self.answer(point, s, o) for s, o in zip(states, options)]


__all__ = ["MAPS", "MappedAnswerer", "SPLIT_FIELDS", "ready"]
