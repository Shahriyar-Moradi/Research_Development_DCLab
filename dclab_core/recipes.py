"""Recipes (package 30.10): a problem is one task kind plus zero or more modifiers, never a new pipeline.

Every part (a kind or a modifier) fills the same nine-field contract: how it is detected, what it needs, the split, the
metrics (one primary, guards that must not get worse), the models (the mandatory baseline, the families), its leakage
checks, the post-processing (calibration, the threshold rule), what the saved model must carry, and what the report must
show. A part has a version and a certification level (experimental, certified, product default) earned on replicas and
real datasets before the product relies on it.

``resolve(kind, modifiers, stated)`` merges the parts into a ``ResolvedRecipe``: every choice with the part that made it
and why. Precedence (the owner's recipes document, 2026-10-16): what the person stated wins; splits combine (time order and
groups give a forward split that keeps each group on one side); for the metric, the calibration and the threshold the most
specific modifier wins (review capacity, then rare positives, then the kind); a kind's time order is never dropped by a
modifier's split; lists add up in order; a conflict the rules cannot settle (two splits that do not combine, modifiers of
equal rank that disagree) is left open and asked (``ask``). Rules decide; a model may only propose a part, never a choice.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union, get_args

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Split = Literal["random", "stratified", "group", "time", "time_group", "rolling_origin", "spatial"]
LEVELS = ("experimental", "certified", "product_default")
SINGLE = ("primary_metric", "baseline", "calibration", "threshold_rule")   # one value: the most specific part sets it
LISTS = ("required_inputs", "guard_metrics", "model_families", "leakage_checks", "artifact_needs", "report_items")  # values add up
SPECIFIC = ("review_capacity", "rare_positives")   # the order in which a modifier's single value wins over the kind's
FORWARD = ("time", "time_group", "rolling_origin")  # a kind's split that keeps time order: never dropped by a modifier's split
TEXT = 300
REASON, PART = 600, 2400


class Detection(Contract):
    signal: str = Field(min_length=1, max_length=80, description="What is read: minority_rate, a time column, the person's words.")
    rule: str = Field(default="", max_length=TEXT, description="The threshold or pattern, e.g. '< 0.05'.")
    asks_user: bool = Field(default=False, description="A person must confirm (review capacity is always asked).")


class RecipePart(Versioned):
    """One kind or one modifier. A kind fills every field; a modifier only the fields it changes."""

    schema_version: Literal["dclab_core.recipe_part.v1"] = _version("recipe_part")
    name: str = Field(min_length=1, max_length=60)
    part_type: Literal["kind", "modifier"]
    version: str = Field(min_length=1, max_length=40)
    level: Literal["experimental", "certified", "product_default"] = "experimental"
    detection: list[Detection] = Field(default_factory=list, max_length=12)
    required_inputs: list[str] = Field(default_factory=list, max_length=12)
    split: Optional[Split] = None
    primary_metric: Optional[str] = Field(default=None, max_length=60)
    guard_metrics: list[str] = Field(default_factory=list, max_length=12)
    baseline: Optional[str] = Field(default=None, max_length=TEXT)
    model_families: list[str] = Field(default_factory=list, max_length=20)
    leakage_checks: list[str] = Field(default_factory=list, max_length=20)
    calibration: Optional[str] = Field(default=None, max_length=TEXT)
    threshold_rule: Optional[str] = Field(default=None, max_length=TEXT)
    artifact_needs: list[str] = Field(default_factory=list, max_length=20)
    report_items: list[str] = Field(default_factory=list, max_length=20)
    certification: list[str] = Field(default_factory=list, max_length=20, description="The replicas and real datasets behind its level.")

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


class Choice(Contract):
    field: str = Field(max_length=40)
    value: Union[str, list[str], None] = None
    part: str = Field(max_length=PART, description="name@version of the part (or parts) that decided it, 'person' (stated), 'solution' "
                      "(the Solution's declared columns), 'machine' (a kind's own machine decides it) or 'ask' (a person must settle it).")
    reason: str = Field(max_length=REASON)


class ResolvedRecipe(Versioned):
    """What one run uses: the kind, the modifiers, each choice with the part that made it and why, and what is left to ask."""

    schema_version: Literal["dclab_core.resolved_recipe.v1"] = _version("resolved_recipe")
    kind: str = Field(max_length=100)
    modifiers: list[str] = Field(default_factory=list, max_length=20)
    choices: list[Choice] = Field(default_factory=list, max_length=40)
    ask: list[str] = Field(default_factory=list, max_length=20, description="The fields a person must settle before the run.")

    def choice(self, name: str) -> Choice:
        found = next((c for c in self.choices if c.field == name), None)
        if found is None:
            raise KeyError(name)
        return found

    def value(self, name: str) -> Any:
        return self.choice(name).value


CONTRACTS["recipe_part"] = RecipePart
CONTRACTS["resolved_recipe"] = ResolvedRecipe


def _choice(field: str, value: Any, part: str, reason: str) -> Choice:
    """A choice whose words fit the contract, whatever the parts' names and reasons hold (a long column name never fails a run)."""
    return Choice(field=field, value=value, part=part[:PART], reason=reason[:REASON])


def _split(kind: RecipePart, modifiers: list[RecipePart]) -> Choice:
    named = [m for m in modifiers if m.split]
    if not named:
        return _choice("split", kind.split, kind.ref, f"the kind {kind.name}'s split")
    splits = {m.split for m in named}
    deciders = [m.ref for m in named]
    if kind.split in FORWARD:  # the kind's time order combines with a modifier's split, never gives way to it (the review of 30.10)
        splits.add(kind.split)
        deciders.insert(0, kind.ref)
    if "time_group" in splits:
        splits -= {"time", "group"}
    refs = ", ".join(deciders)
    if splits <= {"time", "group", "time_group"}:
        value = "time_group" if ("time" in splits and "group" in splits) or "time_group" in splits else next(iter(splits))
        why = ("forward in time, and each group on one side" if value == "time_group" else
               "the latest period is the holdout" if value == "time" else "each group on one side")
        return _choice("split", value, refs, f"{why} ({refs})")
    if len(splits) == 1:
        value = next(iter(splits))
        return _choice("split", value, refs, f"the {value} split ({refs})")
    return _choice("split", None, "ask", f"the parts ask for different splits ({', '.join(sorted(splits))}): ask the person")


def _single(name: str, kind: RecipePart, mods: list[RecipePart]) -> Choice:
    setting = [m for m in mods if getattr(m, name)]
    if setting:
        rank = min(SPECIFIC.index(m.name) if m.name in SPECIFIC else len(SPECIFIC) for m in setting)
        top = [m for m in setting if (SPECIFIC.index(m.name) if m.name in SPECIFIC else len(SPECIFIC)) == rank]
        values = {getattr(m, name) for m in top}
        if len(values) > 1:  # modifiers of equal rank disagree: the rules cannot settle it (the review of 30.10)
            return _choice(name, None, "ask", f"{', '.join(m.ref for m in top)} set different values ({', '.join(sorted(values))}): ask the person")
        winner = top[0]
        beaten = f"; over the kind's {getattr(kind, name)}" if getattr(kind, name) else ""
        return _choice(name, getattr(winner, name), ", ".join(m.ref for m in top), f"the most specific modifier, {winner.name}{beaten}")
    if getattr(kind, name) is not None:
        return _choice(name, getattr(kind, name), kind.ref, f"the kind {kind.name}'s default")
    return _choice(name, None, kind.ref, "no part sets one: the engine's shared rule applies")


def resolve(kind: RecipePart, modifiers: list[RecipePart] | tuple = (), stated: dict[str, Any] | None = None) -> ResolvedRecipe:
    """The parts merged with their precedence; ``stated``: what the person said for a field (it wins, with 'person'). A stated
    split must be one of ``Split``; a stated list field replaces the parts' list; an unknown field is refused."""
    if kind.part_type != "kind":
        raise ValueError(f"{kind.name} is a {kind.part_type}, not a kind")
    mods = list(modifiers)
    for m in mods:
        if m.part_type != "modifier":
            raise ValueError(f"{m.name} is a kind: one kind per problem, the rest are modifiers")
    if len({m.name for m in mods}) != len(mods):
        raise ValueError("a modifier is named twice")
    stated = {k: v for k, v in (stated or {}).items() if v not in (None, "", [])}
    unknown = set(stated) - {"split", *SINGLE, *LISTS}
    if unknown:
        raise ValueError(f"a recipe has no field {', '.join(sorted(unknown))}")
    if "split" in stated and stated["split"] not in get_args(Split):
        raise ValueError(f"split {stated['split']!r} is not one of {', '.join(get_args(Split))}")
    for name in SINGLE:
        if name in stated and not isinstance(stated[name], str):
            raise ValueError(f"{name} takes one value, not {type(stated[name]).__name__}")
    for name in LISTS:
        if name in stated and not (isinstance(stated[name], (list, tuple)) and all(isinstance(v, str) for v in stated[name])):
            raise ValueError(f"{name} takes a list of names")
    choices = [_choice("split", stated["split"], "person", "the person stated it") if "split" in stated else _split(kind, mods)]
    for name in SINGLE:
        choices.append(_choice(name, stated[name], "person", "the person stated it") if name in stated else _single(name, kind, mods))
    for name in LISTS:
        if name in stated:
            choices.append(_choice(name, list(stated[name]), "person", "the person stated it"))
            continue
        values: list[str] = []
        parts: list[str] = []
        for p in (kind, *mods):
            new = [v for v in getattr(p, name) if v not in values]
            if new:
                values += new
                parts.append(p.ref)
        choices.append(_choice(name, values, ", ".join(parts) or kind.ref, "every part's, in order (the kind's first)" if values else "no part asks for one"))
    return ResolvedRecipe(kind=kind.ref, modifiers=[m.ref for m in mods], choices=choices, ask=[c.field for c in choices if c.part == "ask"])


# ---------------------------------------------------------------------------------------------------- certification (30.12)

#: the recipes document's five pass criteria: the detection is right, every planted trap is caught, it beats or matches the
#: baseline (or says it does not), the artifact passes save, reload and predict, and the result is the same with the LLM off
CRITERIA = ("detection", "traps", "baseline", "artifact", "llm_off")
REAL_NEEDED = 2


def run_passes(criteria: dict[str, Optional[bool]]) -> tuple[bool, list[str], list[str]]:
    """A run passes when every criterion was measured and passed: (passed, the criteria failed, the criteria not measured)."""
    failed = [c for c in CRITERIA if criteria.get(c) is False]
    pending = [c for c in CRITERIA if criteria.get(c) is None]
    return not failed and not pending, failed, pending


def certify(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """A part's level from its certification runs (each ``{"name", "source": "replica"|"real", "applies": bool,
    "criteria": {...}}``): certified when a trap replica and at least two real datasets where the part applies each pass
    every criterion (a run where it rightly does not apply checks its absence, and never counts toward the level);
    otherwise experimental, with what failed, what is not measured yet and what is missing. Product default needs more
    than runs (no wrong detection on every R&D dataset, no regression in twin runs): never given here."""
    judged = []
    for r in runs:
        passed, failed, pending = run_passes(dict(r.get("criteria") or {}))
        judged.append({"name": str(r.get("name"))[:120], "source": r.get("source"), "applies": bool(r.get("applies", True)), "passed": passed,
                       "failed": failed, "pending": pending})
    replicas = [j for j in judged if j["source"] == "replica" and j["applies"] and j["passed"]]
    real = [j for j in judged if j["source"] == "real" and j["applies"] and j["passed"]]
    level = "certified" if replicas and len(real) >= REAL_NEEDED else "experimental"
    missing = ([] if replicas else ["a trap replica where the part applies, passing every criterion"]) + \
              ([] if len(real) >= REAL_NEEDED else [f"{REAL_NEEDED - len(real)} more real dataset(s) where the part applies, passing every criterion"])
    return {"level": level, "runs": judged, "missing": missing,
            "pending": sorted({c for j in judged for c in j["pending"]}), "failed": sorted({c for j in judged for c in j["failed"]})}


__all__ = ["Split", "LEVELS", "FORWARD", "Detection", "RecipePart", "Choice", "ResolvedRecipe", "resolve", "CRITERIA", "run_passes", "certify"]
