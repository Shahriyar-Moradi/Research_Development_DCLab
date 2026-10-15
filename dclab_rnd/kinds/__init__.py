"""Problem kinds (Part 5, package 16.1): one module per kind, one registry, one interface.

``kinds.get(key)`` returns a kind; ``kinds.resolve(solution, pack)`` the kind a project runs as (the solution's
declared kind, else its Domain pack's kind for that task, else text + tabular for a binary target with text columns,
else the task's own kind); ``kinds.detect(problem, profile)`` the kind a problem sentence and a profiled table point
to, with why. Every module in this package that defines ``KIND`` is registered on import, so adding a kind touches only
its module and its tests. :class:`ProblemKind` in ``base.py`` is the interface.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from .base import BASE_TASKS, NoFields, ProblemKind, default_split

__all__ = ["BASE_TASKS", "NoFields", "ProblemKind", "all_kinds", "default_split", "describe_all", "detect", "for_pack", "get",
           "register", "resolve", "unregister"]

_REGISTRY: dict[str, ProblemKind] = {}


def register(kind: ProblemKind, replace: bool = False) -> ProblemKind:
    """Add a kind. Its base task must be one the engine runs; a key is registered once unless ``replace``."""
    if not kind.key:
        raise ValueError("A kind needs a key")
    if kind.base_task not in BASE_TASKS:
        raise ValueError(f"Kind {kind.key!r} runs as {kind.base_task!r}, which is not one of {BASE_TASKS}")
    if not kind.metrics:
        raise ValueError(f"Kind {kind.key!r} declares no metric")
    if kind.key in _REGISTRY and not replace and _REGISTRY[kind.key] is not kind:
        raise ValueError(f"Kind {kind.key!r} is already registered")
    _REGISTRY[kind.key] = kind
    return kind


def unregister(key: str) -> None:
    _REGISTRY.pop(key, None)


def _load() -> None:
    for module in pkgutil.iter_modules(__path__):
        if module.name == "base" or module.name.startswith("_"):
            continue
        kind = getattr(importlib.import_module(f"{__name__}.{module.name}"), "KIND", None)
        if isinstance(kind, ProblemKind):
            register(kind, replace=True)


def get(key: str) -> ProblemKind:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown problem kind {key!r}; known: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key]


def all_kinds() -> list[ProblemKind]:
    """The kinds, most specific first, then by key."""
    return sorted(_REGISTRY.values(), key=lambda k: (-k.specificity, k.key))


def for_pack(pack: str | None, task: str | None = None) -> ProblemKind | None:
    """The kind a Domain pack means for this task (the most specific one), or None."""
    return next((k for k in all_kinds() if pack and pack in k.packs and (task is None or k.base_task == task)), None)


def _value(solution: Any, key: str) -> Any:
    return solution.get(key) if isinstance(solution, dict) else getattr(solution, key, None)


def resolve(solution: Any, pack: str | None = None) -> ProblemKind:
    """The kind a solution runs as. ``solution`` is a Solution or its dict; ``pack`` the project's Domain pack key."""
    declared = _value(solution, "kind")
    if declared:
        return get(declared)
    task = _value(solution, "task")
    by_pack = for_pack(pack, task)
    if by_pack is not None and by_pack.key != task and by_pack.applies(solution):  # a text pack on a table without text runs as the task's kind
        return by_pack
    if task == "regression" and _value(solution, "spatial"):  # 20.10: blocks of places adopted under any pack are the places kind
        return get("places")
    if by_pack is not None and by_pack.applies(solution):
        return by_pack
    if task == "binary" and _value(solution, "text_columns"):
        return get("text_tabular")
    return get(task)


def detect(problem: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """The kind the words and the table point to: {"key", "why"}. The most specific kind with a reason wins; then the
    task dclab_core reads in the sentence; then binary, said plainly."""
    for kind in all_kinds():
        if kind.specificity <= 0:
            continue
        why = kind.detect(problem, profile)
        if why:
            return {"key": kind.key, "why": why}
    from dclab_core.understand import understand

    brief = understand(problem or "")
    if brief.task.known and brief.task.value in _REGISTRY:
        return {"key": brief.task.value, "why": f"the problem reads as a {brief.task.value} task"}
    for kind in all_kinds():  # the general kinds' own words last
        why = kind.detect(problem, profile) if kind.specificity <= 0 else None
        if why:
            return {"key": kind.key, "why": why}
    return {"key": "binary", "why": "nothing points elsewhere, so DCLab starts from binary classification"}


def describe_all() -> list[dict[str, Any]]:
    return [k.describe() for k in all_kinds()]


_load()
