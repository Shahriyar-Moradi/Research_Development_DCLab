"""Model help, on the host's terms.

dclab_core never calls a model provider. A host that has one (the R&D gateway, the main product's LLM gateway) passes
an :class:`Advisor`; :func:`consult` sends it a purpose and compact evidence and returns a validated answer, or
``None``, in which case the caller takes its deterministic path. An advisor that fails, times out or answers something
invalid is treated as no answer: it never raises into the solving code. ``on_error`` lets the host record why
(``"failed"`` or ``"invalid"``), as its governance requires.

What crosses is checked before it leaves (:func:`check_evidence`, the rules of :mod:`dclab_core.evidence`): aggregates,
names and short summaries, never a DataFrame, an array, a long list or a table of records, which is how rows (and
holdout labels) would leak to a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable

from dclab_core import evidence as ev

T = TypeVar("T")


class EvidenceError(ValueError):
    """The evidence would carry rows or raw values to a model."""


@runtime_checkable
class Advisor(Protocol):
    def ask(self, purpose: str, evidence: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Answer ``purpose`` from ``evidence`` with a JSON-like mapping, or ``None`` to abstain."""
        ...


class NoAdvisor:
    """The deterministic path: always abstains."""

    def ask(self, purpose: str, evidence: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        return None


def check_evidence(evidence: Any) -> None:
    """Raise :class:`EvidenceError` unless ``evidence`` is a compact summary."""
    if not isinstance(evidence, Mapping):
        raise EvidenceError("evidence is a mapping of named summaries")
    try:
        ev.check(evidence, ev.ADVISOR)
    except ev.NotASummary as exc:
        raise EvidenceError(str(exc)) from None


def consult(advisor: Optional[Advisor], purpose: str, evidence: Mapping[str, Any],
            parse: Callable[[Mapping[str, Any]], T],
            on_error: Optional[Callable[[str, Optional[BaseException]], None]] = None) -> Optional[T]:
    """Ask ``advisor`` and validate its answer with ``parse``; ``None`` means: use the deterministic path.

    The evidence is checked first, and an :class:`EvidenceError` is raised to the caller: sending rows is a bug in the
    caller, not a model failure. On the advisor's side, a failure (``"failed"``) or an answer ``parse`` rejects
    (``"invalid"``) returns ``None`` and is reported to ``on_error``; an abstention (``None``) is not an error.
    """
    check_evidence(evidence)
    if advisor is None:
        return None

    def report(kind: str, exc: Optional[BaseException]) -> None:
        if on_error is not None:
            on_error(kind, exc)

    try:
        answer = advisor.ask(purpose, ev.plain(dict(evidence)))
    except Exception as exc:  # noqa: BLE001 - a provider can fail in any way; the deterministic path takes over
        report("failed", exc)
        return None
    if answer is None:
        return None
    if not isinstance(answer, Mapping):
        report("invalid", None)
        return None
    try:
        return parse(answer)
    except Exception as exc:  # noqa: BLE001 - an invalid answer is no answer
        report("invalid", exc)
        return None
