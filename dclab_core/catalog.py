"""The parts of the data catalog that follow from the contracts (package 23.9): the event kinds with the class of every field,
the decision points, the contracts and their versions, and the field classes with what any export may keep.

A host adds what only it knows (where each kind is emitted, its tables, its retention) and renders the whole; the studio's
``dclab_rnd.data_catalog`` does, and the main product generates its page the same way. Nothing here reads a file or the clock.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# what each class of field is, and whether any export may keep it (the studio's consent.SHAREABLE)
CLASS_NOTES = {
    "meta": ("M", "ids, codes, names of things; leaves at improve and train"),
    "count": ("C", "numbers; leaves at improve and train"),
    "person_text": ("P", "what a person typed; leaves redacted"),
    "agent_text": ("A", "what an agent said; leaves redacted"),
    "model_text": ("X", "a prompt or a reply; leaves only at train, redacted, and only if the workspace kept it and the purpose was allowed to keep it"),
    "cell_values": ("V", "a value of a table; never recorded"),
}
LEAVES = frozenset({"meta", "count", "person_text", "agent_text"})
TRAIN_ONLY = frozenset({"model_text"})  # leaves at train alone


def classes() -> dict[str, Any]:
    return {k: {"letter": v[0], "meaning": v[1], "leaves": k in LEAVES} for k, v in CLASS_NOTES.items()}


def event_kinds(emitted_by: Optional[Mapping[str, list[str]]] = None) -> list[dict[str, Any]]:
    """Every kind of the experience contract: its fields by class, what never leaves, the level an export needs, and (when the
    host says) where it is emitted."""
    from dclab_core import experience as x

    emitted_by = emitted_by or {}
    out = []
    for kind, fields in sorted(x.KINDS.items()):
        names: dict[str, list[str]] = {}
        for field, cls in fields.items():
            names.setdefault(cls, []).append(field)
        out.append({"kind": kind, "family": kind.split(".")[0], "fields": dict(sorted(fields.items())),
                    "never_leaves": sorted(f for c, n in names.items() if c not in LEAVES and c not in TRAIN_ONLY for f in n),
                    "train_only": sorted(f for c, n in names.items() if c in TRAIN_ONLY for f in n),
                    "leaves_at": "improve" if any(c in LEAVES for c in names) else "train" if any(c in TRAIN_ONLY for c in names) else "never",
                    "emitted_by": list(emitted_by.get(kind, []))})
    return out


def decision_points() -> list[dict[str, Any]]:
    from dclab_core.ledger import POINTS

    return [{"point": k, "kind": p.kind, "answer": p.answer, "subjects": list(p.subjects), "about": p.about, "main_product": p.main_product}
            for k, p in sorted(POINTS.items())]


def contracts() -> list[dict[str, Any]]:
    import dclab_core.schemas  # noqa: F401  (registers every contract)
    from dclab_core.contracts import CONTRACTS

    return [{"contract": name, "version": m.model_fields["schema_version"].default, "reads": sorted(m.READS)} for name, m in sorted(CONTRACTS.items())]


def build(emitted_by: Optional[Mapping[str, list[str]]] = None) -> dict[str, Any]:
    from dclab_core import __version__

    return {"dclab_core": __version__, "classes": classes(), "event_kinds": event_kinds(emitted_by), "decision_points": decision_points(), "contracts": contracts()}


__all__ = ["CLASS_NOTES", "LEAVES", "classes", "event_kinds", "decision_points", "contracts", "build"]
