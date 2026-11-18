"""Who wrote a training row, for every kind of training data DCLab keeps (the owner, 2026-11-18).

The owner's rule: everything a provider's model wrote is kept and prepared, and no training run or training export uses it
until the owner records a rights review (``provider_outputs.may_train``). "Everything" is the owner's answer of 2026-11-18:
a label, an answer, a model's thinking, and also the words and truths of a written person or request, whether a model wrote
them through DCLab's gateway or a Claude Code session wrote them into a file. Data that running code produced (an invented
table, a rule's answer, a measured label) is the code's, not a model's.

``writer_of(row)`` names the provider that wrote some part of a stored row (``claude:...``, ``frontier:...``, ``mixed:...``),
or None when no provider did. ``held(row)`` is True when that writer must stay out of training today. Every training reader
and training export asks ``held`` before it uses a row; ``python -m dclab_rnd.store.writers audit`` counts every set.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any, Optional

from . import provider_outputs
from .export_chat import provider_of
from .sets.llm import answerer_of, silver_of

# A test's workspace: the people and requests in it were written for the test (journeys, scripted people, exam people), by
# Claude through the gateway or by a Claude Code session. A real person's words never live there.
TEST_WORKSPACES = ("live_tests__",)
# Written sets whose writer is known from its records (checked 2026-11-18). Any other written set (a journey set, a request
# set) was written in a Claude Code session: every one of them was committed with Claude as co-author.
WRITTEN_SETS = {
    "intent_requests_v1": "claude:claude-sonnet-5-5",  # every row names claude-sonnet-5-5 (home_request_writer a, b and c)
    "homework_v1": "claude:claude-sonnet-5-5",  # journey_writer: the store's 721 requests of that purpose are all Claude Sonnet 5.5
    "homework_v2": "frontier:journey_writer_b",  # written through the gateway; its requests were not captured, so the model is unknown
    "homework_v3": "frontier:journey_writer_d",  # the same
}
CLAUDE_CODE = "claude:claude_code"
LABELLER = "routing_labeler"


class HeldForRights(ValueError):
    """Every row a training run would use was written by a provider's model: nothing is left to train on until the review."""


def may_train() -> bool:
    return provider_outputs.may_train()


def model_answerer(name: Any) -> Optional[str]:
    """A model's name as an answerer (``claude:...``, ``openrouter:...``, ``local:...``); None when no name was given."""
    return answerer_of({"mode": "llm", "model": str(name)}) if name else None


def set_writer(name: Any) -> Optional[str]:
    """Who wrote a written set (a journey set such as ``homework_v1`` or ``sealed_v2``, a request set): never None for a set."""
    return WRITTEN_SETS.get(str(name), CLAUDE_CODE) if name else None


def _provider(who: Optional[str]) -> Optional[str]:
    return who if who and silver_of(who) else None


def _routing(row: dict[str, Any]) -> Optional[str]:
    """A routing row: its words (a written request, an exam's person, a test's scripted person) and its label (a model's)."""
    label_by = str(row.get("label_by") or "")
    parts = label_by.split(":")
    if parts[0] == "model" or row.get("label_source") == "silver":
        if len(parts) >= 3 and parts[1] == "model":
            return _provider(model_answerer(parts[2])) or "frontier:unknown"  # model:model:<name>, a ledger label
        if len(parts) >= 3 and parts[2] == LABELLER:
            return _provider(model_answerer(parts[1])) or "frontier:unknown"  # model:<name>:routing_labeler
        giver = parts[1] if len(parts) > 1 else ""
        reading = next((r for r in row.get("readings") or [] if r.get("by") == "model" and r.get("answerer") == giver and r.get("model")), None)
        return _provider(model_answerer(reading["model"])) if reading else "frontier:unknown"
    if parts[0] == "written" and len(parts) > 1:
        return set_writer(parts[1])
    if label_by.startswith("conversation_exam:"):
        return "frontier:exam_person"  # an exam's written person: Claude Haiku or Qwen, not named on the row
    if str(row.get("origin")) in ("written", "eval", "scripted"):
        return CLAUDE_CODE  # a request written for a test or an eval, in a Claude Code session
    return None


def reading_held(reading: dict[str, Any]) -> bool:
    """A model's reading (its answer and its thinking) on a routing row is a provider's output unless its model says otherwise."""
    if reading.get("by") != "model" or may_train():
        return False
    return bool(_provider(model_answerer(reading.get("model")) or "frontier:unknown"))


def writer_of(row: dict[str, Any]) -> Optional[str]:
    """The provider that wrote some part of a stored row, or None. Read by the row's shape (its schema_version)."""
    schema = str(row.get("schema_version") or "")
    workspace = str((row.get("provenance") or {}).get("workspace") or "")
    if schema.startswith("dclab_core.decision_example"):
        # its inputs are bands code computed; only its label can be a model's (a teacher's: Claude through decision_labeler)
        return "frontier:teacher" if row.get("label") is not None and row.get("label_source") == "teacher" else None
    if schema.startswith("dclab_core.routing_example"):
        return _routing(row)
    if schema.startswith("dclab_core.judged_example"):
        by, judge = str(row.get("candidate_by") or ""), str(row.get("judge") or "")
        if by.startswith("model:"):
            return _provider(model_answerer(by.split(":", 1)[1])) or "frontier:unknown"
        return CLAUDE_CODE if by in ("reviewer", "development") or judge == "developer" else None
    if schema.startswith("dclab_core.episode"):
        return CLAUDE_CODE if row.get("process") == "dclab_code" else None
    if schema.startswith("dclab_core.experiment_record"):
        return None  # measured by code
    if schema.startswith("dclab_core.preference_pair"):
        return CLAUDE_CODE if "dclab_code" in str(row.get("id") or "") else None
    if schema.startswith("dclab_core.retrieval_example"):
        # an agent's own search: a model wrote the query (the rules do not search in words); a person's search is not held
        return "frontier:agent_search" if "agent_step" in str((row.get("provenance") or {}).get("rule") or "") else None
    who, _ = provider_of(row)  # a chat example or a trajectory
    if who:
        return who
    if workspace.startswith(TEST_WORKSPACES) and ("messages" in row or "steps" in row):
        return CLAUDE_CODE  # a test's conversation: its person was written for the test
    return None


def held(row: dict[str, Any]) -> bool:
    """True when a provider wrote some part of the row and the owner has not recorded the rights review."""
    return writer_of(row) is not None and not may_train()


def held_writer(who: Optional[str]) -> bool:
    return bool(_provider(who)) and not may_train()


def audit(layout: Any = None) -> dict[str, Any]:
    """Every set's train and dev rows (sealed rows are never read here), counted by who wrote them."""
    from .layout import Layout

    layout = layout or Layout()
    root = layout.root / "sets"
    out: dict[str, Any] = {}
    for path in sorted(root.glob("**/train.jsonl")) + sorted(root.glob("**/dev.jsonl")):
        name = str(path.parent.relative_to(root))
        doc = out.setdefault(name, {"rows": 0, "provider": Counter()})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                doc["rows"] += 1
                who = writer_of(json.loads(line))
                if who:
                    doc["provider"][who.split(":", 1)[0] + (":claude_code" if who == CLAUDE_CODE else "")] += 1
    return {k: {"rows": v["rows"], "provider": dict(v["provider"])} for k, v in sorted(out.items())}


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m dclab_rnd.store.writers audit``: every set's train and dev rows by who wrote them, as JSON, then one line
    per model type. Reads only; writes nothing."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] != ["audit"]:
        print("usage: python -m dclab_rnd.store.writers audit", file=sys.stderr)
        return 2
    report = audit()
    totals: dict[str, Counter] = {}
    for name, doc in report.items():
        total = totals.setdefault(name.split("/", 1)[0], Counter())
        total["rows"] += doc["rows"]
        total.update(doc["provider"])
    print(json.dumps({"may_train": may_train(), "sets": report}, indent=1))
    for model_type, total in sorted(totals.items()):
        print(model_type, dict(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
