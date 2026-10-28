"""The decision sets (32.11): M1 students, M2 workflow decisions, M3 machine-learning decisions, and the teacher–student
pairs, built from the curators' choices and the shaped ledger records only.

Each placed decision becomes a :class:`dclab_core.store.DecisionExample`: its point, its group (the system, the family, or
the point itself for a student), its state as it was when it was made (bands and codes, never a value), its options, every
answerer's answer, and its label with the label's source and weight, the best one it has:

- a confirmed label (a person's, a measurement's, a benchmark's, or a blind draw's, weighted by its draw) first;
- a teacher's (Claude's, GLiNER's) as silver, at the curator's weight, apart;
- none (the example still teaches the state and the answers; its label is None).

The split is the record's own (its group's hash); a decision whose group is unknown cannot be sealed and is kept out (with
why). For M1, a decision that a student answered beside its teachers is also a teacher–student pair
(``m1_students/<point>/pairs``): the student's answer, each teacher's, and the label.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Optional

from dclab_core.datasets import split_of
from dclab_core.store import CONFIRMED, LEDGER_SOURCES, Answer, DecisionExample, StoreRecord

from ..layout import Layout
from ..store import Store
from . import write_set

MODELS = ("m1_students", "m2_workflow_decisions", "m3_ml_decisions")
RANK = {"person": 0, "measured": 1, "benchmark": 2, "blind": 3, "teacher": 4, "rule": 5}
STUDENTS = ("classifier", "student")
SILVER = 0.5  # the weight of a teacher's or a rule's label against a confirmed one (the owner's rule: silver, apart, lower)



def decision_reasoning(point: str, state: dict[str, Any], options: list[str], answers: list[Any]) -> Optional[str]:
    """None: code cannot explain a decision without its answer (the review of C2). The rule's answer is the label of a
    rule-labelled row and argues against a person's label (a person's label is written when they disagree), so a reasoning
    built from it teaches copying; one built from the state alone restates the input. A decision row's reasoning comes only
    from a real source (a model's recorded thinking, a checked teacher), when there is one; until then it is counted as
    missing, never filled."""
    return None

def placements(layout: Layout) -> dict[tuple[str, str], dict[str, Any]]:
    """The curators' standing choice per (record, model type): the latest a rule made and the write tool let through."""
    store = Store(layout)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in sorted(store.entries("curation"), key=lambda e: (e.captured_at, e.version)):
        for line in store.read(entry).decode("utf-8").splitlines():
            d = json.loads(line)
            if d.get("used") is False or d.get("refused") or not str(d.get("curator", "")).startswith("rule:"):
                continue
            out[(d["record"], d["model_type"])] = d
    return out


def _best_label(labels: list[StoreRecord]) -> tuple[Any, Optional[str]]:
    """The best label among those no later label supersedes (the ledger's rule: a new label supersedes an older one, never
    edits it); an empty value is a retraction (36.2: a route the person finally turned down), never a label."""
    gone = {lab.body.get("supersedes") for lab in labels if lab.body.get("supersedes")}
    best = None
    for lab in labels:
        if lab.body.get("id") in gone or lab.body.get("value") is None:
            continue
        source = LEDGER_SOURCES.get(str(lab.body.get("source")), lab.body.get("source"))
        if source not in RANK:
            continue
        if best is None or RANK[source] < RANK[best[1]]:
            best = (lab.body.get("value"), source)
    return best if best else (None, None)


def build(layout: Optional[Layout] = None, scripted: Optional[frozenset[str]] = None) -> dict[str, Any]:
    """``scripted``: the workspaces whose people are a script (``keepers.workspaces.scripted_names``): a person's label
    there is no human's, so it never confirms a decision (S8, 2026-10-25: test clicks and written people)."""
    from .. import shapers
    from ..keepers.workspaces import scripted_names

    scripted = scripted_names() if scripted is None else scripted
    layout = layout or Layout()
    placed = placements(layout)
    records = {r.id: r for r in shapers.records(layout, "decisions")}
    labels: dict[str, list[StoreRecord]] = defaultdict(list)
    for r in records.values():
        if r.type == "label":
            labels[str(r.body.get("record") or r.body.get("record_id"))].append(r)
    sets: dict[tuple[str, str], list[DecisionExample]] = defaultdict(list)
    out_of_sets: dict[str, int] = defaultdict(int)
    set_aside: set[str] = set()
    for (rid, model_type), choice in sorted(placed.items()):
        record = records.get(rid)
        if model_type not in MODELS or record is None or record.type != "decision" or choice.get("quality") == "out":
            continue
        group = record.body.get("group")
        if not group:
            out_of_sets["no group: it cannot be sealed"] += 1
            continue
        own = labels.get(str(record.body.get("id")), [])
        if record.provenance.workspace in scripted:  # a scripted person's label is a script's: never a confirmed human label
            own = [lab for lab in own if LEDGER_SOURCES.get(str(lab.body.get("source")), lab.body.get("source")) != "person"]
            if len(own) < len(labels.get(str(record.body.get("id")), [])):
                set_aside.add(rid)  # once per record, however many sets it is placed in
        value, source = _best_label(own)
        weight = 1.0 if source in CONFIRMED or source is None else SILVER  # a teacher's or a rule's label weighs less, kept apart by its source
        if source == "blind":  # a blind draw's label carries its draw's weight (the ledger's), never 1 by default
            weight = float(next((lab.body.get("weight") for lab in labels.get(str(record.body.get("id")), []) if lab.body.get("weight") is not None), 1.0))
            weight = min(1.0, max(0.0, weight))
        answers = [Answer(by=str(a.get("by") or a.get("answerer") or "unknown"), value=a.get("value"), reason=a.get("reason"))
                   for a in record.body.get("answers") or [] if isinstance(a, dict)]
        state = {k: v for k, v in (record.body.get("state") or {}).items()}
        why = decision_reasoning(str(record.body.get("point")), state, [str(o) for o in record.body.get("options") or []], answers)
        try:
            ex = DecisionExample(id=f"{model_type}:{rid}", model_type=model_type, split=split_of(str(group)), provenance=record.provenance,
                                 weight=weight, point=str(record.body.get("point")), group=str(choice.get("folder") or "").split("/")[0] or "general",
                                 split_group=str(group),
                                 state=state, options=[str(o) for o in record.body.get("options") or []], answers=answers, label=value, label_source=source,
                                 **({"reasoning": why, "reasoning_source": "from_record"} if why else {}))
        except Exception as exc:  # noqa: BLE001 - a record whose state the contract refuses (a value in it) never enters a set
            out_of_sets[f"refused by its contract: {type(exc).__name__}"] += 1
            continue
        sets[(model_type, str(choice.get("folder")))].append(ex)
        if model_type == "m1_students" and any(a.by.startswith(STUDENTS) for a in answers):
            sets[(model_type, f"{choice.get('folder')}/pairs")].append(ex.model_copy(update={"id": f"{ex.id}:pair"}))
    if set_aside:
        out_of_sets["decisions whose scripted person's label was set aside"] = len(set_aside)
    report = {"sets": {}, "out_of_sets": dict(out_of_sets)}
    abouts = {"m1_students": "Every decision of one point with its teachers' answers and its best label: what a student of that point learns.",
              "m2_workflow_decisions": "The workflow's choices of one point, by system (dclab, rnd, internal).",
              "m3_ml_decisions": "The technical choices of one point on tables, by family."}
    for (model_type, folder), examples in sorted(sets.items()):
        sources = [{"entry": e.provenance.entry, "sha256": e.provenance.sha256} for e in examples]
        about = abouts[model_type] + (" Pairs: the decisions a student answered beside its teachers." if folder.endswith("/pairs") else "")
        m = write_set(layout, model_type, folder, examples, sources={json.dumps(s, sort_keys=True): s for s in sources}.values(), about=about)
        report["sets"][f"{model_type}/{folder}"] = m.counts
    return report


__all__ = ["build", "placements", "MODELS"]
