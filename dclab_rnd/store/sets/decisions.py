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
            out[(d["record"], d["model_type"])] = {k: d.get(k) for k in ("folder", "quality", "curator", "model_type", "record")}  # the fields the sets read:
    return out  # 1.5 million whole choices took 2.7 GB (2026-11-03)


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


def _purposes_of(answerer: str) -> set[str]:
    """The model purposes an answerer's request may have: an agent's own purposes, else the answerer's name (a model use)."""
    from ...agents import harness

    agent = harness.AGENTS.get(str(answerer))
    return set(agent.purposes) if agent is not None and agent.purposes else {str(answerer)}


def thinking_by_request(layout: Layout, ids: set[str]) -> dict[str, tuple[str, str, Any]]:
    """request id -> (purpose, the kept thinking summary, or its redacted text when the purpose saw table values), for the
    requests named by model answers (2026-10-30: model answers name their request in ``source``)."""
    import gzip

    out: dict[str, tuple[str, str, Any]] = {}
    if not ids:
        return out
    base = layout.records_of("model_requests")
    for path in sorted(base.rglob("*.jsonl.gz")) if base.exists() else []:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if '"model_exchanges' not in line:  # every kept exchange is parsed once; its id is then looked up
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                b, prov = r.get("body") or {}, r.get("provenance") or {}
                if prov.get("consent") != "train":  # the review: a workspace that does not allow training lends no thinking
                    continue
                rid = str(b.get("model_request") or "")
                blocks = b.get("thinking") or b.get("thinking_redacted") or []
                text = "\n\n".join(str(x.get("text") or "") for x in blocks if isinstance(x, dict)).strip()
                if rid in ids and text and rid not in out:
                    out[rid] = (str(b.get("purpose") or ""), text[:2000], prov)
    return out


from dclab_core.ledger import SHADOW  # noqa: E402 - a shadow's answer: never an answer of the record


def _model_answers(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [a for a in body.get("answers") or [] if isinstance(a, dict) and a.get("by") == "model_role" and a.get("source") != SHADOW]


def _used(body: dict[str, Any], a: dict[str, Any]) -> bool:
    return body.get("used") == "model_role" or a.get("value") == body.get("value_used")


def _linked_thinking(body: dict[str, Any], thoughts: dict[str, tuple]) -> Optional[tuple[str, Any]]:
    """(the thinking, its exchange's provenance) of the model answer that was used (or gave the value used), from its own
    request of its own agent; None otherwise."""
    for a in _model_answers(body):
        if not a.get("source") or not _used(body, a):
            continue
        found = thoughts.get(str(a["source"]))
        if found and found[0] in _purposes_of(str(a.get("answerer"))):
            return found[1], found[2]
    return None


LINK_NOTE = "lost before 30 Oct: a model's answer recorded before answers named their request, so its thinking cannot be linked"
UNNAMED_NOTE = "a model's answer that named no request (recorded outside a model's own tool call): its thinking cannot be linked"
NOT_KEPT_NOTE = "its request kept no thinking summary: the model did not think, or the call came before 28 Oct"
OTHER_NOTE = "its named request is another agent's call, so that thinking is not this decision's reason"
LINKS_FROM = "2026-10-30T10:00"  # answers recorded inside a model's tool call name their request from here on


def _why_none(body: dict[str, Any], thoughts: Optional[dict[str, tuple]] = None) -> Optional[str]:
    """Why a row with a model's answer has no reasoning (None for a row no model answered: it needs none)."""
    models = _model_answers(body)
    if not models:
        return None
    used = [a for a in models if _used(body, a)]
    if not used:
        return f"the {body.get('used') or 'other'} answer was used; the model answered beside it, so its thinking is not this decision's reason"
    named = [a for a in used if a.get("source")]
    if not named:
        return LINK_NOTE if str(body.get("at") or "") < LINKS_FROM else UNNAMED_NOTE
    if thoughts and any(str(a["source"]) in thoughts for a in named):
        return OTHER_NOTE
    return NOT_KEPT_NOTE


def build(layout: Optional[Layout] = None, scripted: Optional[frozenset[str]] = None) -> dict[str, Any]:
    """``scripted``: the workspaces whose people are a script (``keepers.workspaces.scripted_names``): a person's label
    there is no human's, so it never confirms a decision (S8, 2026-10-25: test clicks and written people)."""
    from .. import shapers
    from ..keepers.workspaces import scripted_names

    scripted = scripted_names() if scripted is None else scripted
    layout = layout or Layout()
    placed = placements(layout)
    # One pass over the shaped records, keeping only what the sets need (2026-11-03: every record of the kind was held at once as an
    # object, 6.6 GB on an 8 GB Mac, and the scheduled whole run was stopped by its swap guard): the decisions the curators placed,
    # every label by the decision it labels, and the model-role sources the answers name.
    wanted = {rid for rid, _ in placed}
    records: dict[str, StoreRecord] = {}
    labels: dict[str, list[StoreRecord]] = defaultdict(list)
    named: set[str] = set()
    for r in shapers.records(layout, "decisions"):
        if r.type == "label":
            labels[str(r.body.get("record") or r.body.get("record_id"))].append(r)
        elif r.type == "decision":
            named |= {str(a.get("source")) for a in r.body.get("answers") or []
                      if isinstance(a, dict) and a.get("by") == "model_role" and str(a.get("source") or "").startswith("mr")}
            if r.id in wanted:
                records[r.id] = r
    sets: dict[tuple[str, str], list[DecisionExample]] = defaultdict(list)
    out_of_sets: dict[str, int] = defaultdict(int)
    set_aside: set[str] = set()
    thoughts = thinking_by_request(layout, named)
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
        linked = _linked_thinking(record.body, thoughts)
        thought = linked[0] if linked else None
        try:
            from dclab_core.store import Provenance

            more = [Provenance.model_validate({k: v for k, v in linked[1].items() if k in Provenance.model_fields})] if linked else []
            ex = DecisionExample(id=f"{model_type}:{rid}", model_type=model_type, split=split_of(str(group)), provenance=record.provenance,
                                 more=more,
                                 weight=weight, point=str(record.body.get("point")), group=str(choice.get("folder") or "").split("/")[0] or "general",
                                 split_group=str(group),
                                 state=state, options=[str(o) for o in record.body.get("options") or []], answers=answers, label=value, label_source=source,
                                 **({"reasoning": thought, "reasoning_source": "recorded"} if thought else
                                    {"reasoning": why, "reasoning_source": "from_record"} if why else
                                    {"no_reasoning": _why_none(record.body, thoughts)} if _why_none(record.body, thoughts) else {}))
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
