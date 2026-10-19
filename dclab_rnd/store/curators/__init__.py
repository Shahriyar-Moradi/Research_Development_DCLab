"""The curators (32.9): the agentic layer that puts the right data in the right place, one curator per model type and a
librarian that sends each record to the curators whose model it could teach.

Each curator decides, for every record the librarian sends it: does it belong to its model, where it goes (its folder: a
point, a system and a point, a family and a point, a stage, a problem kind, a process, what is judged, a corpus), and its
quality (``keep``; ``silver`` with a weight, apart from confirmed labels; ``out`` with the reason: a duplicate, a broken
record, nothing to learn yet). Every proposal goes through ``write.check``; a refused one is kept with why; a record kept
out of a set is listed in ``quarantine/<reason>/``, still whole in ``raw/``.

The rules are the floor and decide by default, with no spend. A curator served by a model (``model.py``, the gateway
purpose ``store_curator``) answers in shadow beside them: its answers are kept, compared and never used until it measures
better on a person's review of a sample and a person switches it on.

``curate(layout, run)`` reads the records the shapers wrote since the last curation (a month file whose hash changed), writes
every choice to ``raw/curation/`` as ``curation/<run>.jsonl`` (one :class:`dclab_core.store.CurationDecision` a line) and the
quarantine lists, and returns the counts per model type and quality.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Callable, Iterator, Optional

from dclab_core.ledger import POINTS
from dclab_core.store import CONFIRMED, KINDS, LEDGER_SOURCES, MODEL_TYPES, StoreRecord

from ..layout import Layout
from ..store import Store, now
from . import write

INTERNAL_POINTS = {"department.proposal", "workflow.g4", "team.next_node", "critic.send_back", "agent.tool_choice"}  # the studio's own automation
M4_STAGES = {"home": "understand_problem", "intern": "plan", "workspace_chat": "understand_data", "twin": "design_solution", "mcp": "plan"}


def _code(text: Any) -> str:
    import re

    return re.sub(r"[^a-z0-9_.\-]+", "_", str(text or "unknown").lower()).strip("_.-")[:100] or "unknown"


def _label_quality(source: Any) -> tuple[str, float]:
    s = LEDGER_SOURCES.get(str(source), source)
    if s in CONFIRMED:
        return "keep", 1.0
    return "silver", 0.5


def _decision_point(record: StoreRecord) -> Optional[str]:
    point = record.body.get("point")
    return point if point in POINTS else None


def place(model_type: str, record: StoreRecord) -> Optional[dict[str, Any]]:
    """A rule curator's placement of one record for one model type (None: it does not belong there)."""
    b, kind, t = record.body, record.kind, record.type
    broken = [f for f in record.flags if f.startswith(("missing:", "untimed:"))]
    out = {"model_type": model_type, "quality": "keep", "weight": 1.0, "reason": ""}
    if model_type in ("m1_students", "m2_workflow_decisions", "m3_ml_decisions") and kind == "decisions" and t in ("decision", "label"):
        point = _decision_point(record)
        if point is None:
            return None
        technical = POINTS[point].kind == "technical"
        if model_type == "m3_ml_decisions" and not technical or model_type == "m2_workflow_decisions" and technical:
            return None
        if model_type == "m3_ml_decisions":
            folder = f"{_code((b.get('state') or {}).get('task') or (b.get('state') or {}).get('kind') or 'general')}/{_code(point)}"
        elif model_type == "m2_workflow_decisions":
            system = "internal" if point in INTERNAL_POINTS else "dclab" if POINTS[point].main_product else "rnd"
            folder = f"{system}/{_code(point)}"
        else:
            folder = _code(point)
        if t == "label":
            out["quality"], out["weight"] = _label_quality(b.get("source"))
            out["reason"] = "a confirmed label" if out["quality"] == "keep" else "a teacher's or a rule's label: silver, apart"
        else:
            out["reason"] = "the decision's state and answers (its label comes apart)"
        return {**out, "folder": folder, **({"quality": "out", "reason": f"broken: {', '.join(broken)}"} if broken else {})}
    if model_type == "m4_workflow_llm" and kind in ("conversations", "model_requests", "workflows") and t in ("turn", "intern_session", "request", "walk", "mcp_call"):
        surface = b.get("surface") or ("intern" if t == "intern_session" else "mcp" if t == "mcp_call" else "home" if t == "turn" else
                                       "report" if t == "request" else "walk")
        stage = M4_STAGES.get(str(surface), "report" if t == "request" else "plan")
        return {**out, "folder": stage, "reason": f"a {t} of the {surface}"}
    if model_type == "m5_solver_llm" and kind in ("agents", "conversations", "workflows") and t in ("transcript_line", "intern_session", "walk"):
        problem = "development" if kind == "agents" else _code(b.get("kind") or b.get("problem_kind") or "studio")
        return {**out, "folder": f"trajectories/{problem}", "reason": "a step of an agent solving a problem"}
    if model_type == "m6_stepwise_decisions" and kind in ("workflows", "tasks") and t in ("walk", "graph_move", "job_state"):
        process = {"walk": "walk", "graph_move": "graph", "job_state": "job"}[t]
        return {**out, "folder": process, "reason": f"a step of the {process}"}
    if model_type == "m7_judges" and (kind in ("fixes", "tests") or (kind == "decisions" and t == "correction") or
                                      (kind == "decisions" and _decision_point(record) == "workflow.g4")):
        judged = "code" if kind in ("fixes", "tests") else "correction" if t == "correction" else "result"
        evidence = "test" if kind in ("fixes", "tests") else "person" if t == "correction" else "measured"
        return {**out, "folder": judged, "reason": f"a {judged} judged by a {evidence}"}
    if model_type == "m8_retrieval" and kind in ("actions", "model_requests") and \
            str(b.get("tool") or b.get("purpose") or "") in ("find_examples", "search_evidence", "evidence_answer", "find_pitfalls"):
        return {**out, "folder": "evidence", "reason": "a search, what it offered and what was used"}
    return None


def librarian(record: StoreRecord) -> list[str]:
    """The model types a record could teach: its capture rule's feeds, else every model type (each curator then says no)."""
    from dclab_core import store_rules

    rule = store_rules.rule_for(record.kind, record.type)
    return list(rule.feeds) if rule and rule.feeds else list(MODEL_TYPES)


def _changed_files(layout: Layout) -> Iterator[tuple[str, Any, str]]:
    state_path = layout.curation_state()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    for kind in sorted(KINDS):
        base = layout.records_of(kind)
        for path in sorted(base.rglob("*.jsonl.gz")) if base.exists() else []:
            rel = layout.relative(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if state.get(rel) != digest:
                yield rel, path, digest


def curate(layout: Optional[Layout] = None, run: Optional[str] = None, model: Optional[Callable[[list[StoreRecord]], list[dict[str, Any]]]] = None,
           sample: int = 50) -> dict[str, Any]:
    """Every new record placed by the rule curators; with ``model``, a sample of them also answered in shadow."""
    layout = layout or Layout()
    store = Store(layout)
    run = run or f"curation-{now().replace(':', '').replace('-', '')}"
    decisions: list[str] = []
    quarantine: dict[str, list[str]] = defaultdict(list)
    counts: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    state_path = layout.curation_state()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    shadow_pool: list[StoreRecord] = []
    for rel, path, digest in list(_changed_files(layout)):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                record = StoreRecord.read(line)
                body_hash = hashlib.sha256(json.dumps(record.body, sort_keys=True, default=str).encode()).hexdigest()
                for model_type in librarian(record):
                    proposal = place(model_type, record)
                    if proposal is None:
                        continue
                    if (model_type, body_hash) in seen:
                        proposal = {**proposal, "quality": "out", "reason": "a duplicate of a record already placed"}
                    seen.add((model_type, body_hash))
                    decision = {"id": f"{run}:{record.id}:{model_type}", "record": record.id, "curator": f"rule:{model_type}", "at": now(), **proposal}
                    made, why = write.check(layout, decision, record)
                    if made is None:
                        continue
                    decisions.append(made.model_dump_json() + "\n")
                    counts[(model_type, made.quality if made.used else "refused")] += 1
                    if made.quality == "out" or not made.used:
                        quarantine[_code(why or made.reason.split(":")[0])].append(json.dumps({"record": record.id, "model_type": model_type, "why": why or made.reason}) + "\n")
                if model is not None and len(shadow_pool) < sample:
                    shadow_pool.append(record)
        state[rel] = digest
    shadow = 0
    if model is not None and shadow_pool:
        try:
            answers = model(shadow_pool)
        except Exception as exc:  # noqa: BLE001 - a model that fails leaves the rules' choices as they are
            answers = [{"error": f"{type(exc).__name__}: {exc}"}]
        by_id = {r.id: r for r in shadow_pool}
        for a in answers:
            record = by_id.get(str(a.get("record"))) if isinstance(a, dict) else None
            if record is None:
                continue
            decision = {"id": f"{run}:{record.id}:{a.get('model_type')}:shadow", "record": record.id, "curator": "model:store_curator", "at": now(),
                        "used": False, **{k: a.get(k) for k in ("model_type", "folder", "quality", "weight", "reason") if a.get(k) is not None}}
            made, why = write.check(layout, decision, record)
            line = (made.model_copy(update={"used": False}).model_dump_json() if made is not None else
                    json.dumps({"id": decision["id"], "record": record.id, "curator": "model:store_curator", "refused": why, "answer": a}))
            decisions.append(line + "\n")
            shadow += 1
    if decisions:
        store.keep("curation", "rnd", f"curation/{run}.jsonl", "".join(decisions).encode(), consent="train",
                   note="every curator's choice of this run; a model's answers in shadow (used false)")
    for reason, lines in quarantine.items():
        target = layout.quarantine_file(reason, run)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines), encoding="utf-8")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    return {"run": run, "decisions": len(decisions), "shadow": shadow, "by_model": {f"{m}:{q}": n for (m, q), n in sorted(counts.items())},
            "quarantined": {r: len(v) for r, v in quarantine.items()}}


__all__ = ["place", "librarian", "curate", "write"]
