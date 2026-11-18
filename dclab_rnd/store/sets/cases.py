"""Finished cases as training examples (package 37.17, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): each case the lead agent
finished, sorted into the workflow model's set ``m4_workflow_llm/case_files`` with its source, never edited.

An example is the conversation in (the person's words and the agent's, in order) and the case file out (goal, decision,
user, data, what it cannot answer, assumptions, unknowns, the route, each fact with its source), so a model learns to
understand a person the way the lead agent's record says it did. Each carries where it came from: ``simulated`` (a
conversation exam's written person, at half weight, apart) or ``real`` (a person's draft). The case file is built by code
from the case's own versions, so no reasoning is claimed for it (``no_reasoning`` says so). A model-run curator that
judges quality (32.9) can read these; this builder spends nothing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from dclab_core.datasets import split_of
from dclab_core.store import ChatExample, Message

FOLDER = "case_files"
INSTRUCTION = ("Read the conversation between a person and DCLab's lead agent. Write the case file it reached: the goal, the "
               "decision it serves, who uses the result, the data, what the data cannot answer, the assumptions, what is still "
               "unknown and the route, each fact with its source (said, computed or assumed).")
NO_REASONING = "the case file is built by code from the case's own versions: a record, not a model's reasoning"
EXAM_ID = re.compile(r"cex-(\d+)", re.IGNORECASE)  # the exam's folder or workspace names it: CEX-012, cex-012


def _view(case: dict[str, Any]) -> dict[str, Any]:
    keep = ("goal", "decision", "user", "data", "cannot_answer", "constraints", "assumptions", "unknowns", "route", "chosen", "findings", "build")
    return {k: case.get(k) for k in keep if case.get(k) not in (None, [], "")}


def example(rid: str, transcript: Iterable[dict[str, Any]], case: dict[str, Any], provenance: Any, origin: str,
            split_key: str, writer: Optional[dict[str, Any]] = None) -> Optional[ChatExample]:
    lines = [f"{'Person' if t.get('role') in ('person', 'user') else 'DCLab'}: {str(t.get('text') or t.get('content') or '').strip()}"
             for t in transcript if str(t.get("text") or t.get("content") or "").strip()]
    view = _view(case)
    if len(lines) < 2 or not view:
        return None
    try:
        return ChatExample(id=f"m4_workflow_llm:case:{rid}"[:200], model_type="m4_workflow_llm", split=split_of(split_key), provenance=provenance,
                           task="case_file", stage=FOLDER, weight=0.5 if origin == "simulated" else 1.0,
                           messages=[Message(role="system", content=INSTRUCTION), Message(role="user", content="\n".join(lines)),
                                     Message(role="assistant", content=json.dumps(view, ensure_ascii=False), no_reasoning=NO_REASONING)],
                           check={"origin": origin, "case_version": case.get("version"), **(writer or {})})
    except Exception:  # noqa: BLE001 - a case the contract refuses never enters the set
        return None


def exam_writers(models: dict[tuple[str, str], dict[str, dict[str, str]]]) -> dict[str, dict[str, dict[str, str]]]:
    """An exam's requests by their draft name alone (``exam:CEX-012:case-d-01`` for the agent, ``exam:CEX-012:person:case-d-01``
    for its written person): the exam names itself, so the workspace its requests were kept under does not matter."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for (_, draft), found in models.items():
        if str(draft).startswith("exam:"):
            out.setdefault(str(draft), {}).update(found)
    return out


def from_records(records: Iterable[Any], models: Optional[dict[tuple[str, str], dict[str, dict[str, str]]]] = None) -> list[ChatExample]:
    """Exam conversations (their kept conversation.json, with the case file in its notes) and drafts with a case. Each says
    who wrote it (``sets.llm.home_answerer``) from the model requests of its draft or its exam (``models``, from
    ``sets.llm.home_models``), so the training export can hold a provider's (the owner, 2026-11-18)."""
    from .llm import home_answerer  # here, not at the top: llm imports this module inside its build

    models = models or {}
    exams = exam_writers(models)
    out = []
    for r in records:
        b = r.body if isinstance(getattr(r, "body", None), dict) else {}
        source = str(getattr(r.provenance, "source", ""))
        # the store keeps an exam's file under a workspace named after its folder, its source only the file's name
        where = f"{getattr(r.provenance, 'workspace', '') or ''}/{source}"
        if source.endswith("conversation.json") and "conversation-exam" in where and (b.get("notes") or {}).get("case_file"):
            found: dict[str, dict[str, str]] = {}
            m = EXAM_ID.search(where)
            arm = str(b.get("arm") or "case")
            for key in (f"exam:CEX-{m.group(1)}:{arm}-{b.get('id')}", f"exam:CEX-{m.group(1)}:person:{arm}-{b.get('id')}") if m else ():
                found.update(exams.get(key, {}))
            # an exam's person and agent are both models: with no request found, it is an unknown model's, never the rules'
            writer = home_answerer({"agent": {"mode": "model"}}, found)
            ex = example(r.id, b.get("transcript") or [], b["notes"]["case_file"], r.provenance, "simulated", f"exam:{b.get('id')}", writer)
        elif (b.get("case_state") or {}).get("case"):
            writer = home_answerer(b, models.get((str(r.provenance.workspace), str(b.get("id") or "")), {}))
            ex = example(r.id, b["case_state"].get("turns") or [], b["case_state"]["case"], r.provenance, "real", f"draft:{b.get('id')}", writer)
        else:
            continue
        if ex is not None:
            out.append(ex)
    return out


__all__ = ["FOLDER", "example", "exam_writers", "from_records"]
