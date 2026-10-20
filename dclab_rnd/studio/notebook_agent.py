"""The next notebook cell, written by a model for the person's request (the owner, 2026-10-20: "show accuracy plot" wrote
nothing real on the Notebook page).

The model (``notebook_cell`` purpose) sees the request, the notebook's code cells as DCLab generated them, and the
table's column names with their kinds: never a row, a cell value or an output of the kernel. Its cell is checked
before the person sees it: it must parse, and it must not touch ``X_test`` or ``y_test`` (the holdout is read once, by
the workflow). The person accepts or rejects it; only an accepted cell runs, in the project's kernel, which has the
training rows only. Every proposal and the person's answer are kept: an accept or a reject beside the proposal is a
preference the store turns into training data (``dclab_rnd.corrections``).
"""

from __future__ import annotations

import ast
import json
import re
import secrets
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from .. import prompts

HOLDOUT = re.compile(r"\b(X_test|y_test)\b")
MAX_REQUEST = 600
MAX_CELLS = 40  # cells kept on a project; refused and rejected ones go first, an accepted one never
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def lock(project_id: str) -> threading.Lock:
    """One lock per project around reading and writing its agent cells, so two requests never lose each other's change."""
    with _LOCKS_LOCK:
        return _LOCKS.setdefault(project_id, threading.Lock())


def keep(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """At most MAX_CELLS, the oldest refused or rejected dropped first, then the oldest proposals; accepted cells always stay."""
    cells = list(cells)
    for drop in (("refused", "rejected"), ("proposed",)):
        while len(cells) > MAX_CELLS:
            old = next((i for i, c in enumerate(cells) if c.get("status") in drop), None)
            if old is None:
                break
            cells.pop(old)
    return cells


def problems(code: str) -> list[str]:
    """Why a cell may not be shown to run: it does not parse, or it touches the holdout."""
    out = []
    try:
        ast.parse(code)
    except SyntaxError as exc:
        out.append(f"the cell does not parse: {exc.msg} (line {exc.lineno})")
    if HOLDOUT.search(code):
        out.append("the cell uses X_test or y_test: the holdout is read once, by the workflow; use X_train, y_train and cv")
    return out


def _context(project: dict[str, Any], records: dict[str, Any]) -> str:
    from . import export

    cells = [c for c in export.notebook(project, records)["cells"] if c["cell_type"] == "code"
             and ((c.get("metadata") or {}).get("dclab") or {}).get("role") != "holdout"]
    code = "\n\n".join(f"# cell {i + 1}\n{c['source'] if isinstance(c['source'], str) else ''.join(c['source'])}" for i, c in enumerate(cells))
    profile = ((project.get("data") or {}).get("profile") or {}).get("columns") or []
    columns = ", ".join(f"{c.get('name')} ({c.get('kind') or c.get('dtype') or '?'})" for c in profile if isinstance(c, dict))[:4000]
    return f"Columns: {columns or 'unknown'}\n\nThe notebook's code cells:\n{code}"


def _parse(text: str) -> dict[str, Any]:
    found = re.search(r"\{.*\}", text or "", re.S)
    if not found:
        raise ValueError("the model's reply was not JSON")
    doc = json.loads(found.group(0))
    if not isinstance(doc.get("code"), str) or not doc["code"].strip():
        raise ValueError("the model's reply had no code")
    return {"code": doc["code"].strip(), "explanation": str(doc.get("explanation") or "")[:600]}


def write(project: dict[str, Any], records: dict[str, Any], request: str, client: Any) -> dict[str, Any]:
    """A proposed cell for ``request``: its code, its explanation and the check's findings. One retry, told what failed."""
    request = request.strip()[:MAX_REQUEST]
    messages = [{"role": "system", "content": prompts.text("notebook_cell")},
                {"role": "user", "content": f"Request: {request}\n\n{_context(project, records)}"}]
    proposal, found = None, ["no reply"]
    for _ in range(2):
        reply = client.complete(messages, max_tokens=1800)
        text = (reply or {}).get("content") or ""
        try:
            proposal = _parse(text)
            found = problems(proposal["code"])
        except (ValueError, json.JSONDecodeError) as exc:
            proposal, found = None, [str(exc)]
        if proposal is not None and not found:
            break
        messages += [{"role": "assistant", "content": text}, {"role": "user", "content": "Fix this and answer with the JSON again: " + "; ".join(found)}]
    return {"id": "a" + secrets.token_hex(5), "request": request, "code": (proposal or {}).get("code", ""),
            "explanation": (proposal or {}).get("explanation", ""), "problems": found if proposal is None or found else [],
            "status": "proposed" if proposal is not None and not found else "refused", "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def answer(cell: dict[str, Any], accepted: bool, by: Optional[str], project_id: str) -> dict[str, Any]:
    """The person's answer to a proposal, kept as a correction (the proposal before, the accepted cell or nothing after)."""
    from .. import corrections

    cell = {**cell, "status": "accepted" if accepted else "rejected", "answered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": by}
    proposed = {"request": cell["request"], "code": cell["code"], "answer": None}
    corrections.record("notebook_cell", proposed, {**proposed, "answer": "accepted" if accepted else "rejected"},
                       reason="accepted" if accepted else "rejected", by=by or "person", project=project_id)
    return cell


__all__ = ["write", "answer", "problems", "keep", "lock", "HOLDOUT"]
