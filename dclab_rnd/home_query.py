"""Answering questions from a draft's own table (package H6 of the Home plan): the studio's side of dclab_core.query
(dclab_rnd/analysis.py is another thing: the research registry's cross-dataset evidence).

``table`` reads the draft's cleaned table only when its SHA-256 matches the one the pipeline recorded; ``write_plan`` asks
the ``home_query`` purpose for a plan from column summaries alone (never a row or a cell value) and keeps it only when
it reads as a plan; ``run`` executes a plan with the row floor (DCLAB_ROW_FLOOR, default 5) and appends what it related
to the draft's exposure log. The answer's words (H7) come later; nothing here shows a row.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from dclab_core import query as Q
from dclab_rnd import prompts

PROMPT = "home_query"
SUMMARY = ("name", "kind", "missing_rate", "unique")  # what the model sees of each column: never a value


def floor() -> int:
    try:
        return max(1, int(os.environ.get("DCLAB_ROW_FLOOR", Q.FLOOR)))
    except ValueError:
        return Q.FLOOR


def table(store: Any, draft: dict[str, Any]) -> pd.DataFrame:
    """The draft's cleaned table, checked against the hash the pipeline recorded (a changed file is refused)."""
    from .audit import sha256_of

    asset = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset") and a.get("status") == "ready"), None)
    if asset is None:
        raise ValueError("there is no prepared table yet: upload or connect one")
    path = Path(store.data_dir(draft["id"])) / str(asset.get("clean_file") or "clean.parquet")
    found = sha256_of(path)
    if asset.get("clean_sha256") and found != asset["clean_sha256"]:
        raise ValueError("the prepared table changed since it was checked; prepare it again")
    if not asset.get("clean_sha256") and found:  # a table prepared before H6 (or a version's rewrite): its hash from now on
        store.update(draft["id"], lambda d: [a.update(clean_sha256=found) for a in d.get("assets") or [] if a.get("id") == asset["id"]])
    return pd.read_parquet(path)


def summaries(draft: dict[str, Any]) -> list[dict[str, Any]]:
    return [{k: c.get(k) for k in SUMMARY} for c in ((draft.get("analysis") or {}).get("columns") or []) if c.get("name")]


def write_plan(client: Any, question: str, draft: dict[str, Any], brief: Optional[dict[str, Any]] = None) -> tuple[Optional[Q.QueryPlan], str]:
    """The model's plan for the question, or None with the reason (no model, a refusal, an unreadable answer)."""
    if client is None:
        return None, "no model serves home_query here"
    user = json.dumps({"question": str(question)[:2000], "brief": {k: (v or {}).get("value") for k, v in (brief or {}).items()},
                       "columns": summaries(draft)}, ensure_ascii=False)
    try:
        reply = client.complete([{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": user}], None)
    except Exception as error:  # noqa: BLE001 - no plan, said why
        return None, f"the model did not answer ({type(error).__name__})"
    text = reply.get("content") if isinstance(reply, dict) else str(reply)
    match = re.search(r"\{.*\}", str(text or ""), re.S)
    try:
        answer = json.loads(match.group(0)) if match else {}
    except ValueError:
        answer = {}
    if answer.get("refuse"):
        return None, str(answer["refuse"])[:300]
    if not answer.get("measures"):
        return None, "the answer was not a plan (no measure)"
    try:
        return Q.QueryPlan.model_validate({k: v for k, v in answer.items() if k in Q.QueryPlan.model_fields and k != "schema_version"}), ""
    except Exception as error:  # noqa: BLE001 - an answer that is not a plan
        return None, f"the answer was not a plan ({str(error).splitlines()[0][:160]})"


def run(store: Any, draft_id: str, plan: Q.QueryPlan) -> Q.QueryResult:
    """Execute a plan on the draft's checked table (refused plans raise ValueError with the reasons) and log what it related."""
    from .draft.store import now

    draft = store.get(draft_id)
    result = Q.execute(plan, table(store, draft), floor=floor())
    entry = {**Q.exposure(plan), "at": now()}
    store.update(draft_id, lambda d: d.setdefault("exposure", []).append(entry))
    return result


__all__ = ["floor", "table", "summaries", "write_plan", "run"]
