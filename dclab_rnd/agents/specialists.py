"""Specialists (package 37.7, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): sub-agents the lead agent sends when a question needs
depth, each with its own context, each returning findings into the case file, never to the person directly.

* ``data_investigator``: what the shared table can and cannot answer for this person's goal (columns, grain, gaps).
* ``leakage_auditor``: which columns would be known only after the moment of the decision, before any model is proposed.
* ``critic``: what is weak or missing in the understanding and the options before they are shown.
* ``domain_expert``: what usually matters for a problem of this kind in this person's business.
* ``solution_designer``: how the chosen work should be set up (outcome, moment, split, measure) once it has started.

Code decides when each may be sent (:func:`allowed`: its trigger in the case, and a budget of consults per case) and what
it sees (the case file, the table's view and what code computed; never a row). Its answer is a list of findings checked
here; no reason is asked in it: the specialist's thinking is kept by the gateway as its summarized thinking.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from dclab_core import case as C

from .. import prompts

PURPOSE = "case_specialist"
PROMPT = "case_specialist"
MAX_CONSULTS = 3  # per case: depth where it helps, never a team on every question
MAX_FINDINGS = 6
KINDS = ("fact", "risk", "question", "assumption")
SPECIALISTS: dict[str, str] = {
    "data_investigator": "Read the case and the table's view. Say what this table can and cannot answer for the person's goal and "
                         "decision: the grain (what one row is), the outcome if there is one, columns that are missing, and gaps.",
    "leakage_auditor": "Read the case and the table's view. Say which columns would only be known after the moment the decision "
                       "is made (they would leak the answer), and what must be asked to be sure.",
    "critic": "Read the case. Say what is weak, assumed without evidence, or missing in the understanding so far, and what a "
              "careful analyst would check before proposing options.",
    "domain_expert": "Read the case. Say what usually matters for this kind of problem in this person's kind of business: the "
                     "decision levers, the usual traps, what success is normally measured by.",
    "solution_designer": "Read the case. The work has started: say how it should be set up: the outcome, the moment of the "
                         "decision, what may be used, how to split time, and the measure that matches the decision.",
}


def virtual_table_text(virtual_tables: list[Mapping[str, Any]], prediction_moment: str = "") -> str:
    """Format a pre-data virtual schema blueprint so specialists can inspect columns, roles, and grain without physical rows."""
    if not virtual_tables:
        return "No table or virtual schema has been shared."
    lines = ["[VIRTUAL SCHEMA BLUEPRINT — NO PHYSICAL ROWS UPLOADED]"]
    if prediction_moment:
        lines.append(f"Prediction moment: {prediction_moment}")
    for vt in virtual_tables[:6]:
        if not isinstance(vt, Mapping):
            continue
        tname = str(vt.get("name") or "virtual_table")
        grain = str(vt.get("grain") or "one row per entity/event")
        ekey = str(vt.get("entity_key") or "entity_id")
        tcol = str(vt.get("time_col") or "event_timestamp")
        target = str(vt.get("target") or "target")
        must = [str(c) for c in (vt.get("must_have_columns") or []) if c]
        tier2 = [str(c) for c in (vt.get("tier2_engineered_columns") or []) if c]
        forbid = [str(c) for c in (vt.get("forbidden_leakage_columns") or []) if c]
        lines.append(
            f"- Table {tname} (grain: {grain}; entity_key: {ekey}; time_col: {tcol}; target: {target} [outcome-like name]): "
            f"Tier-1 columns: {', '.join(must) or 'none'}; "
            f"Tier-2 engineered: {', '.join(tier2) or 'none'}; "
            f"Known forbidden leakage traps: {', '.join(forbid) or 'none'}."
        )
    return "\n".join(lines)


def allowed(
    name: str,
    case: C.CaseFile,
    table: Optional[Mapping[str, Any]],
    consulted: list[str],
    *,
    virtual_columns: Optional[list[str]] = None,
    has_blueprint: bool = False,
) -> Optional[str]:
    """Why a specialist may not be sent now (None when it may): its trigger, and the case's budget of consults."""
    if name not in SPECIALISTS:
        return f"no specialist {name!r}; there are: {', '.join(SPECIALISTS)}"
    if len(consulted) >= MAX_CONSULTS:
        return f"this case has used its {MAX_CONSULTS} consults: decide with what you have"
    if name in consulted:
        return f"the {name} was already consulted on this case: read its findings in the case file"
    columns = list((table or {}).get("columns") or [])
    vcols = [str(c) for c in (virtual_columns or []) if str(c).strip()]
    if name in ("data_investigator", "leakage_auditor") and not columns and not vcols:
        return "no table has been shared: there is nothing for it to read"
    if (
        name == "leakage_auditor"
        and not vcols
        and not has_blueprint
        and not any("outcome-like name" in (c.get("noticed") or []) for c in columns if isinstance(c, Mapping))
        and not (case.route and case.route.track == "prediction")
        and not any(o.track == "prediction" for o in case.options)
    ):
        return "no prediction is in view (no outcome-like column, option or route): leakage cannot arise yet"
    if name == "solution_designer" and case.route is None and not has_blueprint:
        return "nothing has started yet: the solution designer sets up work that has begun"
    if name == "critic" and not C.readiness(case).may_propose and not has_blueprint:
        return "the critic reads an understanding ready for options: ask what is still missing first"
    if name == "domain_expert" and not C.known(case.goal) and not has_blueprint:
        return "the domain expert needs the person's goal first"
    return None


def check(value: Any) -> tuple[list[dict[str, str]], list[str]]:
    """The findings a specialist may write into the case: each a short text of a known kind, at most MAX_FINDINGS."""
    if not isinstance(value, dict) or not isinstance(value.get("findings"), list):
        return [], ["the answer is not {\"findings\": [...]}"]
    kept, problems = [], []
    for f in value["findings"][:MAX_FINDINGS]:
        if not isinstance(f, dict) or f.get("kind") not in KINDS or not str(f.get("text") or "").strip():
            problems.append(f"a finding needs a kind ({', '.join(KINDS)}) and a text: {f!r}"[:200])
            continue
        kept.append({"kind": f["kind"], "text": str(f["text"]).strip()[:600], "field": str(f.get("field") or "")[:60]})
    return kept, problems


def consult(client: Any, name: str, question: str, case_text: str, table_text: str, computed: list[str]) -> dict[str, Any]:
    """One specialist's call: its brief, the case and the table's view; returns {"findings"} or {"error"}."""
    from .person_json import first_json

    brief = SPECIALISTS[name]
    user = (f"You are DCLab's {name.replace('_', ' ')}. {brief}\n\nThe lead agent asks: {question}\n\n{case_text}\n\n{table_text}"
            + (f"\n\nWhat DCLab's code computed so far:\n" + "\n".join(computed[-6:]) if computed else ""))
    out = client.complete([{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": user}], None, max_tokens=4000)
    findings, problems = check(first_json(str(out.get("content") or "")))
    if not findings:
        return {"error": "the specialist gave no finding DCLab could use", "problems": problems}
    return {"specialist": name, "findings": findings, "dropped": problems}


__all__ = ["KINDS", "MAX_CONSULTS", "PURPOSE", "SPECIALISTS", "allowed", "check", "consult", "virtual_table_text"]
