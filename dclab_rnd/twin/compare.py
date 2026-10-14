"""The two lanes side by side: what each decided, where they differ, what each cost.

``compare(workflow, ai_first)`` reads the two summaries (``Lane.summary``) and returns one dict: the solution's
fields that differ, each stage's choice on both sides, the holdout result, the moves the validator refused, the tool
calls, the model usage and the time. ``markdown`` renders it as the run's ``COMPARE.md``. Nothing here judges which
lane was right: the sealed holdout and the planted traps do that, and a reader does the rest.
"""

from __future__ import annotations

import json
from typing import Any


def _solution_diff(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    a, b = a or {}, b or {}
    keys = sorted(set(a) | set(b))
    rows = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, list) or isinstance(vb, list):
            sa, sb = set(va or []), set(vb or [])
            rows[k] = {"workflow": sorted(sa), "ai_first": sorted(sb), "same": sa == sb, "only_workflow": sorted(sa - sb), "only_ai_first": sorted(sb - sa)}
        else:
            rows[k] = {"workflow": va, "ai_first": vb, "same": va == vb}
    return rows


def _stage_rows(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for stage in ("data", "leakage", "features", "models", "final"):
        sa, sb = (a.get("stages") or {}).get(stage) or {}, (b.get("stages") or {}).get(stage) or {}
        row: dict[str, Any] = {"stage": stage, "workflow": {"status": sa.get("status"), "chosen": sa.get("chosen"), "selected": sa.get("selected"), "overridden": sa.get("overridden")},
                               "ai_first": {"status": sb.get("status"), "chosen": sb.get("chosen"), "selected": sb.get("selected"), "overridden": sb.get("overridden")}}
        if stage == "leakage":
            row["workflow"]["declared"], row["ai_first"]["declared"] = sa.get("declared"), sb.get("declared")
            row["workflow"]["flagged"], row["ai_first"]["flagged"] = sa.get("flagged"), sb.get("flagged")
        if stage == "final":
            for side, s in (("workflow", sa), ("ai_first", sb)):
                row[side].update({k: s.get(k) for k in ("primary_metric", "holdout", "holdout_ci", "cv_mean", "cv_to_holdout_gap", "holdout_uses")})
        row["same_choice"] = (sa.get("chosen") == sb.get("chosen")) if sa and sb else None
        out.append(row)
    return out


def _did(summary: dict[str, Any], key: str) -> Any:
    """What a lane finally did on a decision point, read from its summary (its saved solution and stages)."""
    point, _, subject = key.partition(":")
    sol = summary.get("solution") or {}
    stages = summary.get("stages") or {}
    if point == "contract.task":
        return sol.get("task")
    if point == "contract.target_column":
        return sol.get("target")
    if point == "contract.prediction_moment":
        return "stated" if sol.get("prediction_moment") else None
    if point == "metric.primary":
        return sol.get("metric")
    if point == "column.forbidden":
        return "exclude" if subject in (sol.get("forbidden") or []) else "clear" if sol else None
    if point == "column.identifier":
        return "identifier" if subject in (sol.get("identifiers") or []) else "feature" if sol else None
    if point == "column.leakage":
        lk = stages.get("leakage") or {}
        if not lk:
            return None
        return "exclude" if subject in (lk.get("declared") or []) else "review_flag" if subject in (lk.get("flagged") or []) else "clear"
    if point == "model.families":
        return (stages.get("models") or {}).get("chosen")
    if point == "split.strategy":
        return summary.get("split_used")  # the split the data stage made, not the proposal's
    return None


def _same(a: Any, b: Any, key: str = "") -> bool | None:
    if a is None or b is None:
        return None
    if key.startswith("model.families"):  # the rule screens a baseline beside the families; naming it or not is no disagreement
        from .design import BASELINES

        def families(v: Any) -> set[str]:
            return {str(x).lower() for x in (v if isinstance(v, list) else [v]) if str(x).lower() not in BASELINES}

        fa, fb = families(a), families(b)
        return fa == fb if isinstance(a, list) and isinstance(b, list) else bool(fa & fb)
    if isinstance(a, list) or isinstance(b, list):
        la, lb = a if isinstance(a, list) else [a], b if isinstance(b, list) else [b]
        return {str(x).lower() for x in la} == {str(x).lower() for x in lb} or (isinstance(a, list) and not isinstance(b, list) and str(b).lower() in {str(x).lower() for x in la})
    return str(a).strip().lower() == str(b).strip().lower()


def design_rows(workflow: dict[str, Any], ai_first: dict[str, Any]) -> dict[str, Any] | None:
    """How Claude designed the solution (A6.7) beside how the current system designs it (the rule's answers, the
    standard plan) and what each lane finally did."""
    d = ai_first.get("design")
    if not d or d.get("error"):
        return d
    first, last = d.get("intents_first") or {}, d.get("intents_last") or {}
    rules_wf = workflow.get("rule_answers") or {}
    rows = []
    for key in sorted(set(first) | set(last)):
        row = {"key": key, "claude_first": first.get(key), "claude_last": last.get(key), "rule": rules_wf.get(key),
               "did_ai_first": _did(ai_first, key), "did_workflow": _did(workflow, key)}
        row["first_agrees_with_rule"] = _same(row["claude_first"], row["rule"], key)
        row["changed"] = (key in first) != (key in last) or not _same(first.get(key), last.get(key), key)
        row["did_as_last_design"] = _same(row["claude_last"], row["did_ai_first"], key)
        rows.append(row)
    return {"versions": d.get("versions"), "first_workflow": d.get("first_workflow"), "adherence_to_first": d.get("adherence_to_first"),
            "adherence_to_last_revision": d.get("adherence_to_last_revision"), "last_revision": d.get("last_revision"), "design_seconds": d.get("design_seconds"), "changes": d.get("changes"), "retrospective": d.get("retrospective"),
            "decisions": rows, "disagreements_with_rule": [r["key"] for r in rows if r["first_agrees_with_rule"] is False],
            "workflow_lane_plan": (workflow.get("session") or {}).get("plan")}


def compare(workflow: dict[str, Any], ai_first: dict[str, Any]) -> dict[str, Any]:
    sol = _solution_diff(workflow.get("solution"), ai_first.get("solution"))
    stages = _stage_rows(workflow, ai_first)
    fa, fb = (workflow.get("stages") or {}).get("final") or {}, (ai_first.get("stages") or {}).get("final") or {}
    holdout = None
    if fa.get("holdout") is not None and fb.get("holdout") is not None and fa.get("primary_metric") == fb.get("primary_metric"):
        holdout = {"metric": fa["primary_metric"], "workflow": fa["holdout"], "ai_first": fb["holdout"], "difference_ai_first_minus_workflow": fb["holdout"] - fa["holdout"]}
    ma, mb = workflow.get("moves") or {}, ai_first.get("moves") or {}
    sa, sb = workflow.get("session") or {}, ai_first.get("session") or {}

    def seconds(x: dict[str, Any]) -> float | None:
        """The lane's own work: its window less the time Claude spent writing designs (A6.7), which the other lane has none of."""
        if not x.get("started") or not x.get("finished"):
            return None
        from datetime import datetime

        window = (datetime.fromisoformat(x["finished"]) - datetime.fromisoformat(x["started"])).total_seconds()
        return round(window - float((x.get("design") or {}).get("design_seconds") or 0.0), 1)

    out = {
        "status": {"workflow": workflow.get("status"), "ai_first": ai_first.get("status")},
        "errors": {"workflow": workflow.get("error"), "ai_first": ai_first.get("error")},
        "solution": sol,
        "solution_differences": [k for k, v in sol.items() if not v.get("same")],
        "stages": stages,
        "stage_differences": [r["stage"] for r in stages if r.get("same_choice") is False],
        "holdout": holdout,
        "moves": {"workflow": {k: ma.get(k) for k in ("total", "allowed", "refused")}, "ai_first": {k: mb.get(k) for k in ("total", "allowed", "refused")},
                  "sequence": {"workflow": ma.get("sequence"), "ai_first": mb.get("sequence")}},
        "tool_calls": {"workflow": sa.get("tools"), "ai_first": sb.get("tools"), "count": {"workflow": sa.get("steps"), "ai_first": sb.get("steps")}},
        "model": {"workflow": workflow.get("model"), "ai_first": ai_first.get("model")},
        "seconds": {"workflow": seconds(workflow), "ai_first": seconds(ai_first)},
        "design_seconds": (ai_first.get("design") or {}).get("design_seconds"),
        "captured": {"workflow": workflow.get("counts"), "ai_first": ai_first.get("counts")},
        "final_reports": {"workflow": sa.get("final"), "ai_first": sb.get("final")},
    }
    if ai_first.get("design"):
        out["design"] = design_rows(workflow, ai_first)
    if workflow.get("draft") or ai_first.get("draft"):
        da, db = workflow.get("draft") or {}, ai_first.get("draft") or {}
        out["draft"] = {"questions_asked": {"workflow": da.get("questions_asked"), "ai_first": db.get("questions_asked")},
                        "understanding": _solution_diff(da.get("understanding"), db.get("understanding")),
                        "plan": {"workflow": da.get("plan"), "ai_first": db.get("plan")},
                        "workflow_drawn": {"workflow": da.get("workflow"), "ai_first": db.get("workflow")},
                        "agent_turns": {"workflow": da.get("agent_turns"), "ai_first": db.get("agent_turns")},
                        "model_note": {"workflow": da.get("model_note"), "ai_first": db.get("model_note")}}  # a turn that fell back to the standard questions
    return out


def _cell(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, dict)):
        text = json.dumps(v, ensure_ascii=False)
        return text if len(text) <= 80 else text[:77] + "…"
    return str(v)


def markdown(c: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [f"# Twin run {manifest.get('run_id')}", "",
             f"Seed: `{manifest.get('seed')}` · task: {manifest.get('task', '')[:160]}", "",
             f"Workflow lane: {manifest.get('lanes', {}).get('workflow', {}).get('label', 'the current system')} · "
             f"AI-first lane: {manifest.get('lanes', {}).get('ai_first', {}).get('label', 'Claude')}", "",
             "| | workflow | ai_first |", "|---|---|---|",
             f"| status | {_cell(c['status']['workflow'])} | {_cell(c['status']['ai_first'])} |",
             f"| error | {_cell(c['errors']['workflow'])} | {_cell(c['errors']['ai_first'])} |",
             f"| seconds | {_cell(c['seconds']['workflow'])} | {_cell(c['seconds']['ai_first'])} |",
             f"| tool calls | {_cell(c['tool_calls']['count']['workflow'])} | {_cell(c['tool_calls']['count']['ai_first'])} |",
             f"| moves (allowed / refused) | {_cell((c['moves']['workflow'] or {}).get('allowed'))} / {_cell((c['moves']['workflow'] or {}).get('refused'))} | "
             f"{_cell((c['moves']['ai_first'] or {}).get('allowed'))} / {_cell((c['moves']['ai_first'] or {}).get('refused'))} |",
             f"| model requests | {_cell((c['model']['workflow'] or {}).get('requests'))} | {_cell((c['model']['ai_first'] or {}).get('requests'))} |",
             f"| model euros | {_cell((c['model']['workflow'] or {}).get('eur'))} | {_cell((c['model']['ai_first'] or {}).get('eur'))} |",
             f"| requests refused by a cap | {_cell((c['model']['workflow'] or {}).get('refused_in_transcript'))} | {_cell((c['model']['ai_first'] or {}).get('refused_in_transcript'))} |", ""]
    if c.get("holdout"):
        h = c["holdout"]
        lines += [f"**Holdout {h['metric']}**: workflow {h['workflow']:.4f} · ai_first {h['ai_first']:.4f} · difference {h['difference_ai_first_minus_workflow']:+.4f} "
                  "(one holdout each, scored once; a difference inside the intervals is noise)", ""]
    lines += ["## Solution", "", "| field | workflow | ai_first | same |", "|---|---|---|---|"]
    for k, v in c["solution"].items():
        lines.append(f"| {k} | {_cell(v.get('workflow'))} | {_cell(v.get('ai_first'))} | {'yes' if v.get('same') else '**no**'} |")
    lines += ["", "## Stages", "", "| stage | workflow | ai_first | same choice |", "|---|---|---|---|"]
    for r in c["stages"]:
        w, a = r["workflow"], r["ai_first"]
        lines.append(f"| {r['stage']} | {_cell(w.get('chosen'))} ({_cell(w.get('status'))}) | {_cell(a.get('chosen'))} ({_cell(a.get('status'))}) | "
                     f"{'—' if r.get('same_choice') is None else 'yes' if r['same_choice'] else '**no**'} |")
    lk = next((r for r in c["stages"] if r["stage"] == "leakage"), {})
    if lk:
        lines += ["", f"Leakage: declared workflow {_cell(lk['workflow'].get('declared'))} · ai_first {_cell(lk['ai_first'].get('declared'))}; "
                  f"flagged workflow {_cell(lk['workflow'].get('flagged'))} · ai_first {_cell(lk['ai_first'].get('flagged'))}"]
    lines += ["", "## Moves", "", f"workflow: `{' → '.join(c['moves']['sequence'].get('workflow') or [])}`", "",
              f"ai_first: `{' → '.join(c['moves']['sequence'].get('ai_first') or [])}`", "",
              "## Tool calls", "", f"workflow: `{' → '.join(c['tool_calls'].get('workflow') or [])}`", "",
              f"ai_first: `{' → '.join(c['tool_calls'].get('ai_first') or [])}`", ""]
    if c.get("design") and not c["design"].get("error"):
        lines += design_markdown(c["design"])
    if c.get("draft"):
        d = c["draft"]
        lines += ["## Draft chat", "", f"questions asked: workflow {_cell(d['questions_asked']['workflow'])} · ai_first {_cell(d['questions_asked']['ai_first'])}",
                  f"agent turns: workflow {_cell(d['agent_turns']['workflow'])} · ai_first {_cell(d['agent_turns']['ai_first'])}", ""]
        if d.get("model_note", {}).get("ai_first"):
            lines += [f"**The AI-first lane fell back to the standard questions at least once:** {d['model_note']['ai_first']}", ""]
    lines += ["## Final reports", "", "### workflow", "", c["final_reports"].get("workflow") or "—", "", "### ai_first", "", c["final_reports"].get("ai_first") or "—", "",
              "## Captured", "", f"workflow: {json.dumps(c['captured']['workflow'])}", "", f"ai_first: {json.dumps(c['captured']['ai_first'])}", "",
              "Every number above is read from the lanes' own files; see README.md in the run folder for what each file holds."]
    return "\n".join(lines) + "\n"


def design_markdown(d: dict[str, Any]) -> list[str]:
    a, b = d.get("adherence_to_first") or {}, d.get("adherence_to_last_revision") or {}
    lines = ["## How Claude designed it (ai_first/design/DESIGN.md)", "",
             "Versions: " + ", ".join(f"v{v['version']:02d} {v['checkpoint']}{'' if v['parsed'] else ' (unreadable)'} ({v['changes']} changes)" for v in d.get("versions") or []), "",
             "First workflow: `" + " → ".join(f"{s.get('id')} {s.get('tool')}{':' + s['stage'] if s.get('stage') else ''}" for s in d.get("first_workflow") or []) + "`", "",
             f"Plan against act (first design): {a.get('done')} of {a.get('planned_steps')} steps done, {len(a.get('in_order') or [])} in order, "
             f"skipped {_cell(a.get('skipped'))}, not in the design {_cell(a.get('unplanned'))}, failed or refused {_cell(a.get('failed_or_refused'))}; "
             f"against the last revision before the end (v{d.get('last_revision') or 0:02d}): {b.get('done')} of {b.get('planned_steps')}. "
             f"Claude spent {_cell(d.get('design_seconds'))} s writing designs (left out of the lane's seconds above).", ""]
    if d.get("workflow_lane_plan"):
        lines += ["The current system's plan (the workflow lane): " + str(d["workflow_lane_plan"]).replace("\n", " · "), ""]
    rows = d.get("decisions") or []
    if rows:
        lines += ["| decision | Claude first | Claude last | the rule | ai_first did | workflow did |", "|---|---|---|---|---|---|"]
        for r in rows:
            mark = " **≠ rule**" if r.get("first_agrees_with_rule") is False else ""
            lines.append(f"| {r['key']} | {_cell(r['claude_first'])}{mark} | {_cell(r['claude_last'])}{' (changed)' if r.get('changed') else ''} | {_cell(r['rule'])} | "
                         f"{_cell(r['did_ai_first'])} | {_cell(r['did_workflow'])} |")
        lines.append("")
    for ch in d.get("changes") or []:
        lines.append(f"- v{ch['version']:02d} after {ch['checkpoint']}: {ch.get('field')} {_cell(ch.get('from'))} → {_cell(ch.get('to'))}, because {ch.get('because', '')}")
    return lines + [""]
