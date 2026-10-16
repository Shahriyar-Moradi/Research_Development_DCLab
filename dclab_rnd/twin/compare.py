"""The two lanes side by side: what each decided, where they differ, what each cost.

``compare(workflow, ai_first)`` reads the two summaries (``Lane.summary``) and returns one dict: the solution's
fields that differ, each stage's choice on both sides, the holdout result, the moves the validator refused, the tool
calls, the model usage and the time. ``markdown`` renders it as the run's ``COMPARE.md``. ``write_steps`` (25.2) writes the same comparison as
``dclab_core.twin`` steps (the solution's columns, the split and the metric, the recipe, the family), each judged
against the truth when the seed plants one (``planted``: a studied sample's target, the columns its R&D solution
blocks, its identifiers), so the matrix of 25.1 can read many runs. ``compare`` itself never judges.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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
    """What a lane finally did at a ledger point (``point:subject``), read from its summary (its saved solution and stages)."""
    point, _, subject = key.partition(":")
    sol = summary.get("solution") or {}
    stages = summary.get("stages") or {}
    if point == "task.framing":
        return sol.get("task")
    if point == "target.column":
        return sol.get("target")
    if point == "metric.primary":
        return sol.get("metric")
    if point == "column.is_identifier":
        return subject in (sol.get("identifiers") or []) if sol else None
    if point == "column.time":
        return sol.get("time_column") == subject if sol else None
    if point == "column.group":
        return sol.get("group_column") == subject if sol else None
    if point == "feature.leakage_suspect":
        lk = stages.get("leakage") or {}
        if not lk and not sol:
            return None
        return subject in set(lk.get("declared") or []) | set(lk.get("flagged") or []) | set(sol.get("forbidden") or [])
    if point == "model.family":
        return (stages.get("models") or {}).get("chosen")
    if point == "split.strategy":
        return summary.get("split_used")  # the split the data stage made, not the proposal's
    return None


def _same(a: Any, b: Any, key: str = "") -> bool | None:
    if a is None or b is None:
        return None
    if key.startswith("model.family"):  # the rule screens a baseline beside the families; naming it or not is no disagreement
        from .design import BASELINES

        def families(v: Any) -> set[str]:
            return {str(x).lower() for x in (v if isinstance(v, list) else [v]) if str(x).lower() not in BASELINES}

        fa, fb = families(a), families(b)
        return fa == fb if isinstance(a, list) and isinstance(b, list) else bool(fa & fb)
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, list) or isinstance(b, list):
        la, lb = a if isinstance(a, list) else [a], b if isinstance(b, list) else [b]
        return {str(x).lower() for x in la} == {str(x).lower() for x in lb} or (isinstance(a, list) and not isinstance(b, list) and str(b).lower() in {str(x).lower() for x in la})
    return str(a).strip().lower() == str(b).strip().lower()


def design_rows(workflow: dict[str, Any], ai_first: dict[str, Any]) -> dict[str, Any] | None:
    """How Claude, as the expert, designed the solution (A6.7, 25.3) beside how the current system designs it (the
    rule's answers on the lane's ledger, the standard plan) and what each lane finally did."""
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
    if c.get("judged"):
        lines += ["", "## Against the planted truth (twin steps, steps.jsonl)", "", "| step | who was right | fields that differ |", "|---|---|---|"]
        for v in c["judged"]:
            lines.append(f"| {v['step']} | {v['winner']} | {', '.join(v['differs']) or '—'} |")
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
    if c.get("expert_judged"):
        lines += ["## The expert record against the current system (expert_steps.jsonl)", "", "| step | who was right | fields that differ |", "|---|---|---|"]
        lines += [f"| {v['step']} | {v['winner']} | {', '.join(v['differs']) or '—'} |" for v in c["expert_judged"]] + [""]
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
    lines = ["## How Claude, as the expert, designed it (ai_first/design/DESIGN.md)", "",
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
        lines.append(f"- v{ch['version']:02d} after {ch['checkpoint']}: {ch.get('field')} {_cell(ch.get('before'))} → {_cell(ch.get('after'))}, because {ch.get('because') or ''}")
    return lines + [""]


# ------------------------------------------------------------------------------------------------- twin steps (25.2)

def planted(kind: str, ref: str, lane: Any = None) -> dict[str, Any] | None:
    """The truth a seed plants: for a studied sample, its target and the columns its R&D solution blocks as leaks (read
    from the sample). Only a ``--blind`` run is judged by it: otherwise both lanes were given it. None for a project or a draft."""
    if kind != "sample":
        return None
    try:
        from ..studio import data as sd

        _frame, suggestion = sd.load_sample(ref)
    except Exception:  # noqa: BLE001 — a sample that cannot be read plants nothing; the run still compares
        return None
    # a column the solution reads as the time or a group is no leak to forbid (the standard plan drops it the same way); identifiers are
    # not judged: the R&D's list is a floor (the proposal adds the ones it detects), so a set compare would fault a right answer
    roles = {suggestion.get("time_column"), suggestion.get("group_column")} - {None}
    return {"solution.columns": {"target": suggestion.get("target"),
                                 "leakage": sorted(f["column"] for f in suggestion.get("forbidden") or [] if f["column"] not in roles)}}


def _arm(name: str, summary: dict[str, Any], step: str) -> Any:
    from dclab_core.twin import ArmAnswer

    answerer = "rules (standard plan)" if name == "system" else str((summary.get("session") or {}).get("model") or "claude")
    status = "answered" if summary.get("status") in ("completed", "stopped") else "failed" if summary.get("status") == "failed" else "abstained"
    sol, stages = summary.get("solution") or {}, summary.get("stages") or {}
    value: dict[str, Any]
    if step == "solution.columns":
        value = {"target": sol.get("target"), "task": sol.get("task"), "identifiers": sol.get("identifiers"), "leakage": sol.get("forbidden"),
                 "time": [sol["time_column"]] if sol.get("time_column") else [], "group": [sol["group_column"]] if sol.get("group_column") else []} if sol else {}
    elif step == "stage.data":
        value = {"split": summary.get("split_used"), "metric": sol.get("metric")}
    elif step == "stage.features":
        value = {"recipe": (stages.get("features") or {}).get("chosen")}
    elif step == "stage.models":
        value = {"family": (stages.get("models") or {}).get("chosen")}
    else:
        value = {}
    value = {k: v for k, v in value.items() if v is not None}
    if not value and status == "answered":
        status = "abstained"
    model = summary.get("model") or {}
    return ArmAnswer(arm=name, answerer=answerer[:120], status=status, value=value, input_fingerprint=lane_input(summary, step),
                     note=summary.get("error") if summary.get("status") == "stopped" else None,
                     cost_eur=float(model.get("eur") or 0.0) if step == "solution.columns" else 0.0,  # the lane's whole spend, counted once
                     input_tokens=int(model.get("input_tokens") or 0) if step == "solution.columns" else 0,
                     output_tokens=int(model.get("output_tokens") or 0) if step == "solution.columns" else 0)


TWIN_STEPS = ("solution.columns", "stage.data", "stage.features", "stage.models")


def lane_input(summary: dict[str, Any], step: str) -> str:
    """The fingerprint of what a lane was given for a step: its table (the SHA-256 recorded at the upload), whether it
    was given the known solution, and the step. Two lanes seeded alike give the same; ``twin.step`` refuses two."""
    from dclab_core import twin

    return twin.fingerprint({"table": summary.get("data_sha256"), "given_solution": summary.get("given_solution"), "step": step})


def steps(run_id: str, workflow: dict[str, Any], ai_first: dict[str, Any], truth: dict[str, Any] | None = None) -> list[Any]:
    """The run as twin steps: one per step both lanes reached; the input is the run's seed (the same for both)."""
    from dclab_core import twin

    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for name in TWIN_STEPS:
        system, ai = _arm("system", workflow, name), _arm("ai_first", ai_first, name)
        if not system.value and not ai.value:
            continue
        planted_value = (truth or {}).get(name)
        given = {"table": workflow.get("data_sha256"), "given_solution": workflow.get("given_solution"), "step": name}
        out.append(twin.step(f"{run_id}:{name}", run_id, name, at, given, system, ai,
                             input_summary={"run": run_id},
                             truth=twin.Truth(value=planted_value, source="planted", note="the studied sample's R&D solution") if planted_value else None))
    return out


def write_steps(path: Path, run_id: str, workflow: dict[str, Any], ai_first: dict[str, Any], truth: dict[str, Any] | None = None) -> list[Any]:
    records = steps(run_id, workflow, ai_first, truth)
    with Path(path).open("a", encoding="utf-8") as out:
        for r in records:
            out.write(r.model_dump_json() + "\n")
    return records


def verdicts(records: list[Any]) -> list[dict[str, Any]]:
    return [{"step": r.step, "winner": r.verdict.winner, "system": r.verdict.system, "ai_first": r.verdict.ai_first,
             "differs": [d.field for d in r.differences if d.kind not in ("same", "neither")]} for r in records]


def system_values(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The workflow lane's answers in the shapes of the twin steps the expert record is compared on."""
    sol, stages = summary.get("solution") or {}, summary.get("stages") or {}
    the_map: dict[str, Any] = {}
    try:  # the rules' reading of the same words into a problem map (24.1), what the project lane would be built from
        from dclab_core import problem_map

        m = problem_map.decompose(str(summary.get("goal") or ""))
        the_map = {"size": m.size, "families": sorted({sp.family for sp in m.sub_problems if sp.family})}
    except Exception:  # noqa: BLE001
        the_map = {}
    out = {
        "solution.columns": {"target": sol.get("target"), "task": sol.get("task"), "identifiers": sol.get("identifiers"), "leakage": sol.get("forbidden"),
                             "time": [sol["time_column"]] if sol.get("time_column") else [], "group": [sol["group_column"]] if sol.get("group_column") else []} if sol else {},
        "stage.data": {"split": summary.get("split_used"), "metric": sol.get("metric")},
        "stage.models": {"family": (stages.get("models") or {}).get("chosen")},
        "home.understanding": {"target_column": sol.get("target")} if sol else {},
        "home.map": the_map,
    }
    return {k: {f: v for f, v in d.items() if v is not None} for k, d in out.items()}


def write_expert_steps(path: Path, run_id: str, design_dir: Path, workflow: dict[str, Any], truth: dict[str, Any] | None = None) -> list[Any]:
    """Each version of the expert record against the current system's answers, as twin steps (``expert_steps.jsonl``),
    judged where the seed plants a truth. The input is the run's seed for every version: the system answered it once."""
    from dclab_core import expert

    rows = Path(design_dir) / "expert.jsonl"
    if not rows.is_file():
        return []
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    lines = rows.read_text(encoding="utf-8").splitlines()
    # only the first version answered what the system answered (the seed); later ones read the AI-first lane's own stage
    # evidence: they are revisions, kept in design/, never twin steps against the system
    for line in lines[:1]:
        trace = expert.ExpertTrace.model_validate(json.loads(line))
        out += expert.twin_steps(trace, system_values(workflow), session=run_id, at=at,
                                 input={"table": workflow.get("data_sha256"), "given_solution": workflow.get("given_solution")}, answerer=f"ai_first@expert:{trace.model or 'claude'}",
                                 system_answerer="rules (standard plan)", truth=truth)
    Path(path).unlink(missing_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        for r in out:
            f.write(r.model_dump_json() + "\n")
    return out
