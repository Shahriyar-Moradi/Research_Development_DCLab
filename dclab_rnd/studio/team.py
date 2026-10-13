"""The multi-agent workflow (package 18.8): dclab_core's roles placed on the workflow graph's steps, as one team.

The roles already exist, each measured on its own (18.2 to 18.7). Here they work one after the other on a project, each
passed the artefacts of the steps before it, under the same rules as any move:

- WF-01, understand: the problem analyst and the brief checker read the description and the files' column names into
  a brief; the person is asked only what is still open after the brief, the label file and the data have spoken (the
  target, the prediction moment, the checker's questions), and every answer goes into the brief the later roles read;
- WF-04, analyse: the data investigator reads the columns' names and kinds against the moment, on the provisional
  training rows only (the earliest 75 % when the table has a time column);
- WF-01 and WF-03, plan: the solution planner proposes; the plan's validator checks it; the solution is saved through
  the graph validator, the kind of the problem (16.1) asks what it needs (a rare event's review budget), the feature
  engineer's programs are proposed, and the person signs the solution with the programs shown (the solution gate);
- WF-05 to WF-08: the engine's stages, run as the agent's moves, with the critic choosing the review rounds' order
  (inside this run's scope only); the person approves opening the holdout (the holdout gate);
- WF-10, report: the reporter writes summaries above the report's lines, kept only through ``cited.check``.

Every role's output is checked by code before it is used and falls back to the rules when the model abstains, fails or
is refused; every step is written to the project's log (``team_step``: the role, its step, whether the model's answer
was used, what was kept and refused, and why); every write to the project is a move the graph validator checks and
logs. The budget is the advisor's: one ``GatewayAdvisor`` whose euro cap holds for every role together; the stage notes
ask no model inside a team run (the reporter is the team's prose). A private project runs the rules only. A gate is
signed only by ``Person.approve``, which must come from a person: its ``who`` is recorded with the approval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

ESSENTIAL = ("target_column", "prediction_moment")  # what no rule or role may guess for the person
ALIASES = {"target": "target_column", "moment": "prediction_moment"}
TEXT_FIELDS = ("goal", "action", "unit", "target_column", "target_definition", "prediction_moment", "time_column", "horizon",
               "primary_metric")


@dataclass
class Person:
    """The person behind the conversation (Home or the intern), as the team sees them: ``answer(field, question)``
    returns their words or None; ``approve(gate, summary)`` signs a gate or not; ``who`` is recorded with each approval
    ("owner", a member's name, or "scripted person" in a measurement)."""

    answer: Callable[[str, str], Optional[str]]
    approve: Callable[[str, dict[str, Any]], bool]
    who: str = "owner"
    asked: list[dict[str, str]] = field(default_factory=list)

    def ask(self, field_name: str, question: str) -> Optional[str]:
        self.asked.append({"field": field_name, "question": question})
        said = self.answer(field_name, question)
        return said.strip() if isinstance(said, str) and said.strip() else None


class Stopped(Exception):
    def __init__(self, at: str, why: str):
        super().__init__(why)
        self.at, self.why = at, why


def _step(store: Any, project_id: str, steps: list, role: str, wf: str, source: str, reason: str, **extra: Any) -> None:
    entry = {"role": role, "step": wf, "source": source, "reason": str(reason)[:300], **extra}
    store.log(project_id, "team_step", entry)
    steps.append(entry)


def _write(store: Any, project_id: str, what: str, **changes: Any) -> dict[str, Any]:
    """A write to the project (settings, policy, pack, programs) as the agent's validated, logged move."""
    from . import graph

    project = store.get(project_id)
    verdict = graph.check(project, "set_settings", "agent", what=what)
    graph.log(store, project_id, verdict, project, outcome=f"done: {what}" if verdict.allowed else None)
    if not verdict.allowed:
        raise Stopped("WF-03", verdict.message)
    for key, value in changes.items():
        if key == "settings":
            project["settings"] = {**(project.get("settings") or {}), **value}
        elif key == "policy":
            project["policy"] = {**(project.get("policy") or {}), **value}
        else:
            project[key] = value
    store.save(project)
    store.log(project_id, "team_write", {"what": what, "keys": sorted(changes)})
    return project


def _column(said: Optional[str], columns: list[str]) -> Optional[str]:
    """The column the person's words name: exactly, ignoring case, or the one column named inside the words."""
    if not said:
        return None
    if said in columns:
        return said
    lowered = {c.lower(): c for c in columns}
    if said.lower().strip("` ") in lowered:
        return lowered[said.lower().strip("` ")]
    named = [c for c in columns if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c.lower())}(?![A-Za-z0-9_])", said.lower())]
    return named[0] if len(named) == 1 else None


def _share(said: Optional[str], rows: int, span_days: Optional[float]) -> Optional[float]:
    """A review budget in the person's words as a share of rows: "0.5%", ".5%", "0.005", or "70 a day" over the days
    the rows span. A bare number is a share only below 0.05 (0.5 is read as ambiguous, never as half the rows)."""
    if not said:
        return None
    from dclab_core.assess import capacity_share

    share = capacity_share(said, rows, span_days)
    if share is None:
        number = re.fullmatch(r"\s*(0?\.\d+)\s*", said)
        share = float(number.group(1)) if number and float(number.group(1)) < 0.05 else None
    return share if share is not None and 0 < share < 1 else None


def _fact(brief: Any, name: str, value: Any, quote: str) -> Any:
    """The brief with ``name`` stated by the person (their words quoted)."""
    current = getattr(brief, name)
    return brief.model_copy(update={name: type(current)(value=value, status="stated", quote=str(quote)[:600], source="person")})


def run(store: Any, project_id: str, description: str, person: Person, advisor: Any = None,
        quick: Optional[bool] = None) -> dict[str, Any]:
    """Run the team on a project that has its data (and label file, if any). Returns the steps, the questions asked,
    what the person settled, and how the run ended (``completed``, or ``stopped`` with where and why)."""
    steps: list[dict[str, Any]] = []
    settled: dict[str, str] = {}
    try:
        out = _run(store, project_id, description, person, advisor, quick, steps, settled)
    except Stopped as stop:
        _step(store, project_id, steps, "team", stop.at, "stopped", stop.why)
        return {"status": "stopped", "at": stop.at, "why": stop.why, "steps": steps, "asked": person.asked, "settled": settled}
    except Exception as exc:  # noqa: BLE001 - a stage that fails stops the run with a record, never silently
        from . import graph

        why = exc.verdict.message if isinstance(exc, graph.GraphBlocked) else f"{type(exc).__name__}: {str(exc)[:200]}"
        _step(store, project_id, steps, "team", "run", "stopped", why)
        return {"status": "stopped", "at": "a stage", "why": why, "steps": steps, "asked": person.asked, "settled": settled}
    return {**out, "steps": steps, "asked": person.asked, "settled": settled}


def _run(store: Any, project_id: str, description: str, person: Person, advisor: Any, quick: Optional[bool],
         steps: list, settled: dict[str, str]) -> dict[str, Any]:
    import pandas as pd
    from dclab_core.analyse import analyse
    from dclab_core.features import Language, engineer
    from dclab_core.plan import plan as rule_plan, plan_with
    from dclab_core.understand import understand

    from .. import kinds
    from ..core_adapter import solution_fields_from_plan
    from ..intern.tools import Toolbox
    from . import agent, data as sd, engine, graph, solution_report
    from .solution import Solution, provisional_training_rows

    project = store.get(project_id)
    if (project.get("settings") or {}).get("private"):
        advisor = None  # private data: the rules only
    frame = sd.load_table(sd.data_path(store, project))
    if project.get("labels"):
        from . import labels as label_file

        frame = label_file.apply(store, project, frame)
    columns = [str(c) for c in frame.columns]
    suggestion = project.get("suggestion") or {}

    # WF-01: understand; ask only what the brief, the label file and the data leave open; the answers enter the brief
    brief = understand(description, {project["data"]["filename"]: columns}, advisor=advisor, checker=advisor)
    by_model = [name for name, value in brief if getattr(value, "source", None) == "advisor"]
    _step(store, project_id, steps, "problem_analyst", "WF-01", "advisor" if by_model else ("rules" if advisor is None else "fallback"),
          f"filled {', '.join(by_model)}" if by_model else ("no advisor" if advisor is None else "the rules' reading stood"))
    target = brief.target_column.value if brief.target_column.known and brief.target_column.value in columns else suggestion.get("target")
    if not target:
        said = person.ask("target_column", "Which column holds the outcome to predict?")
        target = _column(said, columns)
        if said and not target:
            said = person.ask("target_column", "Which of these columns is the outcome? " + ", ".join(columns[:30]))
            target = _column(said, columns)
        if not target:
            raise Stopped("WF-01", "the target column is still unknown")
        settled["target_column"] = target
    brief = _fact(brief, "target_column", target, settled.get("target_column", target))
    moment = brief.prediction_moment.value if brief.prediction_moment.known else None
    if not moment:
        moment = person.ask("prediction_moment", "When is the prediction made, and what is known at that moment?")
        if not moment:
            raise Stopped("WF-01", "the prediction moment is still unknown")
        settled["prediction_moment"] = moment
        brief = _fact(brief, "prediction_moment", moment, moment)
    task = suggestion.get("task") or (brief.task.value if brief.task.known else sd.detect_task(frame, target)["task"])
    known = {"target_column", "prediction_moment", "task", *(name for name, value in brief if getattr(value, "known", False))}
    for question in brief.open_questions[:3]:  # the checker's disagreements and the analyst's questions, most important first
        name = ALIASES.get(question.field, question.field)
        if name in known or name in settled:
            continue
        said = person.ask(name, question.text)
        if said:
            settled[name] = said
            if name in TEXT_FIELDS:
                brief = _fact(brief, name, said, said)

    # WF-04: analyse the provisional training rows (the earliest, when time orders the rows) with the investigator
    time_column = brief.time_column.value if brief.time_column.known and brief.time_column.value in columns else None
    time_column = time_column or next((c for c in (project["data"].get("profile") or {}).get("time_candidates") or [] if c in columns), None)
    present = frame.index[frame[target].notna()]
    train = provisional_training_rows(present, frame.loc[present, time_column] if time_column else None)
    findings = analyse({"table": frame}, target=target, task=task, brief=brief, train_index=train, advisor=advisor, moment=moment)
    if advisor is not None:
        rules_only = {f.id for f in analyse({"table": frame}, target=target, task=task, brief=brief, train_index=train, moment=moment).findings}
        added = sorted({f.id for f in findings.findings} - rules_only)
    else:
        added = []
    _step(store, project_id, steps, "data_investigator", "WF-04", "advisor" if added else ("rules" if advisor is None else "fallback"),
          f"{len(findings.findings)} findings; the investigator's reading added {len(added)}", added=added[:20])

    # WF-01/WF-03: the plan, validated; the solution saved through the graph validator
    if advisor is not None:
        proposed = plan_with(advisor, brief, findings)
        plan_, kept, refused = proposed["plan"], proposed.get("kept", []), proposed.get("refused", [])
        _step(store, project_id, steps, "solution_planner", "WF-03", "advisor" if kept else "fallback",
              proposed.get("reason") or f"kept {len(kept)}, refused {len(refused)}", kept=[str(k)[:120] for k in kept][:20],
              refused=[str(r)[:200] for r in refused][:20])
    else:
        plan_ = rule_plan(brief, findings)
        _step(store, project_id, steps, "solution_planner", "WF-03", "rules", "no advisor")
    fields = solution_fields_from_plan(plan_, target, moment)
    fields = {**fields, **{k: v for k, v in suggestion.items() if k in ("labels", "label_provenance", "positive_label") and v}}
    detected = kinds.detect(description, {**(project["data"].get("profile") or {}), "labels": project.get("labels") or {}})
    kind = kinds.get(detected["key"])
    pack = kind.packs[0] if kind.base_task == fields.get("task") and kind.packs and not project.get("pack") else None
    if "budget" in (kind.home_fields or ()) and kind.base_task == fields.get("task") and not fields.get("alert_budget"):
        span = None
        time_used = fields.get("time_column")
        if time_used and time_used in frame.columns:
            stamps = pd.to_datetime(frame.loc[train, time_used], errors="coerce").dropna()
            span = (stamps.max() - stamps.min()).total_seconds() / 86400 if len(stamps) > 1 else None
        said = person.ask("budget", "What share of rows can the team review, for example 0.5% of them, or a number a day?")
        share = _share(said, len(train), span)
        if said and share is None:
            said = person.ask("budget", "As a share of the rows, please: for example 0.5% or 0.005.")
            share = _share(said, len(train), span)
        if share:
            fields["alert_budget"] = share
            settled["budget"] = said
    changes: dict[str, Any] = {"policy": {"require_solution_signoff": True, "require_holdout_approval": True}}
    if quick is not None:
        changes["settings"] = {"quick": bool(quick)}
    if pack:
        changes["pack"] = pack
    _write(store, project_id, "the team's gates" + (", the run's rows" if quick is not None else "") + (", the pack" if pack else ""), **changes)
    store.log(project_id, "policy_changed", {"by": "team", "policy": changes["policy"]})
    saved = Toolbox(store).set_solution(project_id, **{k: v for k, v in fields.items() if v not in (None, "", [])})
    if not saved.get("saved"):
        raise Stopped("WF-03", saved.get("error") or "the solution was refused")
    store.update(project_id, plan=plan_.model_dump(mode="json"))  # the plan behind the solution just saved, as adopt_plan keeps it
    solution = Solution(**store.get(project_id)["solution"])

    # WF-06: the feature engineer's programs, proposed before the person signs, so the signature covers them
    programs: list[dict[str, Any]] = []
    if advisor is not None:
        roles = engine.prepare(store, store.get(project_id)).roles
        entities = [c for c, r in roles.items() if r == "group"]
        language = Language([c for c, r in roles.items() if r == "feature"], target=target, time_column=solution.time_column, entities=entities)
        out = engineer(advisor, frame.loc[train].drop(columns=[target]), language, moment=moment, task=solution.task,
                       goal=description[:600], entities=entities)
        programs = out["programs"]
        _step(store, project_id, steps, "feature_engineer", "WF-06", "advisor" if programs else "fallback",
              out.get("reason") or f"{len(programs)} programs kept, {len(out.get('refused', []))} refused",
              kept=[p.get("name") for p in programs][:20], refused=[str(r)[:200] for r in out.get("refused", [])][:20])
        if programs:
            _write(store, project_id, "the feature engineer's programs", feature_programs=programs)
    if not person.approve("solution", {"target": solution.target, "task": solution.task, "moment": solution.prediction_moment,
                                       "forbidden": [f.column for f in solution.forbidden], "labels": solution.labels,
                                       "alert_budget": solution.alert_budget, "feature_programs": [p.get("name") for p in programs]}):
        raise Stopped("solution gate", "the person did not sign the solution")
    graph.approve_gate(store, project_id, "solution", by=person.who, reason="signed in the conversation")

    # WF-05 to WF-08: the stages as the agent's moves; the critic in this run's scope only; the holdout gate signed
    with engine.advisor_scope(advisor), agent.notes_off():
        for stage in ("data", "leakage", "features", "models"):
            engine.execute(store, project_id, stage, actor="agent")
        review = (store.read_stage(project_id, "models").get("evidence") or {}).get("review") or {}
        critic = [r.get("critic") or {} for r in review.get("rounds", [])]
        used = any(c.get("source") == "advisor" for c in critic)
        _step(store, project_id, steps, "critic", "WF-07", "advisor" if used else ("rules" if advisor is None else "fallback"),
              "review rounds in the critic's order" if used else (next((c.get("reason") for c in critic if c.get("reason")), None)
                                                                 or "review rounds in the rules' order"),
              refused=[str(x)[:200] for c in critic for x in c.get("refused") or []][:20])
        models = store.read_stage(project_id, "models")
        if not person.approve("holdout", {"model": (models.get("decision") or {}).get("chosen"), "metric": models.get("primary_metric")}):
            raise Stopped("holdout gate", "the person did not approve opening the holdout")
        graph.approve_gate(store, project_id, "holdout", by=person.who, reason="approved in the conversation")
        engine.execute(store, project_id, "final", actor="agent")

    # WF-10: the report, with the reporter's summaries kept only through cited.check
    report = solution_report.build(store, project_id, advisor=advisor)
    prose = next((e for e in reversed(store.activity(project_id, limit=500)) if e.get("kind") == "report_prose"), None) if advisor is not None else None
    payload = (prose or {}).get("payload") or {}
    _step(store, project_id, steps, "reporter", "WF-10", "advisor" if payload.get("kept") else ("rules" if advisor is None else "fallback"),
          payload.get("reason", "no advisor"), kept=payload.get("kept", 0), dropped=payload.get("dropped", 0))
    final = store.read_stage(project_id, "final")
    return {"status": "completed", "report_sections": len(report.sections),
            "holdout": (final.get("evidence") or {}).get("holdout_metrics", {}).get(final.get("primary_metric")),
            "spent_eur": round(float(advisor.spent_eur()), 4) if advisor is not None and hasattr(advisor, "spent_eur") else 0.0}
