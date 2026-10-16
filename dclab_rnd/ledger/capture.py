"""Where this studio decides, the records it writes (package 19.7). Each function reads what the decision site already
has (a proposal and the saved solution, a stage's record, a move's verdict) and calls :func:`dclab_rnd.ledger.decide`.
They never change what is decided and never raise: a decision stands whether or not its record is kept.

Two rules hold everywhere. An answer is attributed to whoever gave it: the rule that computed it, the model role that
proposed it, the person who saved it; a studied sample's written solution is planted truth, so it becomes a benchmark
label, never an answer. And a state never holds the decision's own value or the verdict it led to (a model trained on
the state must not read its label there)."""

from __future__ import annotations

import re
import sys
from typing import Any, Callable

from . import answer, batch, column_state, decide, label, rows_band

MAX_COLUMNS = 200  # columns recorded per saved solution, besides every column a rule or the person named
TASKS = {"binary": "binary_classification", "multiclass": "multiclass_classification", "regression": "regression"}
DEFAULT_FOLDS = 3


def _quiet(fn: Callable[..., Any]) -> Callable[..., Any]:
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the decision stands; its record is reported and skipped
            print(f"ledger: {fn.__name__} skipped ({type(exc).__name__}: {str(exc)[:120]})", file=sys.stderr, flush=True)
            return None
    run.__name__ = fn.__name__
    return run


def _code(value: Any) -> Any:
    text = str(value or "").lower()
    return text if text and re.fullmatch(r"[a-z0-9_]+([.:-][a-z0-9_]+)*", text) and len(text) <= 80 else None


def _who(actor: str) -> tuple[str, str, int]:
    """Who saved: a person (the value is theirs), or an agent through the validator (a model role acting with a check)."""
    return ("model_role", "intern", 2) if actor == "agent" else ("person", "owner", 0)


def _put(point: str, subject: tuple[str, str], answers: list[Any], value: Any, used: str, *, state: dict[str, Any] | None = None,
         options: Any = (), level: int = 0, exposed: bool = False, review: bool = False, **ids: Any) -> Any:
    answers = [a for a in answers if a is not None]
    record = decide(point, subject, answers=answers, value_used=value, used=used, state=state or {},
                    options=[str(o)[:200] for o in options if o is not None][:64], level=level, review=review, **ids)
    if record is not None and exposed and used == "person":  # 19.8: a person who was shown the rule's answer and chose otherwise
        rule = next((a for a in answers if a.by == "rule" and a.value is not None), None)
        if rule is not None and rule.value != value:
            label(record, value, "user_decision_exposed", next((a.answerer for a in answers if a.by == "person"), "person"))
    return record


# ---------------------------------------------------------------------- a saved solution

@_quiet
def solution_saved(proposal: dict[str, Any] | None, profile: dict[str, Any] | None, solution: dict[str, Any], actor: str = "human",
                   truth: dict[str, Any] | None = None, exposed: bool = True, **ids: Any) -> int:
    """A saved solution against what was proposed for its target: the task, and for each column whether it is an
    identifier, whether it leaks, whether it is the time or the entity column. The rules' answers are the proposal's own
    (``proposal["rule"]``: before a studied sample's written solution is merged in); the leakage reviewer's per-column
    verdicts are a model's answers; the saved value is the person's (or the intern's). A studied sample's written
    solution (``truth``) becomes benchmark labels. A proposal made for another target is not this decision's: its
    answers are left out. Returns how many records were written."""
    proposal = proposal or {}
    if proposal.get("target") not in (None, solution.get("target")):
        proposal = {}  # made for another outcome: the audit and the identifiers it found were about that one
    own = proposal.get("rule") or proposal  # what propose() computed, before a studied sample's solution joined it
    columns = {c.get("name"): c for c in (profile or {}).get("columns") or [] if c.get("name")}
    rows = (profile or {}).get("rows")
    by, who, level = _who(actor)
    target = solution.get("target")
    task = TASKS.get(str(solution.get("task") or ""))
    where = ("project", ids["project"]) if ids.get("project") else ("draft", ids.get("draft") or "?")
    problem = {"rows": rows_band(rows), "columns": len(columns) or None}
    written = 0
    with batch():
        if target:  # nothing ranks the outcome for the person (a key once became one): only their choice is recorded
            written += bool(_put("target.column", where, [answer(by, who, target)], target, by, state={**problem, "outcome_known": True},
                                 options=((profile or {}).get("target_candidates") or [])[:20], level=level, exposed=exposed, **ids))
        detected = (proposal.get("detected") or {}).get("task")
        if task:
            written += bool(_put("task.framing", where, [answer("rule", "rule:detect_task", detected if detected in TASKS else None),
                                                          answer(by, who, solution.get("task"))],
                                 solution.get("task"), by, state=problem, options=list(TASKS), level=level, exposed=exposed, **ids))
        ids_rule = set(own.get("identifiers") or [])
        by_model = {str(d.get("column")): d for d in own.get("decided_by_models") or [] if d.get("answerer")}  # 22.9
        audit = {f.get("column") for f in own.get("forbidden") or [] if "DCLAB-R01" not in (f.get("proof") or [])
                 and not any(str(p).startswith("decision-model:") for p in f.get("proof") or [])}
        saved_ids = set(solution.get("identifiers") or [])
        forbidden = {f.get("column") if isinstance(f, dict) else getattr(f, "column", None) for f in solution.get("forbidden") or []}
        review = proposal.get("review") or {}
        moment_written = bool(str(solution.get("prediction_moment") or "").strip())
        by_moment = {i["column"] for i in review.get("items") or [] if i.get("source") == "moment"}
        verdicts = {v["column"]: v["verdict"] == "after the moment" for v in review.get("verdicts") or [] if v.get("column")}
        truth = truth if truth and ({"identifiers", "forbidden"} & set(truth)) else {}  # a written solution, not the chat's notes
        true_ids, true_forbidden = set(truth.get("identifiers") or []), {f.get("column") for f in truth.get("forbidden") or []}
        named = ids_rule | saved_ids | audit | forbidden | set(verdicts) | by_moment | true_ids | true_forbidden
        order = list(columns)
        for name in [n for n in order if n != target and (n in named or order.index(n) < MAX_COLUMNS)]:
            column = columns[name]
            bands = column_state(column, rows)
            rec = _put("column.is_identifier", ("column", name),
                       [answer("rule", "rule:id_like", name in ids_rule) if own else None, answer(by, who, name in saved_ids)],
                       name in saved_ids, by, state=bands, options=["yes", "no"], level=level, exposed=exposed, **ids)
            written += bool(rec)
            if truth and rec is not None:
                label(rec, name in true_ids, "benchmark", f"sample:{truth.get('key') or 'written'}")
            answers = [answer("rule", "rule:column_audit", name in audit) if own else None]
            if moment_written and proposal:
                answers.append(answer("rule", "rule:moment", name in by_moment))
            if name in verdicts:
                answers.append(answer("model_role", "leakage_review", verdicts[name]))
            if name in by_model and by_model[name].get("model") is not None:  # 22.9: the routed classifier's answer, shown in the proposal
                answers.append(answer("classifier", by_model[name]["answerer"], bool(by_model[name]["model"])))
            answers.append(answer(by, who, name in forbidden))
            rec = _put("feature.leakage_suspect", ("column", name), answers, name in forbidden, by,
                       state={**bands, "target": str(target)[:200] if target else None, "task": task}, options=["yes", "no"], level=level, exposed=exposed,
                       review=bool((by_model.get(name) or {}).get("review")), **ids)
            written += bool(rec)
            if truth and rec is not None:
                label(rec, name in true_forbidden, "benchmark", f"sample:{truth.get('key') or 'written'}")
        times = list(own.get("time_candidates") or [])
        for name in dict.fromkeys(times[:10] + ([solution["time_column"]] if solution.get("time_column") else [])):
            if name in columns:
                written += bool(_put("column.time", ("column", name), [answer("rule", "rule:time_candidates", name == (times[0] if times else None)),
                                                                      answer(by, who, name == solution.get("time_column"))],
                                     name == solution.get("time_column"), by, state=column_state(columns[name], rows), options=["yes", "no"],
                                     level=level, exposed=exposed, **ids))
        groups = list(own.get("group_candidates") or [])
        for name in dict.fromkeys(groups[:10] + ([solution["group_column"]] if solution.get("group_column") else [])):
            if name in columns:
                written += bool(_put("column.group", ("column", name), [answer("rule", "rule:group_candidates", name in groups),
                                                                       answer(by, who, name == solution.get("group_column"))],
                                     name == solution.get("group_column"), by, state=column_state(columns[name], rows), options=["yes", "no"],
                                     level=level, exposed=exposed, **ids))
    return written


def _states(frame: Any, names: list[Any]) -> dict[Any, dict[str, Any]]:
    """Each named column's state, profiled from the stage's rows: its name, dtype and bands, never a value."""
    if frame is None:
        return {}
    from ..studio.data import profile_table

    present = [n for n in names if n in frame.columns]
    if not present:
        return {}
    rows = len(frame)
    by_name = {str(c["name"]): c for c in profile_table(frame[present])["columns"]}
    return {n: column_state(by_name[str(n)], rows) for n in present if str(n) in by_name}


# ---------------------------------------------------------------------- a stage's choices

@_quiet
def stage_decisions(stage: str, record: dict[str, Any], prepared: Any, owner: dict[str, Any], **ids: Any) -> int:
    """A finished stage's decisions, from its record: what the rule chose among the options it measured, and what the
    person had set (the metric, the folds)."""
    decision = record.get("decision") or {}
    evidence = record.get("evidence") or {}
    metric = record.get("primary_metric")
    data = owner.get("data") or {}
    base = {"rows": rows_band(data.get("rows")), "columns": len(data.get("columns") or []) or None,
            "task": TASKS.get(str(record.get("task_type") or record.get("task") or ""))}
    subject = ("project", owner.get("id") or "?")
    settings = owner.get("settings") or {}
    written = 0

    def put(point: str, rule: Any, rule_id: str, *, person: Any = None, options: Any = (), state: dict[str, Any] | None = None) -> Any:
        nonlocal written
        answers = [answer("rule", rule_id, rule)] + ([answer("person", "owner", person)] if person is not None else [])
        who = "person" if person is not None else "rule"
        rec = _put(point, subject, answers, person if person is not None else rule, who, state={**base, **(state or {})}, options=options, **ids)
        written += bool(rec)
        return rec

    with batch():
        if stage == "data" and prepared is not None:
            solution = getattr(prepared, "solution", None)
            spec = getattr(getattr(prepared, "bundle", None), "spec", None)
            if spec is not None:
                put("split.strategy", getattr(spec, "split_strategy", None), "rule:kind_split",
                    state={"has_time": bool(getattr(solution, "time_column", None)), "has_groups": bool(getattr(solution, "group_column", None))})
            chosen_metric = getattr(solution, "metric", None)
            put("metric.primary", None if chosen_metric else metric, "rule:kind_default_metric", person=chosen_metric or None)
            folds = settings.get("folds")
            put("split.folds", DEFAULT_FOLDS, "rule:default_folds", person=int(folds) if isinstance(folds, int) else None)
            sampling = getattr(prepared, "sampling", None) or {}
            if sampling:  # every row, the latest by time, or a seeded random sample: a code, never the rule's words
                rule = str(sampling.get("rule") or "")
                value = "latest_by_time" if "latest" in rule else "random_seeded" if "random" in rule else "all_rows"
                used_rows, total = sampling.get("rows_used"), sampling.get("rows_with_target")
                put("data.sampling", value, "rule:sampling", options=["all_rows", "latest_by_time", "random_seeded"],
                    state={"share": used_rows / total if isinstance(used_rows, int) and isinstance(total, int) and total else None})
            frame = getattr(prepared, "frame", None)
            numeric = set(frame.select_dtypes("number").columns) if frame is not None else set()
            states = _states(frame, list((getattr(prepared, "roles", None) or {}))[:MAX_COLUMNS])  # 22.4: the bands a classifier reads
            decided = {str(c.subject): c for c in getattr(prepared, "decided", None) or []}  # 22.9: a routed model's answers here
            for name, role in list((getattr(prepared, "roles", None) or {}).items())[:MAX_COLUMNS]:
                kind = str(role).split(" ")[0]
                value = {"blocked": "other", "identifier": "identifier", "time": "datetime", "group": "categorical_code",
                         "text": "free_text"}.get(kind, "numeric" if name in numeric else "categorical_code")
                c = decided.get(str(name))
                answers = [answer("rule", "rule:prepare_roles", ("numeric" if name in numeric else "categorical_code") if c else value)]
                if c is not None and c.answerer:
                    answers.append(answer("classifier", c.answerer, c.model_value))
                used = "classifier" if c is not None and c.by_model else "rule"
                written += bool(_put("column.semantic_role", ("column", name), answers, value, used,
                                     state=states.get(name) or {"column": str(name)[:200]},
                                     options=["numeric", "categorical_code", "identifier", "datetime", "free_text", "other"],
                                     **({"level": c.level, "review": c.review} if c is not None else {}), **ids))
            for name in list(getattr(spec, "categorical_columns", None) or [])[:MAX_COLUMNS]:
                if name in numeric:  # a number read as a code for categories
                    written += bool(_put("column.category_code", ("column", str(name)), [answer("rule", "rule:" + (_code(getattr(prepared, "code_rule", "")) or "codes"), True)],
                                         True, "rule", state=states.get(name) or {"column": str(name)[:200], "dtype": "integer"}, options=["yes", "no"], **ids))
        elif stage == "leakage":
            declared = list(evidence.get("declared_leakage_features") or [])
            per = (evidence.get("safe_vs_unsafe_training_cv") or {}).get("per_column") or {}
            findings = {f.get("feature"): f for f in evidence.get("heuristic_review_candidates") or [] if f.get("feature")}
            names = list(dict.fromkeys(declared + list(findings)))
            bands = _states(getattr(prepared, "frame", None), names)  # 22.4: the bands a classifier reads, as the solution's decisions have them
            target = getattr(getattr(prepared, "solution", None), "target", None)
            for name in names:
                lift = (per.get(name) or {}).get("apparent_lift")
                if lift is None and declared == [name]:
                    lift = evidence.get("apparent_lift")
                rec = _put("feature.leakage_suspect", ("column", str(name)),
                           [answer("rule", "rule:leakage_heuristics", name in findings), answer("person", "solution", name in declared)],
                           name in declared, "person", state={**(bands.get(name) or {"column": str(name)[:200]}), "task": base["task"],
                                                              "target": str(target)[:200] if target else None,
                                                              "apparent_lift": float(lift) if isinstance(lift, (int, float)) else None}, options=["yes", "no"], **ids)
                written += bool(rec)
                if rec is not None and isinstance(lift, (int, float)) and name in declared and lift > 0:  # measured: it would have looked better
                    label(rec, True, "measured", "apparent_lift", evidence={"kind": "apparent_lift", "value": float(lift), "ref": f"leakage.evidence.{name}"[:200]})
            gap = (evidence.get("random_vs_time_cv") or {}).get("optimism_gap")
            if isinstance(gap, (int, float)) and gap > 0:  # 19.8 measured: a random split would have looked better than it is
                from . import current_ledger

                book = current_ledger()
                splits = [r for r in (book.records(project=ids.get("project"), point="split.strategy") if book is not None and ids.get("project") else [])
                          if r.value_used == "time"]  # the gap is measured against a time split: it confirms that one only
                if splits:
                    label(splits[-1], "time", "measured", "optimism_gap",
                          evidence={"kind": "optimism_gap", "value": float(gap), "ref": "leakage.evidence.random_vs_time_cv"})
        elif stage == "features" and decision.get("chosen") is not None:
            put("features.recipe", str(decision.get("selected", decision["chosen"]))[:200], "rule:select_within_tolerance",
                options=[o.get("id") for o in decision.get("options") or []])
        elif stage == "models" and decision.get("chosen") is not None:
            put("model.family", str(decision.get("selected", decision["chosen"]))[:200], "rule:rank_rows",
                options=[o.get("id") for o in decision.get("options") or []])
            rounds = [r for r in (evidence.get("review") or {}).get("rounds") or [] if isinstance(r, dict)]
            for r in rounds[:20]:
                critic = (r.get("critic") or {}).get("order") or []
                experiment = r.get("experiment")
                model_pick = critic[0] if critic else None
                by_model = bool(critic) and model_pick == experiment and r.get("rule_first") != experiment
                routed = r.get("decision_model") or {}  # 22.9: a routed decision model's pick, after the validator
                by_classifier = routed.get("used") not in (None, "rule", "none") and routed.get("model") == experiment
                written += bool(_put("experiment.next", subject, [answer("rule", "rule:review_ranking", r.get("rule_first")),
                                                                  answer("model_role", "critic", model_pick) if critic else None,
                                                                  answer("classifier", routed["answerer"], routed.get("model")) if routed.get("answerer") else None],
                                     experiment, "classifier" if by_classifier else "model_role" if by_model else "rule",
                                     state=routed.get("state") or {**base, "step": int(r.get("round") or 0)},  # 22.9: the state the model was shown
                                     options=r.get("options") or [experiment], review=bool(routed.get("review")),
                                     level=int(routed.get("level") or 2) if by_classifier else 2 if by_model else 0, **ids))
                if isinstance(r.get("adopted"), bool):
                    put("experiment.accept", r["adopted"], "rule:paired_bound", options=["yes", "no"],
                        state={"kind": _code(experiment), "step": int(r.get("round") or 0)})
            if rounds:
                put("model.review", "change" if any(r.get("adopted") for r in rounds) else "keep", "rule:paired_bound",
                    options=["keep", "change"], state={"step": len(rounds)})
        elif stage == "final":
            tuning = evidence.get("tuning_decision") or {}
            accepted = tuning.get("accepted") if isinstance(tuning, dict) else None
            if isinstance(accepted, bool):
                put("tuning.accept", accepted, "rule:tuning_margin", options=["yes", "no"],
                    state={"tuning_gain": tuning.get("gain") if isinstance(tuning.get("gain"), (int, float)) else None})
            point = (evidence.get("task_assessment") or {}).get("threshold")
            value = point.get("value") if isinstance(point, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                put("threshold.operating_point", float(value), "rule:task_assessment", state={"metric": _code(metric)})
    return written


# ---------------------------------------------------------------------- the workflow graph

@_quiet
def move(verdict: Any, entry: dict[str, Any], project: dict[str, Any], project_id: str) -> int:
    """A move the workflow validator checked: whether it was allowed at that state, among the moves allowed there. A
    final stage asked again after the holdout was used is also a decision on reusing it; a completed stage run again
    is a revisit."""
    stage = (verdict.args or {}).get("stage")
    key = verdict.move + (f":{stage}" if stage else "")
    used_holdout = int(project.get("holdout_uses") or 0)
    state = {"move": _code(verdict.move), "stage": _code(stage), "from_state": _code(verdict.node_from), "holdout_used": used_holdout}
    by = "person" if verdict.actor == "human" else "model_role"
    written = 0
    with batch():
        written += bool(_put("workflow.move", ("workflow", key[:200]),
                             [answer(by, verdict.actor, True), answer("rule", "rule:graph_validator", verdict.allowed, reason=str(verdict.message or "")[:600] or None)],
                             verdict.allowed, "rule", state=state, options=entry.get("allowed") or [], project=project_id))
        if verdict.move == "run_stage" and stage == "final" and used_holdout >= 1:
            reason = bool((verdict.args or {}).get("reuse_reason"))
            written += bool(_put("holdout.reuse", ("project", project_id), [answer(by, verdict.actor, True), answer("rule", "rule:holdout_once", verdict.allowed)],
                                 verdict.allowed, "rule", state={**state, "reason_given": reason}, options=["yes", "no"], project=project_id))
        status = ((project.get("stages") or {}).get(stage) or {}).get("status") if stage else None
        if verdict.move == "run_stage" and verdict.allowed and status in ("completed", "approved"):  # an agent's run, checked as it starts
            written += revisit(str(stage), verdict.actor, project_id, used_holdout) or 0
    return written


@_quiet
def revisit(stage: str, actor: str, project_id: str, holdout_used: int = 0) -> int:
    """A completed stage asked to run again (by a person through the page, or an agent): which step is revisited."""
    by = "person" if actor == "human" else "model_role"
    return int(bool(_put("workflow.revisit", ("workflow", f"run_stage:{stage}"), [answer(by, actor, stage)], stage, by,
                         state={"stage": _code(stage), "holdout_used": holdout_used}, level=2 if by == "model_role" else 0, project=project_id)))


@_quiet
def approval(stage: str, record: dict[str, Any] | None, choice: Any, actor: str, project_id: str) -> int:
    """A stage approved; and when the person (or the intern) chose another option than the rule's, that choice at the
    stage's own point (the recipe, the family), with both answers."""
    decision = (record or {}).get("decision") or {}
    selected = decision.get("selected", decision.get("chosen"))  # the rule's: an override has already replaced "chosen"
    by, who, level = _who(actor)
    options = [str(o.get("id"))[:200] for o in decision.get("options") or []][:64]
    written = 0
    with batch():
        written += bool(_put("workflow.stage_approval", ("project", project_id), [answer(by, who, True)], True, by,
                             state={"stage": _code(stage)}, options=options, level=level, project=project_id))
        point = {"features": "features.recipe", "models": "model.family"}.get(stage)
        if point and choice not in (None, "") and str(choice) != str(selected):
            written += bool(_put(point, ("project", project_id), [answer("rule", "rule:stage_selection", str(selected)[:200] if selected is not None else None),
                                                                  answer(by, who, str(choice)[:200])],
                                 str(choice)[:200], by, state={"stage": _code(stage)}, options=options, level=level, exposed=True, project=project_id))
    return written


# ---------------------------------------------------------------------- Home

SOURCES = {"user": "person", "model": "model_role", "agent": "model_role"}  # who set a pack; "problem", "data", "default": the rule


@_quiet
def pack(draft_id: str, key: str, source: str) -> int:
    by = SOURCES.get(source, "rule")
    return int(bool(_put("problem.pack", ("draft", draft_id), [answer(by, f"pack:{source}", key)], key, by,
                         level=2 if by == "model_role" else 0, draft=draft_id, conversation=draft_id)))


@_quiet
def map_drawn(draft_id: str, m: Any, rules_size: str, by_model: bool) -> int:
    """The problem map drawn (24.2): one problem or several. The rules' reading is an answer; the decomposer's, when it served, another."""
    answers = [answer("rule", "rule:decompose", rules_size)]
    if by_model:
        answers.append(answer("model_role", "decomposer", m.size))
    state = {"parts": len(m.parts), "template": _code(m.template) or "none"}
    return int(bool(_put("map.size", ("draft", draft_id), answers, m.size, "model_role" if by_model else "rule", state=state, options=["single", "complex"],
                         level=2 if by_model else 0, draft=draft_id, conversation=draft_id)))


@_quiet
def map_edited(draft_id: str, action: str, before: int, after: int) -> int:
    """A person's change to the map (add, remove, merge, confirm): what they did, and how many parts there were before and after."""
    return int(bool(_put("map.edit", ("draft", draft_id), [answer("person", "owner", action)], action, "person",
                         state={"parts_before": int(before), "parts_after": int(after)}, options=["add", "remove", "merge", "confirm", "redraw"],
                         draft=draft_id, conversation=draft_id)))


@_quiet
def route_chosen(draft_id: str, sub: Any, route: Any, data: Any, parts: int) -> int:
    """The approach routed to for one sub-problem (24.2's map, 24.3's catalogue): the rules' choice, the advisor's when it served, with the
    candidates as the options. The state holds the family, the label situation and what the data card shows (counts and flags, never a value)."""
    rules_choice = route.chosen if route.source != "advisor" else None
    answers = []
    if route.source == "advisor":
        base = next((c.approach for c in route.candidates if not c.blocking), None)
        answers = [answer("rule", "rule:route", base), answer("model_role", "solution_router", route.chosen)]
    else:
        answers = [answer("rule", "rule:route", rules_choice, reason=route.reason[:600] or None)]
    tables = getattr(data, "tables", None) or []
    columns = [c for t in tables for c in (getattr(t, "columns", None) or [])]
    state = {"family": sub.family, "labels": sub.labels if sub.labels in ("column", "file", "none") else "unknown", "parts": int(parts), "sub_problem": _code(sub.id) or "sp",
             "tables": len(tables), "related_tables": sum(1 for t in tables if getattr(t, "role", None) == "related"), "has_time": any(getattr(c, "purpose", None) == "time" for c in columns),
             "has_text": any(getattr(c, "purpose", None) == "text" for c in columns)}
    by = "model_role" if route.source == "advisor" else "rule"
    return int(bool(_put("solution.route", ("draft", draft_id), answers, route.chosen, by, state=state, options=[c.approach for c in route.candidates],
                         level=2 if by == "model_role" else 0, draft=draft_id, conversation=draft_id)))


@_quiet
def composition_chosen(project_id: str, p: Any) -> int:
    """The composition of a programme chosen on validation (24.5): the rule's choice among every leaf alone and the fusions, with the candidates as the options. When the
    holdout was read for the first time, a measured label follows: the composition when it beat the best leaf with its interval above zero, the best leaf when the
    composition was clearly below it, nothing when the interval crosses zero or the holdout had been read before. What it says is what the holdout compared: the chosen
    composition against the best single leaf (the holdout scores only those); it does not say the chosen was the best of all the candidates, and its evidence names the
    comparison (``holdout_vs_best_leaf``)."""
    c, h = p.composition, p.holdout
    if c is None:
        return 0
    state = {"parts": len(c.weights), "positives": int(c.positives), "holdout_used": int(p.reads_before)}
    rec = _put("solution.compose", ("project", project_id), [answer("rule", "rule:compose_on_validation", c.chosen, reason=c.rule[:600] or None)], c.chosen, "rule", state=state,
               options=list(c.candidates), project=project_id)
    if rec is not None and h is not None and p.reads_before == 0:  # a second look at a holdout already seen is optimistic (PIT-006): it labels nothing
        value = c.chosen if h.beats_best_leaf else (c.best_leaf if h.difference.high < 0 else None)
        if value is not None:
            label(rec, value, "measured", "holdout_gain", evidence={"kind": "holdout_vs_best_leaf", "value": float(h.difference.value), "ref": "programme.holdout.difference.value"})
    return 1 if rec is not None else 0


@_quiet
def workflow_drawn(draft_id: str, key: str, workflow: dict[str, Any], previous: dict[str, Any] | None, proposed: bool, problems: list[str]) -> int:
    """The solution workflow drawn: the pack's template, or a model's proposal the validator accepted or refused.
    Written when something was decided: a model proposed one, or the pack or the source changed."""
    before = previous or {}
    if not proposed and before.get("pack") == workflow.get("pack") and before.get("source") == workflow.get("source"):
        return 0  # drawn again with the same template: nothing new was decided
    model = workflow.get("source") == "model"
    answers = [answer("rule", "rule:pack_template", key)]
    if proposed or model:
        answers.append(answer("model_role", "home_agent", key if not problems else None, reason="; ".join(problems)[:600] or None))
    return int(bool(_put("workflow.template", ("draft", draft_id), answers, key, "model_role" if model else "rule",
                         state={"pack": _code(key)}, level=2 if model else 0, draft=draft_id, conversation=draft_id)))


@_quiet
def next_question(draft: dict[str, Any], asked: Any) -> int:
    """The question actually asked next (its field), or none: recorded after the agent asked, so a question the plan
    wanted but could not ask (too many, or still pending) is recorded as none."""
    field = asked[0] if asked else None
    u = draft.get("understanding") or {}
    questions = [q for q in draft.get("questions") or [] if not q.get("superseded")]
    state = {"open_questions": sum(1 for f in ("target", "prediction_moment", "action") if not u.get(f)), "step": len(questions),
             "outcome_known": bool(u.get("target")), "moment_known": bool(u.get("prediction_moment")),
             "pack": _code((draft.get("pack") or {}).get("key"))}
    value = _code(field) or "none"
    return int(bool(_put("workflow.next_question", ("conversation", draft["id"]), [answer("rule", "rule:plan_fields", value)], value, "rule",
                         state=state, options=["target", "prediction_moment", "action", "labels", "none"], draft=draft["id"], conversation=draft["id"])))


@_quiet
def problem_card(draft: dict[str, Any], card: Any) -> int:
    """At a build: the card's family and whether the problem is ready (each without its own answer in its state)."""
    stated = getattr(card.family, "status", None) == "stated"  # the person said the family; otherwise the card's rules
    base = {"labels": card.labels.value if card.labels.value in ("column", "file", "none") else "unknown",
            "open_questions": len(card.questions or []), "outcome_known": bool(card.outcome)}
    written = 0
    with batch():
        if card.family.value:
            by = "person" if stated else "rule"
            written += bool(_put("problem.family", ("draft", draft["id"]), [answer(by, "card:family", card.family.value)], card.family.value, by,
                                 state=base, options=list(card.in_play or []), draft=draft["id"], conversation=draft["id"]))
        written += bool(_put("problem.ready", ("draft", draft["id"]), [answer("rule", "card:ready", bool(card.ready), reason="; ".join(card.blocking or [])[:600] or None)],
                             bool(card.ready), "rule", state={**base, "family": card.family.value}, options=["yes", "no"], draft=draft["id"], conversation=draft["id"]))
    return written


@_quiet
def cleaning(draft_id: str, log: list[dict[str, Any]], rows: int | None) -> int:
    """Each cleaning step the rules applied, per column it changed (the step's key; never a cell)."""
    written = 0
    with batch():
        for entry in log[:50]:
            step = _code(entry.get("step"))
            for column in (entry.get("columns") or ["*"])[:20]:
                if step:
                    written += bool(_put("data.cleaning_step", ("column", str(column)[:200]), [answer("rule", "rule:clean", step)], step, "rule",
                                         state={"rows": rows_band(rows)}, draft=draft_id, conversation=draft_id))
    return written


@_quiet
def cases_table(draft_id: str, name: str, cases: dict[str, Any] | None, profile: dict[str, Any]) -> int:
    """Whether a table is a list of known cases (the labels of another table) rather than rows to score."""
    found = cases is not None
    return int(bool(_put("data.cases_table", ("draft", draft_id), [answer("rule", "rule:cases_table", found)], found, "rule",
                         state={"rows": rows_band(profile.get("rows")), "columns": len(profile.get("columns") or []) or None},
                         options=["yes", "no"], draft=draft_id, conversation=draft_id)))


@_quiet
def join_key(review: dict[str, Any], person_key: list[str] | None = None, **ids: Any) -> int:
    """The key that joins a label file to the table: the rule's (most label rows found, never inflating), or the person's."""
    candidates = review.get("candidates") or []
    chosen = "+".join(review.get("key") or []) or None
    rule = max((c for c in candidates if c.get("verdict") in ("possible", "chosen")),
               key=lambda c: (c["label_rows_found"], -len(c["key"]), -c["rows_per_label"]), default=None)
    answers = [answer("rule", "rule:label_join", "+".join(rule["key"]) if rule else None)]
    used = "rule"
    if person_key:
        answers.append(answer("person", "owner", "+".join(person_key)))
        used = "person"
    share = review.get("label_rows_found") / review["label_rows"] if review.get("label_rows") else None
    where = ("project", ids["project"]) if ids.get("project") else ("draft", ids.get("draft") or "?")
    return int(bool(_put("labels.join_key", where, answers, chosen, used, state={"share": share, "labels": "file"},
                         options=["+".join(c["key"]) for c in candidates], **ids)))


# ---------------------------------------------------------------------- agents, the report, lessons

@_quiet
def tool_choice(policy: Any, step: Any, allowed: list[str], trace: Any) -> int:
    """A model's choice of a tool among those its policy offered (acted on through the scope's guard)."""
    run = getattr(trace, "run_id", None)
    state = {"agent": _code(policy.name), "step": int(step.n), "tools_offered": len(allowed), "steps_left": max(int(policy.max_steps) - int(step.n), 0)}
    answers = [answer("model_role", str(policy.name), str(step.tool)[:200])]
    from . import use

    if use.routed("agent.tool_choice"):  # 22.9: a routed decision model checks the agent's choice; it never replaces it
        chosen = str(step.tool)[:200]
        checked = use.choose("agent.tool_choice", state, list(allowed), chosen, f"agent:{_code(policy.name)}", subject=str(run or policy.name),
                             check=lambda v: [] if v == chosen else ["a decision model checks an agent's tool and never replaces it"])
        if checked.answerer and checked.model_value is not None:
            answers.append(answer("classifier", checked.answerer, str(checked.model_value)[:200]))
    return int(bool(_put("agent.tool_choice", ("conversation", str(run or policy.name)), answers,
                         str(step.tool)[:200], "model_role", state=state, options=allowed, level=2, **({"run": run} if run else {}))))


@_quiet
def stopped(policy: Any, result: Any, trace: Any, interrupted: bool = False) -> int:
    """Why an agent's run ended: the model said it was done (an answer, a final tool), or a rule ended it (a budget, an
    error, a person's Stop)."""
    reason = getattr(result, "stopped", None) if result is not None else ("stopped" if interrupted else "failed")
    run = getattr(trace, "run_id", None)
    by_rule = reason in ("steps", "time", "replies", "failed", "tools_as_text", "stopped") or interrupted
    answers = [answer("rule", "rule:budget" if reason in ("steps", "time", "replies") else "rule:ended", True)] if by_rule \
        else [answer("model_role", str(policy.name), True)]
    steps = len(getattr(result, "steps", None) or [])
    return int(bool(_put("workflow.stop", ("workflow", f"{policy.name}:{run or ''}"[:200]), answers, True, "rule" if by_rule else "model_role",
                         state={"agent": _code(policy.name), "step": steps, "steps_left": max(int(policy.max_steps) - steps, 0)},
                         level=0 if by_rule else 2, **({"run": run} if run else {}))))


@_quiet
def report_claims(project_id: str, info: dict[str, Any]) -> int:
    """The reporter model's summary sentences, each kept or dropped by the citation check (the lines code writes always
    stay: they are not decisions)."""
    written = 0
    with batch():
        for number, (_, section) in enumerate((info.get("sections") or {}).items()):
            verdicts = [(True, None)] * int(section.get("kept") or 0) + [(False, str(r)[:600]) for r in section.get("dropped") or []]
            for kept, why in verdicts[:40]:
                written += bool(_put("report.claim", ("project", project_id), [answer("model_role", "reporter", True), answer("rule", "rule:cited_check", kept, reason=why)],
                                     kept, "rule", state={"step": number}, options=["yes", "no"], project=project_id))
    return written


@_quiet
def lesson_review(lesson: dict[str, Any], action: str, role: str | None) -> int:
    """A reviewer's verdict on a lesson a finished project proposed (accepted, edited and so accepted, or rejected)."""
    kept = action != "reject"
    project = lesson.get("project_id")
    proposer = answer("model_role", "lesson_proposal", True) if lesson.get("written_by") == "model" else answer("rule", "rule:lesson_drafts", True)
    return int(bool(_put("lesson.accept", ("project", str(project or "?")), [proposer, answer("person", str(role or "reviewer"), kept)], kept, "person",
                         state={"kind": _code(lesson.get("written_by"))}, options=["yes", "no"], **({"project": project} if project else {}))))
