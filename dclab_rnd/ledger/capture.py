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
from typing import Any, Callable, Optional

from dclab_core.bands import ratio_band

from . import answer, batch, column_state, decide, label, rows_band
from .states import table_state, tuning_state  # the fields a student reads, built one way

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
    """Who saved: a person (the value is theirs), an agent through the validator (a model role acting with a check), or the
    rule itself when its proposal is kept as it is with nobody choosing (``"rule"``: the real-table shadow run, 2026-10-23)."""
    if actor == "rule":
        return ("rule", "rule:proposal_kept", 0)
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
            from .states import task_state  # 29.2: the fields the homework's students read, built one way

            written += bool(_put("task.framing", where, [answer("rule", "rule:detect_task", detected if detected in TASKS else None),
                                                          answer(by, who, solution.get("task"))],
                                 solution.get("task"), by, state=task_state(profile, target) if target else problem, options=list(TASKS),
                                 level=level, exposed=exposed, **ids))
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
                       [answer("rule", "rule:id_like+core_keys", name in ids_rule) if own else None, answer(by, who, name in saved_ids)],
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
        # step 5 (2026-10-25): the split student's own fields beside the modifiers' (ledger.mapped: it answers groups and time order)
        from .states import split_state

        split_view = split_state(profile, times[0] if times else None, (own.get("group_candidates") or [None])[0])
        split_extra = {k: split_view[k] for k in ("columns", "date_columns", "id_columns", "repeat_share") if split_view.get(k) is not None}
        for found in own.get("modifiers") or []:  # 30.11: each modifier the rows showed, against what the saved solution keeps
            name = found.get("name")
            # what the saved solution applies (the review of 30.11): a group or time column of any name, the rare metric
            kept = {"rare_positives": solution.get("metric") == "average_precision" or (not solution.get("metric") and bool(found.get("applies"))),
                    "groups": bool(solution.get("group_column")), "time_order": bool(solution.get("time_column")),
                    "noisy_labels": bool(solution.get("label_source_column")), "review_capacity": bool(solution.get("capacity_k"))}.get(name)  # 30.12
            if kept is None:
                continue
            facts = own.get("modifier_facts") or {}
            key = next((k for k in facts.get("keys") or [] if k.get("column") == found.get("value")), {})
            state = {"task": task, "positive_rate": ratio_band(facts.get("minority_rate")) if facts.get("minority_rate") is not None else None,
                     "rows": rows_band(rows), "has_time": bool(facts.get("time_column")), "has_groups": bool(facts.get("keys")),
                     "rows_per_key": key.get("rows_per_key"), "rate_change": facts.get("rate_change"), "class_shift": facts.get("class_shift"),
                     "level_shift": facts.get("level_shift"), "strength": found.get("strength"),
                     **(split_extra if name in ("groups", "time_order") else {})}
            written += bool(_put(f"modifier.{name}", where, [answer("rule", f"rule:modifier.{name}", bool(found.get("applies")), reason=str(found.get("reason") or "")[:600]),
                                                              answer(by, who, kept)],
                                 kept, by, state={k: v for k, v in state.items() if v is not None},
                                 options=["yes", "no"], level=level, exposed=False, **ids))  # no page shows the modifiers' rule yet: never exposed
        groups = list(own.get("group_candidates") or [])
        for name in dict.fromkeys(groups[:10] + ([solution["group_column"]] if solution.get("group_column") else [])):
            if name in columns:
                written += bool(_put("column.group", ("column", name), [answer("rule", "rule:group_candidates+core_keys", name in groups),
                                                                       answer(by, who, name == solution.get("group_column"))],
                                     name == solution.get("group_column"), by, state=column_state(columns[name], rows), options=["yes", "no"],
                                     level=level, exposed=exposed, **ids))
    return written


def _spec(prepared: Any, name: str) -> Any:
    try:
        return getattr(prepared.bundle.spec, name)
    except Exception:  # noqa: BLE001 - a kind's own machine has no bundle
        return None


def _usable(prepared: Any) -> Any:
    try:
        return int(prepared.bundle.X.shape[1])
    except Exception:  # noqa: BLE001
        return None


def _first_family(prepared: Any) -> Any:
    """The family the models stage screens first: the kind's own order for this task (the engine's own function on the same
    inputs), or None when it cannot be known."""
    try:
        from ..expansion import runner as rn
        from ..studio import engine

        families = prepared.kind.model_families(rn.model_families(prepared.bundle.task_type), engine._rare_event(prepared))
        return str(families[0]) if families else None
    except Exception:  # noqa: BLE001 - a kind's own machine, or no bundle: no first family to record
        return None


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

    def put(point: str, rule: Any, rule_id: str, *, person: Any = None, options: Any = (), state: dict[str, Any] | None = None,
            choice: dict[str, Any] | None = None) -> Any:
        nonlocal written
        made = {"source": str(choice.get("part") or "")[:120] or None, "reason": str(choice.get("reason") or "")[:600] or None} if choice else {}
        answers = [answer("rule", rule_id, rule, **made)] + ([answer("person", "owner", person)] if person is not None else [])
        who = "person" if person is not None else "rule"
        rec = _put(point, subject, answers, person if person is not None else rule, who, state={**base, **(state or {})}, options=options, **ids)
        written += bool(rec)
        return rec

    with batch():
        if stage == "data" and prepared is not None:
            solution = getattr(prepared, "solution", None)
            spec = getattr(getattr(prepared, "bundle", None), "spec", None)
            made = {c.get("field"): c for c in ((getattr(prepared, "recipe", None) or {}).get("choices") or [])}  # 30.10: the part behind each
            if spec is not None:
                from .states import split_state  # 29.2: the fields the homework's students read, built one way

                put("split.strategy", getattr(spec, "split_strategy", None), "rule:kind_split",
                    state=split_state(data.get("profile"), getattr(solution, "time_column", None), getattr(solution, "group_column", None)),
                    choice=made.get("split"))
            chosen_metric = getattr(solution, "metric", None)
            put("metric.primary", None if chosen_metric else metric, "rule:kind_default_metric", person=chosen_metric or None,
                choice=made.get("primary_metric"))
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
            for d in getattr(getattr(prepared, "machine", None), "image_decisions", None) or []:  # 26.8: an image table's points
                written += bool(_put(d["point"], ("project", f"{subject[1]}:{d['key']}"[:200]), [answer("rule", f"rule:{d['point']}", d["value"])],
                                     d["value"], "rule", state=d["state"], options=["yes", "no"], **ids))
        elif stage == "features" and decision.get("chosen") is not None:
            rec = put("features.recipe", str(decision.get("selected", decision["chosen"]))[:200], "rule:select_within_tolerance",
                      options=[o.get("id") for o in decision.get("options") or []])
            _measured(rec, decision, _higher(prepared, record))  # 29.3: the recipe that measured clearly best on validation, if one did
        elif stage == "models" and decision.get("chosen") is not None:
            rec = put("model.family", str(decision.get("selected", decision["chosen"]))[:200], "rule:search_select" if decision.get("searched") else "rule:rank_rows",
                      options=[o.get("id") for o in decision.get("options") or []])
            _measured(rec, decision, _higher(prepared, record))  # 29.3: the family that measured clearly best on validation, if one did
            first = _first_family(prepared)
            if first is not None:  # S7: the family tried first (the kind's own order) against the table at a glance; its truth, the one
                # that then measured clearly best: what should have been tried first
                rec = put("model.first", first, "rule:kind_order", options=[o.get("id") for o in decision.get("options") or []],
                          state=table_state((owner.get("data") or {}).get("profile"), (owner.get("solution") or {}).get("target"),
                                            record.get("task_type") or record.get("task"), _spec(prepared, "positive_rate"), _usable(prepared)))
                _measured(rec, decision, _higher(prepared, record))
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
                put("tuning.accept", accepted, "rule:tuning_margin", options=["yes", "no"], state=tuning_state(tuning, evidence, metric))
            point = (evidence.get("task_assessment") or {}).get("threshold")
            value = point.get("value") if isinstance(point, dict) else None
            core = evidence.get("decision") or {}
            if core.get("decision") == "operating_point" and isinstance(core.get("share"), (int, float)):  # 34.4: the share the decision core acts on
                put("threshold.operating_point", float(core["share"]), "rule:decision_core", state={"metric": _code(metric), "of": "share"})
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                put("threshold.operating_point", float(value), "rule:task_assessment", state={"metric": _code(metric), "of": "threshold"})
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
            rule = "rule:search_select" if stage == "models" and decision.get("searched") else "rule:stage_selection"  # 34.7: one name a decision
            rec = _put(point, ("project", project_id), [answer("rule", rule, str(selected)[:200] if selected is not None else None),
                                                        answer(by, who, str(choice)[:200])],
                       str(choice)[:200], by, state={"stage": _code(stage)}, options=options, level=level, exposed=True, project=project_id)
            written += bool(rec)
            _measured(rec, decision, _higher(None, record))  # 29.3: a choice validation clearly beat is labelled wrong, one it clearly backs right
    return written


# ---------------------------------------------------------------------- Home

SOURCES = {"user": "person", "model": "model_role", "agent": "model_role"}  # who set a pack; "problem", "data", "default": the rule


@_quiet
def pack(draft_id: str, key: str, source: str) -> int:
    by = SOURCES.get(source, "rule")
    return int(bool(_put("problem.pack", ("draft", draft_id), [answer(by, f"pack:{source}", key)], key, by,
                         level=2 if by == "model_role" else 0, draft=draft_id, conversation=draft_id)))


TRACKS = ("prediction", "analysis", "preparation", "reporting", "business", "advice")  # draft/intent.py (H2)


@_quiet
def intent_read(draft_id: str, value: dict[str, Any], *, has_data: bool, words: int) -> Optional[str]:
    """H2: the kind of help read from the person's words (the rules, or the model where they found no winner), with the
    rules' own answer beside it; the record's id, kept on the draft's intent, so the person's choice labels this reading."""
    track, rule_track = value.get("track"), value.get("rule_track")
    by = "model_role" if value.get("source") == "model" else "rule"
    from ..draft.intent import RULES_VERSION  # the rules that read it (H4: v2), never a stale name

    answers = [answer("rule", f"rule:{RULES_VERSION}", rule_track if rule_track in TRACKS else None)]
    if by == "model_role":
        answers.append(answer("model_role", "home_intent", track))
    record = _put("request.intent", ("draft", draft_id), answers, track, by, options=TRACKS, level=2 if by == "model_role" else 0,
                  state={"has_data": bool(has_data), "words": int(words), "signals": len(value.get("signals") or [])},
                  draft=draft_id, conversation=draft_id)
    return getattr(record, "id", None)


@_quiet
def routing_read(draft_id: str, point: str, value_used: Any, used_by: str, *, options: Any = (), state: dict[str, Any] | None = None,
                 rule: dict[str, Any] | None = None, model: dict[str, Any] | None = None, own: dict[str, Any] | None = None) -> Optional[str]:
    """P8 (2026-10-29): one routing decision Home makes on a draft (``request.intent`` and its other routing points), with every
    answerer's reading, for the routing model's data (m7_routing). For the model-first order (P1): the model reads, the rules
    check, and the rules answer alone only when no model does.

    - ``rule``: ``{"answerer": "rule:intent_rules_v2", "value": ...}``, the rules' reading (kept whether or not it was used);
    - ``model``: ``{"answerer": "home_intent", "value": ..., "confidence": 0.8, "request": <model request id>}``, the model's
      reading; ``request`` joins it to its kept exchange and API thinking summary exactly (else the collector joins by draft
      and time); a failed or refused call is ``{"answerer": ..., "value": None, "failed": "why"}``;
    - ``own``: ``{"answerer": "classifier:intent_tfidf_v1", "value": ..., "confidence": ...}``, our own model's answer in
      shadow, recorded and never used (``used_by`` may not name it);
    - ``used_by``: ``model_role``, ``rule`` or ``person`` (whose route was taken).

    The state holds bands and codes only (has_data, words, signals: never the words). Returns the record's id, which the
    person's choice on the card then labels (``intent_chosen``)."""
    if used_by not in ("model_role", "rule", "person"):
        return None
    answers = []
    if rule is not None:
        answers.append(answer("rule", str(rule.get("answerer") or "rule"), rule.get("value")))
    if model is not None:
        conf = model.get("confidence")
        answers.append(answer("model_role", str(model.get("answerer") or "model"), model.get("value"),
                              confidence=float(conf) if isinstance(conf, (int, float)) and 0 <= conf <= 1 else None,
                              source=str(model["request"])[:120] if model.get("request") else None,
                              reason=("failed: " + str(model["failed"]))[:600] if model.get("failed") else None))
    if own is not None:
        conf = own.get("confidence")
        answers.append(answer("classifier", str(own.get("answerer") or "classifier"), own.get("value"),
                              confidence=float(conf) if isinstance(conf, (int, float)) and 0 <= conf <= 1 else None))
    record = _put(point, ("draft", draft_id), answers, value_used, used_by, options=options, level=2 if used_by == "model_role" else 0,
                  state=state or {}, draft=draft_id, conversation=draft_id)
    return getattr(record, "id", None)


EXITS = ("ml_project", "analysis", "dataset", "dashboard", "decision_brief", "answer")  # draft/exits.py (H8)


@_quiet
def request_exit(draft_id: str, track: str, kind: str, suggested: Optional[str]) -> Optional[str]:
    """H8: where the draft's kind of help leads, by rule from its track, with the cross-route suggestion the person's words
    gave (none, or another track); labelled by exit_reached when the request really leaves (an answer, a built project)."""
    if kind not in EXITS:
        return None
    record = _put("request.exit", ("draft", draft_id), [answer("rule", "rule:exits_v1", kind)], kind, "rule", options=EXITS,
                  state={"track": track, **({"suggested": suggested} if suggested in TRACKS else {})}, draft=draft_id, conversation=draft_id)
    return getattr(record, "id", None)


@_quiet
def exit_reached(record_id: Optional[str], kind: str) -> int:
    """H8: where the request really left (an answer given, a project built) labels the exit read for it."""
    if not record_id or kind not in EXITS:
        return 0
    from types import SimpleNamespace

    return int(bool(label(SimpleNamespace(id=record_id, point="request.exit"), kind, "measured", "studio")))


@_quiet
def recipe_step_answered(draft_id: str, recipe: dict[str, Any], index: Any, decided: Optional[str]) -> Optional[str]:
    """H9: a guarded preparation step (one that removes many rows, or a join that changes the row count): the rule's answer is
    not to apply it unasked; the person's yes or no labels it. The state is the step's kind and the share it removes, never a value."""
    if decided not in ("yes", "skip"):
        return None
    try:
        raw = recipe["planned"][int(index)]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    asked = next((a for a in reversed(recipe.get("asked") or []) if a.get("index") == int(index)), {})
    record = _put("recipe.step", ("draft", draft_id), [answer("rule", "rule:recipe_guard_v1", False)], False, "rule",
                  state={"step_kind": str(raw.get("kind")), "share": float(asked.get("removed_share") or 0.0)}, draft=draft_id, conversation=draft_id)
    if record is not None:
        label(record, decided == "yes", "user_decision_exposed", "person")
    return getattr(record, "id", None)


def plan_shape(plan: Any) -> str:
    """A query plan as its decision: the operation and its column, the grouping columns, the grain, the filtered columns.
    Column names only, never a filter's value (S4)."""
    p = plan if isinstance(plan, dict) else plan.model_dump(mode="json")
    m = (p.get("measures") or [{}])[0]
    parts = [f"{m.get('op')}({m.get('column') or 'rows'})"]
    if p.get("group_by"):
        parts.append("by:" + ",".join(map(str, p["group_by"])))
    period = p.get("period") or {}
    if period.get("grain"):
        parts.append(f"per:{period['grain']}({period.get('column')})")
    if period.get("start") or period.get("end"):
        parts.append(f"dates:{period.get('column')}")
    if p.get("filters"):
        parts.append("where:" + ",".join(sorted({str(f.get("column")) for f in p["filters"]})))
    return " ".join(parts)[:300]


@_quiet
def query_planned(draft_id: str, plan: Any, by: str, *, fields: int, columns: int, planner: Optional[str] = None,
                  rules_asked: bool = True) -> Optional[str]:
    """S4: how an analysis question was computed: the rules' plan from the brief, or a model's, by its purpose (``planner``:
    home_query when the rules could not map the words, home_agent when its query_data tool planned it); the rules' answer
    is written only when they were asked (an abstention), never on the agent's path. The record's id is kept on the
    answer, so the plan of the answer the person keeps labels it."""
    if by not in ("rules", "model"):
        return None
    shape = plan_shape(plan)
    answers = [answer("rule", "rule:plan_from_brief_v1", shape if by == "rules" else None)] if by == "rules" or rules_asked else []
    if by == "model":
        answers.append(answer("model_role", planner or "home_query", shape))
    record = _put("query.plan", ("draft", draft_id), answers, shape, "rule" if by == "rules" else "model_role",
                  level=2 if by == "model" else 0, state={"brief_fields": int(fields), "columns": int(columns)},
                  draft=draft_id, conversation=draft_id)
    return getattr(record, "id", None)


@_quiet
def query_judged(record_id: Optional[str], plan: Any) -> int:
    """S4: the person kept an answer to the question: the kept answer's plan labels the question's first plan. Kept after
    an edit, a different plan is a correction (it counts); kept as it was, it is agreement, which a label shown to the
    person never counts as evidence (``effective_label``). An edit alone labels nothing: it may be another view."""
    if not record_id:
        return 0
    from types import SimpleNamespace

    return int(bool(label(SimpleNamespace(id=record_id, point="query.plan"), plan_shape(plan), "user_decision_exposed", "person")))


COLUMN_CHECK = ("feature", "keep_out")


REVIEWER_CODES = {"available": "available", "after the moment": "after_moment"}  # the leakage reviewer's verdicts as codes


@_quiet
def column_checked(draft_id: str, item: dict[str, Any], answered: str, profile: Optional[dict[str, Any]] = None) -> Optional[str]:
    """S4: a column the audit kept out on statistics alone (H3.1): the rule recommends (a feature only when the leakage
    reviewer read it as known in time and said when it is set); the person, shown the recommendation, answers: known (a
    feature) or written later (kept out). "Not sure" is no label. The record is the column's (joined with its
    feature.leakage_suspect record by name); its state is the column's bands from the draft's profile and what the rule
    read, never a value."""
    from ..draft import column_check as checks

    name = str(item.get("column") or "")
    if not name:
        return None
    rule = "feature" if checks.recommended(item) else "keep_out"
    person = {"known": "feature", "later": "keep_out"}.get(answered)
    answers = [answer("rule", "rule:column_check_v1", rule)] + ([answer("person", "person", person)] if person else [])
    profiled = next((c for c in (profile or {}).get("columns") or [] if str(c.get("name")) == name), None)
    bands = column_state(profiled) if profiled else {"column": name[:200]}
    record = _put("column.statistics_only", ("column", name[:200]), answers, person or rule,
                  "person" if person else "rule", options=COLUMN_CHECK,  # labelled below, agreeing or not
                  state={**bands, "reviewer": REVIEWER_CODES.get(str(item.get("reviewer") or ""), "none"),
                         "names_moment": bool(item.get("names_moment"))},
                  draft=draft_id, conversation=draft_id)
    if record is not None and person:
        label(record, person, "user_decision_exposed", "person")
    return getattr(record, "id", None)


@_quiet
def intent_chosen(record_id: Optional[str], track: str) -> int:
    """H2: the person's choice on the card is the label of this draft's own reading, never of another draft's."""
    if not record_id or track not in TRACKS:
        return 0
    from types import SimpleNamespace

    return int(bool(label(SimpleNamespace(id=record_id, point="request.intent"), track, "user_decision_exposed", "person")))


OUTCOME_BY = "person:outcome"  # 36.2: the labels track_outcome writes, so it retracts only its own (never the person's track choice)


@_quiet
def track_outcome(draft_id: str, outcome: Any, *, exit_record: Optional[str], exit_kind: Optional[str], intent_record: Optional[str],
                  log: Any = None) -> int:
    """36.2: what the person did with what the routing produced (a dclab_core.track_outcome.TrackOutcome, read by its rule
    from the draft's ordered actions). Call it when the outcome changes. One track.outcome event each time, names and
    counts only (open too: produced and not judged yet is counted). The draft's own exit (the exit kind taken) and reading
    (the track) carry the person's label only while the final verdict is kept or changed: a later refused, asked again or
    open supersedes it with an empty label, so a route the person finally turned down is never confirmed. Refused and asked
    again are never labels of a route themselves (no right answer is guessed from them). Returns the labels written."""
    if outcome is None:
        return 0
    from types import SimpleNamespace

    from .. import experience
    from . import current_ledger

    experience.record("track.outcome", "person" if outcome.by == "person" else "system",
                      {"track": outcome.track, "exit": exit_kind if exit_kind in EXITS else None, "produced": list(outcome.produced),
                       "verdict": outcome.verdict, "by": outcome.by, "actions": outcome.actions, "evidence": list(outcome.evidence)},
                      log=log, conversation=draft_id)
    confirms = outcome.verdict in ("kept", "changed") and outcome.by == "person"
    ledger_ = current_ledger()
    n = 0
    for record_id, point, value in ((exit_record, "request.exit", exit_kind if exit_kind in EXITS else None),
                                    (intent_record, "request.intent", outcome.track if outcome.track in TRACKS else None)):
        if not record_id or value is None:
            continue
        mine = [lb for lb in (ledger_.labels(record=record_id) if ledger_ is not None else []) if getattr(lb, "by", None) == OUTCOME_BY]
        gone = {lb.supersedes for lb in mine if lb.supersedes}
        current = next((lb for lb in reversed(mine) if lb.id not in gone), None)  # this function's latest label, not superseded
        held = current is not None and current.value is not None
        if confirms and not held:
            n += int(bool(label(SimpleNamespace(id=record_id, point=point), value, "user_decision_exposed", OUTCOME_BY,
                                supersedes=current.id if current is not None else None)))
        elif not confirms and held:
            n += int(bool(label(SimpleNamespace(id=record_id, point=point), None, "user_decision_exposed", OUTCOME_BY, supersedes=current.id)))
    return n


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
def route_chosen(draft_id: str, sub: Any, route: Any, data: Any, parts: int, shadow: Any = None, shadow_by: str = "") -> Any:
    """The approach routed to for one sub-problem (24.2's map, 24.3's catalogue): the rules' choice, the advisor's when it served, with the
    candidates as the options. The state holds the family, the label situation and what the data card shows (counts and flags, never a value).
    ``shadow`` (28.10): another answerer's route for the same sub-problem (a local model in shadow), recorded beside the others and never used.
    The record (or None)."""
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
    if shadow is not None:
        from dclab_core.ledger import SHADOW

        answers.append(answer("model_role", f"shadow:{shadow_by or 'model'}"[:120], getattr(shadow, "chosen", None), source=SHADOW))
    return _put("solution.route", ("draft", draft_id), answers, route.chosen, by, state=state, options=[c.approach for c in route.candidates],
                level=2 if by == "model_role" else 0, draft=draft_id, conversation=draft_id)


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
def more_rows(project_id: str, result: Any, *, rows: int, columns: int, positive_rate: Optional[float] = None) -> int:
    """Would more rows help (34.13): the rule's answer from the learning curve on the training folds. The state is what the answer
    could be predicted from (the training rows' band, the feature count, the metric, how a fit's rows were cut, the score with every
    row), never the curve's gain the rule read, nor a segment's level."""
    state: dict[str, Any] = {"rows": rows_band(rows), "columns": int(columns), "metric": _code(result.metric), "strategy": _code(result.subsample),
                             "has_time": result.subsample == "time", "has_groups": result.subsample == "group"}
    if result.points:
        state["cv_mean"] = float(result.points[-1].score)
    if positive_rate is not None:
        state["positive_rate"] = ratio_band(float(positive_rate))
    rec = _put("data.more_rows", ("project", project_id), [answer("rule", f"rule:{result.rule}", result.answer, reason=result.plain[:600] or None)],
               result.answer, "rule", state={k: v for k, v in state.items() if v is not None}, options=("likely_help", "unlikely_help", "cannot_tell"),
               project=project_id)
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


def _labels_band(draft: dict[str, Any]) -> str:
    """Whether the cases come as a file of their own (then the outcome question is withdrawn: the file defines them)."""
    try:
        from ..draft.plan import label_asset

        return "file" if label_asset(draft) is not None else "unknown"
    except Exception:  # noqa: BLE001 - a draft the helper cannot read: unknown
        return "unknown"


@_quiet
def next_question(draft: dict[str, Any], asked: Any) -> int:
    """The question actually asked next (its field), or none: recorded after the agent asked, so a question the plan
    wanted but could not ask (too many) is recorded as none. 26.3: the options are every field Home can ask, a field
    asked under a second name (a contradiction's check) is recorded under its own, the state says how many questions
    are left before the conversation's cap, and nothing is recorded while a question waits for its answer (nothing
    was decided then)."""
    from dclab_core.workflow_labels import QUESTION_LIMIT, QUESTIONS, SAME

    field = asked[0] if asked else None
    u = draft.get("understanding") or {}
    questions = [q for q in draft.get("questions") or [] if not q.get("superseded")]
    if field is None and any(not q.get("answered") for q in questions):
        return 0  # a question is pending: the next one is not decided yet
    state = {"open_questions": sum(1 for f in ("target", "prediction_moment", "action") if not u.get(f)), "step": len(questions),
             "steps_left": max(QUESTION_LIMIT - len(questions), 0), "labels": _labels_band(draft),
             "outcome_known": bool(u.get("target")), "moment_known": bool(u.get("prediction_moment")),
             "pack": _code((draft.get("pack") or {}).get("key"))}
    field = "kind" if str(field or "").startswith("kind_") else field  # S9.3: a field of the understood kind (randomised, a series)
    field = "entity" if field == "entity_check" else field  # the repeating key's question (the owner, 2026-10-26) asks for the entity
    value = _code(SAME.get(field, field)) or "none"
    return int(bool(_put("workflow.next_question", ("conversation", draft["id"]), [answer("rule", "rule:plan_fields", value)], value, "rule",
                         state=state, options=[*QUESTIONS, "kind"], draft=draft["id"], conversation=draft["id"])))


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


COLUMN_STEPS = ("missing_markers", "numbers", "dates", "booleans", "trim")  # the steps that act on one column's values


@_quiet
def cleaning(draft_id: str, log: list[dict[str, Any]], rows: int | None, profile: dict[str, Any] | None = None) -> int:
    """Each cleaning step the rules applied, per column it changed (the step's key; never a cell). With ``profile`` (the
    raw table's, 29.2): every column, its first column step or ``none``, with the raw column's bands, as the homework's."""
    from .states import cleaning_state

    written = 0
    # 29.6: the steps after "names" name the columns it renamed by their new names; the raw profile knows the old ones
    raw_of = {str(new): str(old) for entry in log if entry.get("step") == "names" for old, new in entry.get("renamed") or []}
    with batch():
        if profile is not None:
            first: dict[str, str] = {}
            for entry in log:
                if entry.get("step") in COLUMN_STEPS:
                    for column in entry.get("columns") or []:
                        first.setdefault(raw_of.get(str(column), str(column)), str(entry["step"]))
            for col in (profile.get("columns") or [])[:MAX_COLUMNS]:
                name = str(col.get("name"))
                step = first.get(name, "none")
                written += bool(_put("data.cleaning_step", ("column", name[:200]), [answer("rule", "rule:clean", step)], step, "rule",
                                     state=cleaning_state(profile, name), options=[*COLUMN_STEPS, "none"], draft=draft_id, conversation=draft_id))
        for entry in log[:50]:
            step = _code(entry.get("step"))
            for column in (entry.get("columns") or ["*"])[:20]:
                if not step:
                    continue
                # 29.6: every step applied to every column it changed (the rules keep each; until a person can revert a
                # step or homework marks one wrong, the point holds one answer), the column by its raw name and bands
                raw = raw_of.get(str(column), str(column))
                bands = cleaning_state(profile, raw) if profile is not None and column != "*" else {"rows": rows_band(rows)}
                written += bool(_put("data.cleaning_keep", ("column", raw[:200]), [answer("rule", "rule:clean", True)], True, "rule",
                                     state={**bands, "cleaning": step}, options=["yes", "no"], draft=draft_id, conversation=draft_id))
                if profile is not None and step in COLUMN_STEPS:
                    continue  # its first step recorded above, with every column
                written += bool(_put("data.cleaning_step", ("column", raw[:200]), [answer("rule", "rule:clean", step)], step, "rule",
                                     state={"rows": rows_band(rows)}, draft=draft_id, conversation=draft_id))
    return written


@_quiet
def structure_read(draft_id: str, path: Any, fmt: str | None) -> Any:
    """29.6: the reader the structure reader chose for an uploaded file (``info["reader"]``; ``format`` is what that
    reader then found), with what it looked at (``structure.evidence``: codes and flags, never a line). The truth is the
    construction's where a file's format is known (the tests' files)."""
    from dclab_core.ledger import STRUCTURE_FORMATS

    from ..draft.structure import evidence

    value = _code(fmt)
    if value not in STRUCTURE_FORMATS:
        return None
    seen = evidence(path)  # inside the quiet capture: a failure here never fails the upload that was read
    return _put("data.structure_format", ("draft", draft_id), [answer("rule", "rule:structure", value)], value, "rule",
                state={k: v for k, v in seen.items() if v is not None}, options=list(STRUCTURE_FORMATS), draft=draft_id, conversation=draft_id)


@_quiet
def synthetic_shape(draft_id: str, info: dict[str, Any], rows: int | None, model_available: bool) -> Any:
    """29.6: the design a synthetic table follows: a template the person picked, the model's own design, or the template
    the rules fell back to; never the person's words. The truth is the people's: whether the draft goes on with it."""
    template, source = _code(info.get("template")), str(info.get("source") or "")
    value = template if template else "model" if source == "model" else None
    if value is None:
        return None
    by = "person" if info.get("chosen_by") == "user" else "model_role" if source == "model" else "rule"
    who = {"person": "owner", "model_role": "synthetic_schema", "rule": "rule:template_for"}[by]
    answers = [answer(by, who, value)]
    if info.get("fallback_reason"):  # the model was asked and gave no valid design: its answer, none, with why (never in the state)
        answers.insert(0, answer("model_role", "synthetic_schema", None, reason=str(info["fallback_reason"])[:600]))
    state = {"rows": rows_band(rows) if rows else None, "model_available": bool(model_available)}  # what was known before the design
    return _put("data.synthetic_shape", ("draft", draft_id), answers, value, by,
                state={k: v for k, v in state.items() if v is not None}, level=2 if by == "model_role" else 0,  # the model's design acts after the spec's check
                draft=draft_id, conversation=draft_id)


@_quiet
def evidence_search(question: str, hits: list[dict[str, Any]], weak: float) -> Any:
    """29.6: the record the evidence search ranked first for a question (subject: an id of its own, never its words),
    with how many came back, the best score and the method; the truth is the people's, or the retrieval set's 60 questions."""
    import uuid

    if not hits:
        return None
    top = max(float(h.get("score") or 0.0) for h in hits)
    key = uuid.uuid4().hex[:16]  # each search its own subject: a hash of a short question could be reversed by a dictionary
    value = str(hits[0].get("record_id") or "")[:200]
    if not value:
        return None
    method = _code(hits[0].get("method")) or "unknown"
    return _put("evidence.records", ("conversation", key), [answer("rule", f"retrieval:{method}", value)], value, "rule",
                state={"hits": len(hits), "top_score": round(top, 4), "method": method, "weak": top < weak},
                options=sorted(str(h.get("record_id"))[:200] for h in hits if h.get("record_id")), conversation=key)  # sorted: the order never says the answer


@_quiet
def gate_g4(project: dict[str, Any], gate: dict[str, Any], evidence: dict[str, Any]) -> Any:
    """30.15: G4 on the final stage's facts, as bands (never a score or a row): how often the holdout was read, whether the
    folds explain it, a score too good, the brief's verdict, a rule, the monitoring plan's reach. The rule's answer is
    passed or the phase it sends the work back to."""
    checks = {c.get("name"): bool(c.get("ok")) for c in gate.get("checks") or []}
    plan = evidence.get("monitoring") or {}
    state = {"holdout_used": int(project.get("holdout_uses") or 0),
             "folds_explain": checks.get("The folds explain the holdout"), "too_good": None if "Not too good to be honest" not in checks
             else not checks["Not too good to be honest"], "bar": _code(str((evidence.get("success_bar") or {}).get("verdict") or "none")),
             "rule_set": checks.get("A rule to act on"), "monitoring": "score" if plan.get("score") else ("inputs" if plan.get("columns") else "none"),
             "calibrated": _code(str(((evidence.get("calibration_fit") or {}).get("calibrator") or {}).get("method") or "none"))}
    value = "passed" if gate.get("passed") else str(gate.get("back_to"))
    return _put("workflow.g4", ("project", project["id"]), [answer("rule", "workflow_run:g4", value, reason="; ".join(gate.get("reasons") or [])[:600] or None)],
                value, "rule", state={k: v for k, v in state.items() if v is not None}, options=["passed", "problem", "data", "brief", "workflow"],
                project=project["id"])


@_quiet
def fit_verdict(draft: dict[str, Any], verdict: Any, checked: Any = None, card: Any = None, accepted: bool = False) -> Any:
    """30.13: G2's verdict (fits, fits with changes, does not fit), with what it was read from: the mismatches, blocking
    and not, the family, the labels, whether a time column exists and whether noisy labels were found (never the verdict's
    own edits or missing data). At a build the rule's answer; when a person accepts the edits, theirs beside it (the truth
    the point is labelled by)."""
    mismatches = list(getattr(checked, "mismatches", None) or [])
    labels = getattr(getattr(card, "labels", None), "value", None)
    state = {"blocking": sum(m.blocking for m in mismatches), "open_questions": sum(not m.blocking for m in mismatches),
             "family": getattr(getattr(card, "family", None), "value", None), "labels": labels if labels in ("column", "file", "none") else "unknown",
             "has_time": not any(m.code == "time_absent" for m in mismatches),
             "noisy": any(str(r).startswith("noisy_labels") for r in verdict.reasons)}
    answers = [answer("rule", "meet:fit", verdict.verdict, reason="; ".join(verdict.reasons)[:600] or None)]
    if accepted:  # the person saw the verdict and its edits, and accepted them
        answers.append(answer("person", "owner", verdict.verdict))
    return _put("data.fit_verdict", ("draft", draft["id"]), answers, verdict.verdict, "person" if accepted else "rule",
                state={k: v for k, v in state.items() if v is not None}, options=["fits", "fits_with_changes", "does_not_fit"],
                exposed=accepted, draft=draft["id"], conversation=draft["id"])


@_quiet
def cases_table(draft_id: str, name: str, cases: dict[str, Any] | None, profile: dict[str, Any]) -> int:
    """Whether a table is a list of known cases (the labels of another table) rather than rows to score."""
    from .states import cases_state  # 29.2: the fields the homework's students read, built one way

    found = cases is not None
    return int(bool(_put("data.cases_table", ("draft", draft_id), [answer("rule", "rule:cases_table", found)], found, "rule",
                         state=cases_state(profile, name),
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


@_quiet
def leak_screened(project_id: str, verdict: Any, profile: dict[str, Any] | None, decided: dict[str, bool] | None = None,
                  screened: list[str] | None = None) -> int:
    """28.5: the auditor's screen before a holdout is read, on the point ``feature.leakage_suspect``: every column it screened, the
    rule's answer (a suspect or not, with how cleanly it alone separates the cases), and for a suspect the person's decision when
    they made one (kept out or kept; shown the rule's answer first, so a person who keeps a suspect contradicts it). Columns
    it did not flag are recorded once, when screened; a suspect when the person decides (or with the rule alone when the run
    stopped on it). Names and bands only."""
    columns = {c.get("name"): c for c in (profile or {}).get("columns") or []}
    rows = (profile or {}).get("rows")
    suspects = {s.column: s for s in getattr(verdict, "suspects", [])}
    flagged = set(suspects) | {i.column for i in getattr(verdict, "after_moment", [])}  # the screen's suspects and what the moment says comes after it
    scores = dict(getattr(verdict, "scores", {}) or {})
    written = 0
    names = list(decided) if decided is not None else [c for c in (screened or []) if c not in flagged]
    if decided is None and getattr(verdict, "verdict", None) == "stop":
        names += sorted(flagged)
    for name in dict.fromkeys(names):
        s = suspects.get(name)
        state = {**(column_state(columns[name], rows) if name in columns else {"column": str(name)[:200]}), "task": "binary_classification",
                 "separation": round(2 * abs(scores[name] - 0.5), 4) if name in scores else None,  # every screened column's, a suspect's or not
                 "positives": int(getattr(verdict, "positives", 0) or 0)}
        answers = [answer("rule", "rule:auditor_screen", name in flagged, reason=(s.reason[:600] if s is not None else None))]
        if decided is not None and name in decided:
            answers.append(answer("person", "team audit", bool(decided[name])))
            used, value = "person", bool(decided[name])
        else:
            used, value = "rule", name in flagged
        written += bool(_put("feature.leakage_suspect", ("column", str(name)), answers, value, used, state={k: v for k, v in state.items() if v is not None},
                             options=["yes", "no"], exposed=used == "person", project=project_id))
    return written


TEAM_NODES = ("answer", "draw_map", "confirm_map", "bring_table", "analyse", "reread", "plan_workflow", "build", "critic", "critic_gate",
              "route", "design", "audit", "audit_gate", "compose", "drop_gate", "report", "stop", "done", "wait_for_data")


@_quiet
def team_node(owner: tuple[str, str], team: str, state_code: str, rule_next: str, chosen: str, *, blockers: int = 0, open_questions: int = 0,
              pending: bool = False, critic_passed: bool | None = None, step: int = 0) -> Any:
    """29.6: the node a team's walk takes next. The coordinator's situation answers (``rule_next``); the walk chose ``chosen``
    (the critic before the workflow, a pending question first, a stop on the step budget). Codes and counts only."""
    state = {"team": _code(team) or "team", "from_state": _code(state_code) or "state", "blocking": int(blockers), "open_questions": int(open_questions),
             "pending": bool(pending), "step": int(step), **({"critic_passed": bool(critic_passed)} if critic_passed is not None else {})}
    answers = [answer("rule", "rule:situation", str(rule_next)), answer("rule", f"rule:{team}.situate", str(chosen))]
    kind, ident = owner
    return _put("team.next_node", (kind, ident), answers, str(chosen), "rule", state=state, options=list(TEAM_NODES),
                **({"draft": ident, "conversation": ident} if kind == "draft" else {"project": ident}))


def _higher(prepared: Any, record: dict[str, Any] | None = None) -> Any:
    """29.3: the metric's direction as the engine keeps it (a kind's machine, else the tabular bundle's spec, as the final
    stage's outcome reads it); without a prepared project, from the stage record (its machine's, else its metric's);
    None when it is not known, and then nothing is labelled by measurement."""
    machine = getattr(prepared, "machine", None)
    if machine is not None:
        value = getattr(machine, "higher_is_better", None)
        return value if isinstance(value, bool) else None
    try:
        value = prepared.bundle.spec.higher_is_better
    except Exception:  # noqa: BLE001 - no spec: read the record
        value = None
        try:
            if (record or {}).get("machine"):
                from .. import kinds

                value = getattr(kinds.get(str(record["machine"])).own_machine(), "higher_is_better", None)
            elif (record or {}).get("primary_metric"):
                from ..expansion.runner import HIGHER_IS_BETTER

                value = HIGHER_IS_BETTER.get(str(record["primary_metric"]))
        except Exception:  # noqa: BLE001 - a kind or a metric the tables do not know
            value = None
    return value if isinstance(value, bool) else None


def _measured(record: Any, decision: dict[str, Any], higher: Any) -> None:
    """29.3: a measured label on a choice among measured options (``dclab_core.measured.clear_best``): the option that
    measured clearly best on validation, in the metric's own direction; nothing when the record was not kept or validation
    cannot tell the options apart."""
    if record is None:
        return
    from dclab_core.measured import clear_best

    got = clear_best(decision.get("options") or [], higher)
    if got is not None:
        value, evidence = got
        label(record, value, "measured", evidence["kind"], evidence=evidence)


@_quiet
def second_opinion(draft_id: str, opinion: Any, chosen: Any, stands: Any = None) -> Any:
    """28.6: the second opinion on the card's family: the conversation reader's family (a model role's) and the card rules',
    the judge's verdict, and the family the person chose (or none: the card stands). The person was shown both readings."""
    state = {"verdict": _code(opinion.verdict) or "ask", "agreed": opinion.reader is not None and opinion.reader == opinion.rules}
    answers = [answer("model_role", "conversation_reader", opinion.reader), answer("rule", "card:family", opinion.rules, reason=opinion.why[:600]),
               answer("person", "person", chosen) if chosen is not None else None]
    used = "person" if chosen is not None else "model_role" if stands is not None and stands == opinion.reader else "rule"
    return _put("problem.family", ("draft", draft_id), answers, chosen if chosen is not None else (stands or opinion.rules), used,
                state=state, options=list(opinion.choices), exposed=chosen is not None, draft=draft_id, conversation=draft_id)


@_quiet
def critic_weighed(draft_id: str, field: str, source: str, sent: bool, reason: str = "", rounds: int = 0) -> Any:
    """29.6: one concern the understanding critic weighed for a card field: sent back to the person or not (a round spent, the
    field asked before, a model's doubt about what the person said, its words refused), with why."""
    state = {"field": _code(field) or "field", "concern_source": _code(source) or "rules", "rounds": int(rounds)}
    return _put("critic.send_back", ("draft", draft_id), [answer("rule", f"rule:critic.{_code(source) or 'rules'}", bool(sent), reason=(reason or None) and reason[:600])],
                bool(sent), "rule", state=state, options=["yes", "no"], draft=draft_id, conversation=draft_id)


# ---------------------------------------------------------------------- a department agent's proposal (30.22)

@_quiet
def department_proposal(agent: str, step: Any, right: Optional[bool] = None, person: str = "person") -> Any:
    """One proposal of a department agent: its claim (yes, it is one to make), whether the policy let it through and it was applied, at
    which level; with the person's judgment as an exposed label (the person saw the proposal). The agent's track record is these labels."""
    p = step.proposal
    change = min(1.0, abs(p.after - p.before) / abs(p.before)) if p.before else 1.0
    state = {"move": _code(p.action) or "action", "change": round(change, 4), "allowed": bool(step.verdict.allowed),
             "needs_approval": bool(step.verdict.needs_approval), "applied": bool(step.applied), "agent_level": _code(step.level) or "shadow"}
    record = _put("department.proposal", ("proposal", f"{_code(agent) or 'agent'}:{step.id}"),
                  [answer("rule", f"agent:{_code(agent) or 'agent'}", True, reason=(p.reason or None) and p.reason[:600])], bool(step.applied), "rule",
                  state=state, options=["yes", "no"])
    if record is not None and right is not None:
        label(record, bool(right), "user_decision_exposed", person)
    return record
