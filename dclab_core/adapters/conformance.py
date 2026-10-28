"""The conformance gate of every package (package 23.12): what the main product's adapters cover and why they leave the rest.

Every contract, every event kind, every decision point and every id has either a mapping in an adapter or a *reason* it has none,
written here and in the adapters' own lists (``KINDS_MISSING``, ``IDS_MISSING``, ``FIELDS_MISSING``); nothing is left unsaid. ``problems()``
returns what is missing or wrong, empty when the adapters are conformant: the studio's test runs it, and the main product's
repository can run it on its side when it adopts a package or when its own commit moves (the field lists are re-read then).
"""

from __future__ import annotations

from typing import Any

from dclab_core.adapters import (main_product, main_product_experience, main_product_ledger, main_product_lineage, main_product_outcomes,
                                 main_product_pipeline, main_product_problem_map, main_product_twin)

ADAPTERS = {m.__name__: m for m in (main_product, main_product_experience, main_product_ledger, main_product_lineage, main_product_outcomes,
                                    main_product_pipeline, main_product_problem_map, main_product_twin)}  # the modules a ``module:function`` below may name
LIBRARY = "library output: no row of the main product maps onto it; the host renders or stores it from the library's calls (CORE_INTEGRATION.md)"

#: contract name -> ``module:function`` of the adapter that fills it from the main product's rows, or the reason it has none
CONTRACT_ADAPTERS: dict[str, str] = {
    "experience_event": "dclab_core.adapters.main_product_experience:events",
    "experience_session": LIBRARY + " (``experience.session_card`` over adapted events)",
    "run_record": "dclab_core.adapters.main_product_lineage:from_experiment",
    "decision_record": "dclab_core.adapters.main_product_ledger:from_semantic_answer",
    "decision_label": "dclab_core.adapters.main_product_ledger:from_semantic_answer",
    "decision_model_card": LIBRARY + " (the library's record of a classifier it fitted)",
    "outcome": "dclab_core.adapters.main_product_outcomes:from_experiment",
    "pipeline": "dclab_core.adapters.main_product_pipeline:pipeline_version",
    "problem_map": "dclab_core.adapters.main_product_problem_map:problem_specs",
    "route": LIBRARY + " (the approach chosen for a sub-problem, with the candidates and what blocks them; the main product reads it as a proposal at its own points)",
    "solution_map": "dclab_core.adapters.main_product_problem_map:holdout_plan",
    "programme": "dclab_core.adapters.main_product_problem_map:programme_experiments",
    "map_review": LIBRARY + " (the reviewer's questions about a map; the main product asks them through its own assistant)",
    "problem_brief": "dclab_core.adapters.main_product:problem_spec",
    "solution_plan": "dclab_core.adapters.main_product:problem_spec",
    "data_findings": "dclab_core.adapters.main_product:problem_profile",
    "decision": "dclab_core.adapters.main_product:problem_spec",
    "review": LIBRARY + " (a reviewer's verdict on a plan; the main product's ExperimentReview is read as a decision record)",
    "data_card": LIBRARY + " (drawn from a table the host hands the library)",
    "dataset_manifest": LIBRARY + " (written by the builders over adapted events and records)",
    "problem_card": LIBRARY + " (the understanding conversation's result)",
    "problem_data_check": LIBRARY + " (the problem set against its tables)",
    "fit_verdict": LIBRARY + " (G2: the problem card against the data card, 30.13)",
    "calibrator": LIBRARY + " (step 8 of the Workflow Run, fitted on out-of-fold scores, carried in the artifact, 30.15)",
    "monitoring_plan": LIBRARY + " (step 12: the training rows' bins and the alert lines, carried in the artifact, 30.15)",
    "monitor_report": LIBRARY + " (what monitor() says of new rows; the main product's deployment reads it, 30.15)",
    "gate_result": LIBRARY + " (a gate's verdict with the phase it sends the work back to, 30.15)",
    **{name: LIBRARY + f" (the store's {what}, 32.1: the studio keeps its training data in its own store; the main product would keep its "
                          "own beside its records, through an adapter when it trains on them)" for name, what in (
        ("store_entry", "kept source"), ("store_record", "normalised record"), ("decision_example", "decision example"),
        ("chat_example", "chat example"), ("preference_pair", "preference pair"), ("solver_trajectory", "solver trajectory"),
        ("episode", "step-by-step episode"), ("judged_example", "judged example"), ("retrieval_example", "retrieval example"),
        ("curation_decision", "curator's choice"), ("set_manifest", "set manifest"), ("capture_rule", "capture rule"),
        ("workflow_record", "workflow record of a walk"), ("experiment_record", "experiment record of a fitted model"))},
    **{name: LIBRARY + f" (the service's {what}, 32.24 to 32.31: built from the studio's store; the main product would build its own from "
                          "its projects' records, through an adapter, when it offers the service)" for name, what in (
        ("model_build", "record of one model built"), ("build_verdict", "verdict of a build against DCLab's rules"),
        ("learning_curve", "learning curve"), ("knowledge_item", "knowledge item"), ("service_engagement", "engagement record"),
        ("playbook_item", "playbook item"))},
    **{name: LIBRARY + f" (the harness that learns' {what}, H0 and H2, 2026-10-23: built from invented challenges and the studio's "
                          "graded attempts; the main product would grade its own agents through the same contracts when it runs the loop)" for name, what in (
        ("challenge_pack", "public challenge pack"), ("challenge_key", "challenge key, read by the grader only"), ("grade", "grade of one solve"),
        ("playbook", "playbook of lessons"), ("playbook_delta", "change to a playbook"))},
    "decision_spec": LIBRARY + " (what acting on a row is worth and costs, 34.1; the main product's objective takes its costs through "
                     "adapters.main_product.objective_of, the rest goes to the person choosing the operating point)",
    "prediction_record": LIBRARY + " (one row scored by a model in use, its key a salted hash, 34.9; the main product runs customer models "
                         "and writes one per row it scores, CORE_INTEGRATION.md 'The prediction and outcome log')",
    "outcome_record": LIBRARY + " (the true outcome of a scored key once known, 34.9; the main product writes it where its customers report "
                      "outcomes, joined to its predictions by serving.join)",
    "data_value": LIBRARY + " (would more rows help: the learning curve on the training folds, 34.13; the main product draws it with its own "
                  "trainer in each fold, CORE_INTEGRATION.md 'Would more rows help?')",
    "selection_record": LIBRARY + " (how a candidate was chosen among many: the selection fold read once and the measured optimism, 34.6; "
                        "the main product's experiment runner sets the fold aside before its search, CORE_INTEGRATION.md 'Honest selection')",
    "batch_profile": LIBRARY + " (one scored batch's inputs as the plan's bins see them, never a value, 34.10; the main product's scoring "
                     "path writes one per request beside its prediction records, CORE_INTEGRATION.md 'Monitoring that runs')",
    "live_report": LIBRARY + " (a model in use read window by window from its logs, 34.10; the main product's monitoring job reads its "
                   "own logs through live.monitor_log and records the alerts as its events)",
    "retrain_policy": LIBRARY + " (what starts a retrain: a drift or performance alert, a schedule, enough new labels, 34.11; the main "
                      "product's model registry keeps one per model in use)",
    "challenger_record": LIBRARY + " (one champion/challenger comparison on the latest window neither trained on, 34.11; the main product "
                         "promotes its challenger only on promoted=true and keeps every record, CORE_INTEGRATION.md 'Retrain and champion/challenger')",
    "collect_advice": LIBRARY + " (what to collect next: the groups that carry the score on the training folds, the domain's usual kinds "
                      "missing, where labels help, 34.14; the main product shows it beside its data requests)",
    "memory_item": LIBRARY + " (one remembered line, 35.1: the main product has no memory table at the pinned commit; "
                   "adapters/main_product_memory.py is the row it would add, as_row and from_row)",
    "query_plan": LIBRARY + " (H6: a question asked of a table, aggregates only; the main product has no such front door at the pinned commit)",
    "query_result": LIBRARY + " (H6: the aggregates a plan returned, small groups suppressed and counted; stored with an answer, never as rows)",
    "recipe": LIBRARY + " (H9: a table prepared step by step on request; the main product has no such front door at the pinned commit)",
    "change_result": LIBRARY + " (H11: where a number changed between two periods, by segment; the main product has no such front door at the pinned commit)",
    "dashboard": LIBRARY + " (H10: KPIs as query plans with their version, tiles and refresh runs; the main product has no such front door at the pinned commit)",
    "memory_brief": LIBRARY + " (35.1: what an agent reads at a session's start, built by memory.select; never stored as a row)",
    "exam": LIBRARY + " (a model type's frozen exam, 36.4: rows named by key and hash, never their content; the main product trains "
            "no model of its own at the pinned commit, so it has no exam table)",
    "exam_result": LIBRARY + " (one model version's score on its exam and the no-forgetting gate, 36.4; kept with the version beside its card)",
    "replay_round": LIBRARY + " (36.5: a training round's set, new rows and a share of every earlier round; the main product trains no model of its own at the pinned commit)",
    "track_outcome": LIBRARY + " (36.2: what a person did with an answer, table, brief or dashboard Home produced; the main product has no such front door at the pinned commit)",
    "table_entry": LIBRARY + " (a table's catalog entry with its owner and checks, 30.23; the main product's data catalog keeps its own entries)",
    "project_request": LIBRARY + " (a requested project scored, 30.24; the main product's intake keeps its own requests and scores them through it)",
    "portfolio": LIBRARY + " (the projects chosen of the requested and what they reuse, 30.24)",
    "department_policy": LIBRARY + " (a department agent's policy and limits, 30.22; the main product's agents keep theirs beside their tools)",
    "semantic_layer": LIBRARY + " (a workspace's approved metric definitions, 30.20; the main product's analytics keep their own layer and pass it as is)",
    "metric_answer": LIBRARY + " (the answer to why a number changed, with its queries, 30.20; stored by the host beside the question)",
    "workflow_run": LIBRARY + " (the twelve steps read from a run's records, 30.15; the main product's pipeline runs map onto it field by field later)",
    "solution_brief": LIBRARY + " (G3: signed before training, its bar judged at the holdout read, 30.14; the main product's Objective has no bar to map it onto yet)",
    "coordinator_situation": LIBRARY + " (where a conversation stands on the way to a build)",
    "solution_workflow": LIBRARY + " (the workflow drawn from the two cards)",
    "solution_report": LIBRARY + " (rendered from the records of a project)",
    "twin_step": "dclab_core.adapters.main_product_twin:twin_steps",
    "twin_matrix": LIBRARY + " (``twin.effectiveness`` over adapted twin steps)",
    "evaluation_board": LIBRARY + " (scores over adapted twin steps and decision records)",
    "orchestra_board": LIBRARY + " (28.2: a team's board, kept by whatever orchestrates the team; the main product has no team of agents yet)",
    "team_record": LIBRARY + " (28.2: a team's walk, node by node; the main product has no team of agents yet)",
    "audit_verdict": LIBRARY + " (28.5: the auditor's screen before a holdout is read; the main product's experiments read their holdout without one yet)",
    "business_example": LIBRARY + " (30.5: knowledge for the understanding and planning agents; the main product has no store of examples yet)",
    "recipe_part": LIBRARY + " (30.10: a task kind or a modifier as one contract; the main product's task kinds and its expansion pack can be read as parts the same way)",
    "resolved_recipe": LIBRARY + " (30.10: the choices one run resolves to, each with its part and reason; the main product's training plan has no such record yet)",
    "second_opinion": LIBRARY + " (28.6: the judge on two readings of a card's family; the main product reads a problem with one reader and keeps no such record)",
    "debate": LIBRARY + " (28.6: two solution designers and a judge over a plan; the main product plans with one designer and keeps no such record)",
    "expert_trace": LIBRARY + " (an AI-first agent's record of a problem and its solution; the main product's Jev answers single questions and writes none)",
    "decision_sample": LIBRARY + " (built from decision records: the main product's semantic_decision_answers reach it through adapters/main_product_ledger)",
    "training_example": LIBRARY + " (built by the host's lake builders from its own consented records)",
    "gold_manifest": LIBRARY + " (written beside each gold set by the host's lake builder)",
}

#: the contracts whose adapter READS the main product's rows into the library's record; every other adapter above maps a library
#: contract OUT to the main product's own (a ``ProblemSpec`` proposal, a pipeline version): both are adapters, in different directions
FROM_ROWS = frozenset({"experience_event", "run_record", "decision_record", "decision_label", "outcome", "twin_step"})

_NO_NAME = "the main product has no governed decision point of this name: it is this studio's {what}; its decisions of the kind are project_decision_records of other types, read as decision events"
POINTS_NOT_MAPPED: dict[str, str] = {
    "request.intent": "H2: the studio's Home reads the kind of help (prediction, analysis, ...) before anything else; the main product has no such front door yet",
    "request.exit": "H8: where a request leaves the studio's Home (an ML project, answers, a prepared table, a dashboard, a decision brief, a cited answer); the main product has no such front door yet",
    "recipe.step": "H9: a guarded preparation step on the studio's Home, labelled by the person's yes or no; the main product has no such front door yet",
    "query.plan": "S4: how the studio's Home computed a question about the person's table, labelled by the answer they kept; the main product has no such front door yet",
    "column.statistics_only": "S4: the studio's Home asks the person about a column its audit kept out on statistics alone (H3.1); the main product's leakage review is feature.leakage_suspect",
    "department.proposal": "the main product has no department agents yet: its governed point is added with them (30.22)",
    **{p: _NO_NAME.format(what="data preparation step") for p in (
        "column.category_code", "column.time", "column.group", "task.framing", "labels.situation", "labels.join_key", "data.cases_table",
        "data.cleaning_step", "data.sampling", "split.folds", "data.structure_format", "data.cleaning_keep", "data.synthetic_shape",
        "data.fit_verdict")},
    **{p: _NO_NAME.format(what="evidence search (29.6)") for p in ("evidence.records",)},
    **{p: _NO_NAME.format(what="data value question (34.13)") for p in ("data.more_rows",)},
    **{p: _NO_NAME.format(what="modelling choice") for p in (
        "features.recipe", "tuning.accept", "holdout.reuse", "experiment.next", "model.first")},  # 0.112.0: model.first (S7)
    **{p: _NO_NAME.format(what="conversation and workflow point (the understanding agents, the coordinator)") for p in (
        "problem.family", "problem.pack", "map.size", "map.edit", "solution.route", "solution.compose", "workflow.next_question", "workflow.template", "workflow.move", "workflow.revisit",
        "workflow.stage_approval", "workflow.stop", "workflow.g4", "agent.tool_choice", "team.next_node", "critic.send_back")},
    **{p: _NO_NAME.format(what="reporting and learning point") for p in ("report.claim", "lesson.accept")},
    **{p: _NO_NAME.format(what="image problem's check (26.8)") for p in ("image.near_copy", "image.label_suspect", "image.flip_safe")},
    **{p: _NO_NAME.format(what="recipe modifier (30.11)") for p in ("modifier.rare_positives", "modifier.groups", "modifier.time_order",
                                                                         "modifier.noisy_labels", "modifier.review_capacity")},
}


def _resolves(target: str) -> bool:
    module, _, name = target.partition(":")
    return module in ADAPTERS and hasattr(ADAPTERS[module], name)


def problems() -> list[str]:
    """What is not conformant: a contract with neither an adapter nor a reason; an adapter that does not exist; an event kind neither
    produced nor listed as missing with a reason; a decision point with neither a main-product name nor a reason; an id of the record
    the adapters neither fill nor list as missing; a field the adapters name that the lists read at the pinned commit do not."""
    import dclab_core.schemas  # noqa: F401  (registers every contract)
    from dclab_core import experience as x
    from dclab_core.adapters import main_product_experience as mp
    from dclab_core.adapters.main_product import recorded
    from dclab_core.contracts import CONTRACTS
    from dclab_core.adapters.main_product_ledger import DECISION_TYPE_POINTS
    from dclab_core.ledger import POINTS

    out: list[str] = []
    for name in sorted(CONTRACTS):
        found = CONTRACT_ADAPTERS.get(name)
        if not found:
            out.append(f"contract {name}: no adapter and no reason (add it to CONTRACT_ADAPTERS)")
        elif not found.startswith("library") and not _resolves(found):
            out.append(f"contract {name}: the adapter {found} does not exist (a value is an existing module:function, or a reason that starts with 'library output')")
    out += [f"CONTRACT_ADAPTERS names {n}, which is not a contract" for n in sorted(set(CONTRACT_ADAPTERS) - set(CONTRACTS))]
    out += [f"FROM_ROWS names {n}, which has no adapter" for n in sorted(FROM_ROWS) if n not in CONTRACT_ADAPTERS or CONTRACT_ADAPTERS[n].startswith("library")]
    produced = produced_kinds()
    for kind in sorted(set(x.KINDS)):
        if kind not in produced and not (mp.KINDS_MISSING.get(kind) or "").strip():
            out.append(f"event kind {kind}: neither produced by the adapter nor listed in KINDS_MISSING with a reason")
    out += [f"KINDS_MISSING names {k}, which is not a kind" for k in sorted(set(mp.KINDS_MISSING) - set(x.KINDS))]
    through_records = set(DECISION_TYPE_POINTS.values())  # read as ``project_decision_records`` by ``main_product_ledger.from_decision_record``
    for key, point in sorted(POINTS.items()):
        mapped = bool(point.main_product) or key in through_records
        if not mapped and not (POINTS_NOT_MAPPED.get(key) or "").strip():
            out.append(f"decision point {key}: no main-product name, no decision record type and no reason (add it to POINTS_NOT_MAPPED)")
        if mapped and key in POINTS_NOT_MAPPED:
            out.append(f"decision point {key} is both mapped and listed as not mapped")
    out += [f"POINTS_NOT_MAPPED names {k}, which is not a point" for k in sorted(set(POINTS_NOT_MAPPED) - set(POINTS))]
    ids = set(x.Ids.model_fields) - {"workspace"}
    filled = {"user", "project", "run", "job", "conversation", "model_request"}  # what the adapter's events carry (``from_agent_event``, ``from_llm_invocation``, ...)
    for ident in sorted(ids - filled):
        if not (mp.IDS_MISSING.get(ident) or "").strip():
            out.append(f"id {ident}: neither filled by the adapter nor listed in IDS_MISSING")
    out += [f"IDS_MISSING names {k}, which is not an id" for k in sorted(set(mp.IDS_MISSING) - ids)]
    tables = recorded()["experience_tables"]
    for table in ("agent_runs", "agent_events", "llm_invocations", "project_decision_records", "ml_run_events", "experiments", "pipeline_stage_runs"):
        if table not in tables:
            out.append(f"the field lists read at {recorded().get('commit')} no longer name the table {table}: read the main product again")
    return out


def produced_kinds() -> set[str]:
    """The event kinds an adapter function writes (its kind tables, the stage boundaries, the run record and the feedback)."""
    from dclab_core.adapters import main_product_experience as mp

    return set(mp.AGENT_EVENT_KINDS.values()) | set(mp.DECISION_KINDS.values()) | {"stage.started", "stage.finished", "stage.failed", "run.recorded", "feedback.given"}


def summary() -> dict[str, Any]:
    """Counts for a page or a log line: how much is mapped and how much is left with a reason. A kind an adapter produces is not
    counted as missing even when its note says what the adapter cannot fill (``feedback.given``)."""
    from dclab_core import experience as x
    from dclab_core.adapters import main_product_experience as mp
    from dclab_core.adapters.main_product_ledger import DECISION_TYPE_POINTS
    from dclab_core.contracts import CONTRACTS
    from dclab_core.ledger import POINTS

    adapted = [name for name, v in CONTRACT_ADAPTERS.items() if not v.startswith("library")]
    return {"contracts": len(CONTRACTS), "contracts_read_from_rows": len([n for n in adapted if n in FROM_ROWS]),
            "contracts_mapped_out": len([n for n in adapted if n not in FROM_ROWS]), "contracts_library_only": len(CONTRACTS) - len(adapted),
            "kinds": len(x.KINDS), "kinds_produced": len(produced_kinds() & set(x.KINDS)), "kinds_missing": len(set(mp.KINDS_MISSING) - produced_kinds()),
            "points": len(POINTS), "points_not_mapped": len(POINTS_NOT_MAPPED), "points_through_records": len(set(DECISION_TYPE_POINTS.values()) & set(POINTS)),
            "commit": main_product.recorded().get("commit")}


__all__ = ["CONTRACT_ADAPTERS", "FROM_ROWS", "POINTS_NOT_MAPPED", "problems", "produced_kinds", "summary"]
