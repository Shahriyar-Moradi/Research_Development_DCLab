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
    "coordinator_situation": LIBRARY + " (where a conversation stands on the way to a build)",
    "solution_workflow": LIBRARY + " (the workflow drawn from the two cards)",
    "solution_report": LIBRARY + " (rendered from the records of a project)",
    "twin_step": "dclab_core.adapters.main_product_twin:twin_steps",
    "twin_matrix": LIBRARY + " (``twin.effectiveness`` over adapted twin steps)",
}

#: the contracts whose adapter READS the main product's rows into the library's record; every other adapter above maps a library
#: contract OUT to the main product's own (a ``ProblemSpec`` proposal, a pipeline version): both are adapters, in different directions
FROM_ROWS = frozenset({"experience_event", "run_record", "decision_record", "decision_label", "outcome", "twin_step"})

_NO_NAME = "the main product has no governed decision point of this name: it is this studio's {what}; its decisions of the kind are project_decision_records of other types, read as decision events"
POINTS_NOT_MAPPED: dict[str, str] = {
    **{p: _NO_NAME.format(what="data preparation step") for p in (
        "column.category_code", "column.time", "column.group", "task.framing", "labels.situation", "labels.join_key", "data.cases_table",
        "data.cleaning_step", "data.sampling", "split.folds")},
    **{p: _NO_NAME.format(what="modelling choice") for p in (
        "features.recipe", "tuning.accept", "holdout.reuse", "experiment.next")},
    **{p: _NO_NAME.format(what="conversation and workflow point (the understanding agents, the coordinator)") for p in (
        "problem.family", "problem.pack", "map.size", "map.edit", "solution.route", "solution.compose", "workflow.next_question", "workflow.template", "workflow.move", "workflow.revisit",
        "workflow.stage_approval", "workflow.stop", "agent.tool_choice")},
    **{p: _NO_NAME.format(what="reporting and learning point") for p in ("report.claim", "lesson.accept")},
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
