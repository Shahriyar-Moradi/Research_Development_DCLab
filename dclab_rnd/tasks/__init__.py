"""Non-executing task previews and typed catalog planning. No scheduler or runtime entry point."""

from .adapter import adapt_workflow, capability_snapshot
from .catalog import (
    compile_catalog_to_task_graph,
    deterministic_catalog_plan,
    execute_catalog_task,
    needs_model_planning,
    plan_with_model,
    project_catalog_inputs,
    replan_from_evidence,
)

__all__ = [
    "adapt_workflow",
    "capability_snapshot",
    "compile_catalog_to_task_graph",
    "deterministic_catalog_plan",
    "execute_catalog_task",
    "needs_model_planning",
    "plan_with_model",
    "project_catalog_inputs",
    "replan_from_evidence",
]
