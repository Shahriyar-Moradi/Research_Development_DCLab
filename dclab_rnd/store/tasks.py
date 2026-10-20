"""Which capture rules each of the studio's agents is checked by (32.1): every agent of the harness records its steps
(``agent_step``) and the tasks it does, each with its own rule in ``dclab_core.store_rules``. A test reads the harness, so a
new agent without its tasks here fails the suite."""

from __future__ import annotations

AGENT_TASKS: dict[str, tuple[str, ...]] = {
    "home": ("agent_step", "conversation_turn", "decision", "model_request"),
    "home.script": ("agent_step", "conversation_turn", "decision"),
    "intern": ("agent_step", "intern_session", "conversation_turn", "model_request"),
    "intern.standard": ("agent_step", "intern_session"),
    "workspace_chat": ("agent_step", "conversation_turn", "model_request", "audit_row"),
    "mcp": ("agent_step",),
    "team": ("agent_step", "team_walk", "decision"),
    "orchestra": ("agent_step", "team_walk"),
    "orchestra.walk": ("agent_step", "team_walk", "decision"),
    "orchestra.solution": ("agent_step", "team_walk", "stage_run"),
    "campaign": ("agent_step", "stage_run", "model_request"),
    "coordinator": ("agent_step", "decision", "graph_move"),
    "twin": ("agent_step", "conversation_turn", "teacher_answer", "model_request"),
    "conductor": ("agent_step",),
    "tutor": ("agent_step", "student_version"),
}

__all__ = ["AGENT_TASKS"]
