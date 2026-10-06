"""The four roles and what each may do (package 10.2). Stored as owner, data_scientist, reviewer, viewer; shown with
the Admin page's names (the owner's choice, 2026-10-07): Owner, ML engineer, Reviewer, Business viewer. The intern
is the agent, not a person's role.

    read           every page and every question (the Evidence and project "ask" too)
    write          projects, data, solutions, stage runs, drafts, the intern, jobs (Stop, Retry)
    approve_stage  confirm a stage's choice (a model, a feature set)
    approve_gate   approve a gate: the solution sign-off, opening the holdout
    review         accept or reject a workspace lesson; move a model purpose to another tier
    admin          gate switches, the training opt-in, members and their roles
"""

from __future__ import annotations

ROLES = ("owner", "data_scientist", "reviewer", "viewer")
LABELS = {"owner": "Owner", "data_scientist": "ML engineer", "reviewer": "Reviewer", "viewer": "Business viewer"}

PERMISSIONS: dict[str, frozenset[str]] = {
    "public": frozenset({*ROLES, "anonymous"}),
    "read": frozenset(ROLES),
    "write": frozenset({"owner", "data_scientist"}),
    "approve_stage": frozenset({"owner", "data_scientist", "reviewer"}),
    "approve_gate": frozenset({"owner", "reviewer"}),
    "review": frozenset({"owner", "reviewer"}),
    "admin": frozenset({"owner"}),
}


def allows(role: str | None, permission: str) -> bool:
    return (role or "anonymous") in PERMISSIONS[permission]


def label(role: str | None) -> str:
    return LABELS.get(role or "", "Signed out")
