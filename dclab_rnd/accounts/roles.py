"""The four roles and what each may do (package 10.2). Stored as owner, data_scientist, reviewer, viewer; shown with
the Admin page's names (the owner's choice, 2026-10-07): Owner, ML engineer, Reviewer, Business viewer. The intern
is the agent, not a person's role.

    read           every page and every question (the Evidence and project "ask" too)
    write          projects, data, solutions, stage runs, drafts, the intern, jobs (Stop, Retry)
    approve_stage  confirm a stage's choice (a model, a feature set)
    approve_gate   approve a gate: the solution sign-off, opening the holdout
    review         accept or reject a workspace lesson; move a model purpose to another tier
    admin          gate switches, the training opt-in, members and their roles
    timelines      read the record of every session: what people wrote, what the agents answered (19.4)
    audit          read the audit trail (every member with accounts; behind the Admin sign-in on a machine without them)
    contribute        do a team's work on a shared problem: take a task, act on it, report back (W2 of the collaborative workspace)
    approve_decision  accept a decision a team's agent proposes for the business (a price, a launch, a target)
    contact_customers let an agent reach a customer for the person (an email, a call list); no role has it by default

A role is not a team. A role says what a person may do; a team (storage/teams.py) says whom an agent represents, and a
person may be in several teams whatever their role.
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
    "audit": frozenset(ROLES),  # 2026-11-03: reading the audit trail: every member with accounts; the Admin sign-in's without them
    "timelines": frozenset({"owner"}),  # the owner's choice (confirmed 2026-10-15): owner and admin, which is the owner here
    # W2 of the collaborative workspace (2026-11-20): what a person may do in the teams' shared work
    "contribute": frozenset({"owner", "data_scientist"}),
    "approve_decision": frozenset({"owner"}),
    "contact_customers": frozenset(),  # nobody by default: reaching a customer is granted, never assumed
}


def allows(role: str | None, permission: str) -> bool:
    return (role or "anonymous") in PERMISSIONS[permission]


def label(role: str | None) -> str:
    return LABELS.get(role or "", "Signed out")
