"""The product's PostgreSQL schema (package 9.2).

A document that is always read whole (a project, a stage record, a draft, an intern session) is one JSONB column.
What a list filters, sorts or joins on is a real column, with an index. The three append-only logs (activity,
transition, draft_event) are rows that are only ever inserted.

Everything belongs to a workspace. Until accounts exist (package 10.2) the product runs in one workspace per
workspace folder (``DCLAB_AGENT_HOME``); the users and memberships tables are here so that step adds no table.
``migrations/versions`` holds the changes; ``tests/test_storage_postgres.py`` fails when they and this file disagree.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

metadata = sa.MetaData(naming_convention={
    "ix": "ix_%(table_name)s_%(column_0_N_name)s", "pk": "pk_%(table_name)s", "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s", "ck": "ck_%(table_name)s_%(constraint_name)s"})

TIME = sa.DateTime(timezone=True)
NOW = sa.text("now()")


def _id(name: str = "id") -> sa.Column:
    return sa.Column(name, sa.String(64), primary_key=True)


workspace = sa.Table(
    "workspace", metadata, _id(),
    sa.Column("key", sa.Text, nullable=False),  # where it lives: a folder today, a slug later
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("created", TIME, nullable=False, server_default=NOW),
    sa.UniqueConstraint("key", name="uq_workspace_key"))

app_user = sa.Table(
    "app_user", metadata, _id(),
    sa.Column("email", sa.Text, nullable=False),
    sa.Column("name", sa.Text, nullable=False, server_default=""),
    sa.Column("password_hash", sa.Text),  # null for single-sign-on users
    sa.Column("subject", sa.Text),  # the identity provider's id for the user
    sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
    sa.Column("created", TIME, nullable=False, server_default=NOW),
    sa.UniqueConstraint("email", name="uq_app_user_email"))

membership = sa.Table(
    "membership", metadata,
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("user_id", sa.String(64), sa.ForeignKey("app_user.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("role", sa.String(32), nullable=False),
    sa.CheckConstraint("role in ('owner', 'data_scientist', 'reviewer', 'viewer')", name="role"))

project = sa.Table(
    "project", metadata, _id(),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("industry", sa.Text, nullable=False),
    sa.Column("doc", JSONB, nullable=False),  # the whole project document
    sa.Column("created", TIME, nullable=False, server_default=NOW),
    sa.Column("updated", TIME, nullable=False, server_default=NOW),
    sa.Index("ix_project_workspace_updated", "workspace_id", sa.text("updated DESC")))

stage_record = sa.Table(
    "stage_record", metadata,
    sa.Column("project_id", sa.String(64), sa.ForeignKey("project.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("stage", sa.String(16), primary_key=True),
    sa.Column("record", JSONB, nullable=False),
    sa.Column("written", TIME, nullable=False, server_default=NOW),
    sa.CheckConstraint("stage in ('data', 'leakage', 'features', 'models', 'final')", name="stage"))

activity = sa.Table(
    "activity", metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("project_id", sa.String(64), sa.ForeignKey("project.id", ondelete="CASCADE"), nullable=False),
    sa.Column("at", sa.Text, nullable=False),  # the entry's own timestamp, as written by the application
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("payload", JSONB),
    sa.Index("ix_activity_project_id", "project_id", "id"))

transition = sa.Table(
    "transition", metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("project_id", sa.String(64), sa.ForeignKey("project.id", ondelete="CASCADE"), nullable=False),
    sa.Column("at", sa.Text),
    sa.Column("move", sa.Text),
    sa.Column("status", sa.Text),
    sa.Column("actor", sa.Text),
    sa.Column("entry", JSONB, nullable=False),
    sa.Index("ix_transition_project_id", "project_id", "id"))

draft = sa.Table(
    "draft", metadata, _id(),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("doc", JSONB, nullable=False),
    sa.Column("event_seq", sa.BigInteger, nullable=False, server_default="0"),  # last event number; serialises emits per draft
    sa.Column("created", TIME, nullable=False, server_default=NOW),
    sa.Column("updated", TIME, nullable=False, server_default=NOW),
    sa.Index("ix_draft_workspace_updated", "workspace_id", sa.text("updated DESC")))

draft_event = sa.Table(
    "draft_event", metadata,
    sa.Column("draft_id", sa.String(64), sa.ForeignKey("draft.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("seq", sa.BigInteger, primary_key=True),
    sa.Column("at", sa.Text, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("data", JSONB, nullable=False))

intern_session = sa.Table(
    "intern_session", metadata, _id(),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("doc", JSONB, nullable=False),
    sa.Column("created", TIME, nullable=False, server_default=NOW),
    sa.Column("updated", TIME, nullable=False, server_default=NOW),
    sa.Index("ix_intern_session_workspace_updated", "workspace_id", sa.text("updated DESC")))

model_request = sa.Table(  # the model gateway's usage log (dclab_rnd/models/usage.py); one row per request
    "model_request", metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
    sa.Column("at", sa.Text, nullable=False),
    sa.Column("purpose", sa.Text, nullable=False),
    sa.Column("tier", sa.Text, nullable=False),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("endpoint", sa.Text, nullable=False),  # host only: never a key or a full URL with credentials
    sa.Column("project_id", sa.String(64)),
    sa.Column("draft_id", sa.String(64)),
    sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
    sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
    sa.Column("seconds", sa.Float, nullable=False, server_default="0"),
    sa.Column("attempts", sa.Integer, nullable=False, server_default="1"),
    sa.Column("outcome", sa.Text, nullable=False),
    sa.Column("prompt", sa.Text),  # only with DCLAB_LOG_PROMPTS=1
    sa.Column("cost_eur", sa.Float),  # null when the model has no configured price
    sa.Column("cost_basis", sa.Text),  # "price", "local" or "no price" (dclab_rnd/models/prices.py)
    sa.Index("ix_model_request_workspace_at", "workspace_id", "at"),
    sa.Index("ix_model_request_project_id", "project_id"))

model_output_check = sa.Table(  # whether a model's output passed the code that checks it (package A1.3)
    "model_output_check", metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
    sa.Column("at", sa.Text, nullable=False),
    sa.Column("purpose", sa.Text, nullable=False),
    sa.Column("tier", sa.Text, nullable=False),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("passed", sa.Boolean, nullable=False),
    sa.Column("reason", sa.Text),  # why it failed, in words; never the model's output or the user's data
    sa.Index("ix_model_output_check_workspace_at", "workspace_id", "at"))

agent_step = sa.Table(  # one row per step of an agent run (dclab_rnd/agents/traces.py, package A2.3); never a cell value
    "agent_step", metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
    sa.Column("run_id", sa.String(64), nullable=False),  # an intern session id, or a draft id and its turn
    sa.Column("agent", sa.Text, nullable=False),
    sa.Column("n", sa.Integer, nullable=False),
    sa.Column("reply", sa.Integer),  # which model reply asked for the step; null when no model chose it (the standard plan)
    sa.Column("at", sa.Text, nullable=False),
    sa.Column("state", sa.Text),  # a summary of what the agent saw (for a project, the graph state string)
    sa.Column("tool", sa.Text, nullable=False),
    sa.Column("arguments", JSONB, nullable=False),
    sa.Column("verdict", sa.Text, nullable=False),  # ok, error, blocked, needs_approval
    sa.Column("result", sa.Text),  # the tool's one-line summary
    sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
    sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
    sa.Column("seconds", sa.Float, nullable=False, server_default="0"),
    sa.Index("ix_agent_step_workspace_id_run_id", "workspace_id", "run_id"))

workspace_lesson = sa.Table(  # a lesson from a finished project, proposed, then accepted or rejected by a reviewer (dclab_rnd/lessons.py, A5.3)
    "workspace_lesson", metadata, _id(),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
    sa.Column("project_id", sa.String(64), nullable=False),  # kept when the project is deleted: an accepted lesson is evidence on its own
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("doc", JSONB, nullable=False),
    sa.Column("created", TIME, nullable=False, server_default=NOW),
    sa.Column("updated", TIME, nullable=False, server_default=NOW),
    sa.CheckConstraint("status in ('proposed', 'accepted', 'rejected', 'superseded')", name="status"),
    sa.Index("ix_workspace_lesson_workspace_id_status", "workspace_id", "status"),
    sa.Index("ix_workspace_lesson_project_id", "project_id"))

model_routing = sa.Table(  # which tier serves or shadows each purpose, with its history (dclab_rnd/models/routing.py, A6.4); one row per workspace
    "model_routing", metadata,
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("doc", JSONB, nullable=False),
    sa.Column("updated", TIME, nullable=False, server_default=NOW))

model_shadow = sa.Table(  # how a shadow model's answer compared with the served one (dclab_rnd/models/shadow.py, A6.4); never content
    "model_shadow", metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
    sa.Column("at", sa.Text, nullable=False),
    sa.Column("purpose", sa.Text, nullable=False),
    sa.Column("primary_tier", sa.Text), sa.Column("primary_model", sa.Text),
    sa.Column("shadow_tier", sa.Text, nullable=False), sa.Column("shadow_model", sa.Text, nullable=False),
    sa.Column("project_id", sa.String(64)), sa.Column("draft_id", sa.String(64)),
    sa.Column("primary_tool", sa.Text), sa.Column("shadow_tool", sa.Text),
    sa.Column("same_tool", sa.Boolean), sa.Column("same_arguments", sa.Boolean),
    sa.Column("outcome", sa.Text, nullable=False),
    sa.Column("seconds", sa.Float, nullable=False, server_default="0"),
    sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
    sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
    sa.Index("ix_model_shadow_workspace_at", "workspace_id", "at"))

stored_file = sa.Table(  # a table or artifact kept in file storage: where, how big, its hash (dclab_rnd/storage/files.py, 9.3); never a credential
    "stored_file", metadata,
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("key", sa.Text, primary_key=True),  # the path under the workspace folder, made of safe names
    sa.Column("size", sa.BigInteger, nullable=False),
    sa.Column("sha256", sa.String(64), nullable=False),
    sa.Column("content_type", sa.Text, nullable=False),
    sa.Column("backend", sa.Text, nullable=False),  # local or s3
    sa.Column("recorded", sa.Text, nullable=False))

job = sa.Table(  # a piece of background work: a stage run, a data pipeline, synthetic data, an intern turn (dclab_rnd/jobs, 10.3)
    "job", metadata, _id(),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("key", sa.Text, nullable=False),  # what it works on: a project, an intern session, a draft's asset
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("cancel_requested", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("worker", sa.Text),  # the worker that claimed it
    sa.Column("heartbeat", sa.Float),  # seconds since the epoch; a running job whose worker stops beating is interrupted
    sa.Column("created", TIME, nullable=False, server_default=NOW),
    sa.Column("doc", JSONB, nullable=False),  # the whole job: payload, progress, times, error, attempts
    sa.CheckConstraint("status in ('queued', 'running', 'done', 'failed', 'cancelled', 'interrupted')", name="status"),
    sa.Index("ix_job_workspace_id_status_created", "workspace_id", "status", "created"),
    sa.Index("ix_job_workspace_id_kind_key", "workspace_id", "kind", "key"),
    # one queued or running job per key: two app instances cannot start the same project's stages at once
    sa.Index("uq_job_active_key", "workspace_id", "kind", "key", unique=True, postgresql_where=sa.text("status in ('queued', 'running')")))

audit_event = sa.Table(  # every governed action, append-only: a trigger refuses update and delete (dclab_rnd/audit.py, 10.4)
    "audit_event", metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id"), nullable=False),  # no cascade: an audit outlives what it records
    sa.Column("at", sa.Text, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("actor", sa.Text, nullable=False),  # human or agent
    sa.Column("who", sa.Text, nullable=False),  # how the page names them
    sa.Column("user_id", sa.String(64)),  # the signed-in user, once accounts exist (10.2)
    sa.Column("project_id", sa.String(64)),  # kept when the project is deleted
    sa.Column("project_name", sa.Text),
    sa.Column("draft_id", sa.String(64)),
    sa.Column("move", sa.Text),
    sa.Column("status", sa.Text),
    sa.Column("detail", JSONB, nullable=False),
    sa.Index("ix_audit_event_workspace_id_id", "workspace_id", "id"),
    sa.Index("ix_audit_event_workspace_id_kind", "workspace_id", "kind"),
    sa.Index("ix_audit_event_workspace_id_project_id", "workspace_id", "project_id"))

TABLES = [t.name for t in metadata.sorted_tables]
