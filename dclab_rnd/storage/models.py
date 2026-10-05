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

TABLES = [t.name for t in metadata.sorted_tables]
