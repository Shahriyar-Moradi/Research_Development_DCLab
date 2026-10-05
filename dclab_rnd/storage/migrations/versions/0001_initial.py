"""The first schema: workspaces, users, projects, drafts, intern sessions and the three append-only logs.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TIME = sa.DateTime(timezone=True)
NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table("workspace", sa.Column("id", sa.String(64), primary_key=True), sa.Column("key", sa.Text, nullable=False),
                    sa.Column("name", sa.Text, nullable=False), sa.Column("created", TIME, nullable=False, server_default=NOW),
                    sa.UniqueConstraint("key", name="uq_workspace_key"))
    op.create_table("app_user", sa.Column("id", sa.String(64), primary_key=True), sa.Column("email", sa.Text, nullable=False),
                    sa.Column("name", sa.Text, nullable=False, server_default=""), sa.Column("password_hash", sa.Text), sa.Column("subject", sa.Text),
                    sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()), sa.Column("created", TIME, nullable=False, server_default=NOW),
                    sa.UniqueConstraint("email", name="uq_app_user_email"))
    op.create_table("membership",
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_membership_workspace_id_workspace"), primary_key=True),
                    sa.Column("user_id", sa.String(64), sa.ForeignKey("app_user.id", ondelete="CASCADE", name="fk_membership_user_id_app_user"), primary_key=True),
                    sa.Column("role", sa.String(32), nullable=False),
                    sa.CheckConstraint("role in ('owner', 'data_scientist', 'reviewer', 'viewer')", name="ck_membership_role"))
    op.create_table("project", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_project_workspace_id_workspace"), nullable=False),
                    sa.Column("name", sa.Text, nullable=False), sa.Column("industry", sa.Text, nullable=False), sa.Column("doc", JSONB, nullable=False),
                    sa.Column("created", TIME, nullable=False, server_default=NOW), sa.Column("updated", TIME, nullable=False, server_default=NOW))
    op.create_index("ix_project_workspace_updated", "project", ["workspace_id", sa.text("updated DESC")])
    op.create_table("stage_record",
                    sa.Column("project_id", sa.String(64), sa.ForeignKey("project.id", ondelete="CASCADE", name="fk_stage_record_project_id_project"), primary_key=True),
                    sa.Column("stage", sa.String(16), primary_key=True), sa.Column("record", JSONB, nullable=False),
                    sa.Column("written", TIME, nullable=False, server_default=NOW),
                    sa.CheckConstraint("stage in ('data', 'leakage', 'features', 'models', 'final')", name="ck_stage_record_stage"))
    op.create_table("activity", sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
                    sa.Column("project_id", sa.String(64), sa.ForeignKey("project.id", ondelete="CASCADE", name="fk_activity_project_id_project"), nullable=False),
                    sa.Column("at", sa.Text, nullable=False), sa.Column("kind", sa.Text, nullable=False), sa.Column("payload", JSONB))
    op.create_index("ix_activity_project_id", "activity", ["project_id", "id"])
    op.create_table("transition", sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
                    sa.Column("project_id", sa.String(64), sa.ForeignKey("project.id", ondelete="CASCADE", name="fk_transition_project_id_project"), nullable=False),
                    sa.Column("at", sa.Text), sa.Column("move", sa.Text), sa.Column("status", sa.Text), sa.Column("actor", sa.Text),
                    sa.Column("entry", JSONB, nullable=False))
    op.create_index("ix_transition_project_id", "transition", ["project_id", "id"])
    op.create_table("draft", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_draft_workspace_id_workspace"), nullable=False),
                    sa.Column("status", sa.Text, nullable=False), sa.Column("doc", JSONB, nullable=False),
                    sa.Column("event_seq", sa.BigInteger, nullable=False, server_default="0"),
                    sa.Column("created", TIME, nullable=False, server_default=NOW), sa.Column("updated", TIME, nullable=False, server_default=NOW))
    op.create_index("ix_draft_workspace_updated", "draft", ["workspace_id", sa.text("updated DESC")])
    op.create_table("draft_event",
                    sa.Column("draft_id", sa.String(64), sa.ForeignKey("draft.id", ondelete="CASCADE", name="fk_draft_event_draft_id_draft"), primary_key=True),
                    sa.Column("seq", sa.BigInteger, primary_key=True), sa.Column("at", sa.Text, nullable=False), sa.Column("kind", sa.Text, nullable=False),
                    sa.Column("data", JSONB, nullable=False))
    op.create_table("intern_session", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_intern_session_workspace_id_workspace"), nullable=False),
                    sa.Column("status", sa.Text, nullable=False), sa.Column("doc", JSONB, nullable=False),
                    sa.Column("created", TIME, nullable=False, server_default=NOW), sa.Column("updated", TIME, nullable=False, server_default=NOW))
    op.create_index("ix_intern_session_workspace_updated", "intern_session", ["workspace_id", sa.text("updated DESC")])


def downgrade() -> None:
    for table in ("intern_session", "draft_event", "draft", "transition", "activity", "stage_record", "project", "membership", "app_user", "workspace"):
        op.drop_table(table)
