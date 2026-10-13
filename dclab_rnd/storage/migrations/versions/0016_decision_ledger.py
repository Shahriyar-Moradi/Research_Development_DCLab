"""The decision ledger: every decision DCLab makes, and the labels that say what was right (package 19.7).

Each ``decision_record`` row is a ``dclab_core.decision_record.v1`` document with what finds it in columns (the point,
its kind, who decided, the project, draft, run and conversation); each ``decision_label`` row a
``dclab_core.decision_label.v1``. Both are append-only (a trigger refuses UPDATE and DELETE): a correction is a new
label superseding the old, never an edit.

Revision ID: 0016
Revises: 0015
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None

TABLES = ("decision_record", "decision_label")


def upgrade() -> None:
    op.create_table("decision_record",
                    sa.Column("id", sa.String(80), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", name="fk_decision_record_workspace_id_workspace"), nullable=False),
                    sa.Column("at", sa.Text, nullable=False), sa.Column("point", sa.Text, nullable=False), sa.Column("kind", sa.Text, nullable=False),
                    sa.Column("used", sa.Text, nullable=False), sa.Column("project_id", sa.String(64)), sa.Column("draft_id", sa.String(64)),
                    sa.Column("run_id", sa.String(80)), sa.Column("conversation", sa.String(80)), sa.Column("doc", JSONB, nullable=False))
    for column in ("at", "point", "project_id", "draft_id", "run_id"):
        op.create_index(f"ix_decision_record_workspace_id_{column}", "decision_record", ["workspace_id", column])
    op.create_table("decision_label",
                    sa.Column("id", sa.String(80), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", name="fk_decision_label_workspace_id_workspace"), nullable=False),
                    sa.Column("at", sa.Text, nullable=False), sa.Column("record_id", sa.String(80), nullable=False), sa.Column("point", sa.Text, nullable=False),
                    sa.Column("source", sa.Text, nullable=False), sa.Column("supersedes", sa.String(80)), sa.Column("doc", JSONB, nullable=False))
    for column in ("record_id", "point", "source"):
        op.create_index(f"ix_decision_label_workspace_id_{column}", "decision_label", ["workspace_id", column])
    for table in TABLES:
        op.execute(f"""
            create function {table}_append_only() returns trigger language plpgsql as $$
            begin
                raise exception '{table} is append-only: % is not allowed', tg_op;
            end $$;
        """)
        op.execute(f"create trigger {table}_append_only before update or delete on {table} for each row execute function {table}_append_only()")


def downgrade() -> None:
    for table in TABLES:
        op.execute(f"drop trigger if exists {table}_append_only on {table}")
        op.execute(f"drop function if exists {table}_append_only()")
        op.drop_table(table)
