"""The job table: background work that outlives the process that started it (package 10.3).

Revision ID: 0009
Revises: 0008
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("job",
                    sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_job_workspace_id_workspace"), nullable=False),
                    sa.Column("kind", sa.Text, nullable=False), sa.Column("key", sa.Text, nullable=False),
                    sa.Column("status", sa.Text, nullable=False),
                    sa.Column("cancel_requested", sa.Boolean, nullable=False, server_default=sa.false()),
                    sa.Column("worker", sa.Text), sa.Column("heartbeat", sa.Float),
                    sa.Column("created", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.CheckConstraint("status in ('queued', 'running', 'done', 'failed', 'cancelled', 'interrupted')", name="ck_job_status"))
    op.create_index("ix_job_workspace_id_status_created", "job", ["workspace_id", "status", "created"])
    op.create_index("ix_job_workspace_id_kind_key", "job", ["workspace_id", "kind", "key"])
    op.create_index("uq_job_active_key", "job", ["workspace_id", "kind", "key"], unique=True, postgresql_where=sa.text("status in ('queued', 'running')"))


def downgrade() -> None:
    op.drop_table("job")
