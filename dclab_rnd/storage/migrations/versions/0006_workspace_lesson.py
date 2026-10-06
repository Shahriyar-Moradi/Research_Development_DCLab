"""Lessons from finished projects, reviewed by a person (package A5.3).

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("workspace_lesson", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_workspace_lesson_workspace_id_workspace"), nullable=False),
                    sa.Column("project_id", sa.String(64), nullable=False), sa.Column("status", sa.Text, nullable=False),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("created", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
                    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
                    sa.CheckConstraint("status in ('proposed', 'accepted', 'rejected', 'superseded')", name="ck_workspace_lesson_status"))
    op.create_index("ix_workspace_lesson_workspace_id_status", "workspace_lesson", ["workspace_id", "status"])
    op.create_index("ix_workspace_lesson_project_id", "workspace_lesson", ["project_id"])


def downgrade() -> None:
    op.drop_table("workspace_lesson")
