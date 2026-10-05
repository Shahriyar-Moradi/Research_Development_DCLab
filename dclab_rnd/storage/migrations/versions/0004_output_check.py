"""Whether a model's output passed the code that checks it (package A1.3).

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("model_output_check", sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_model_output_check_workspace_id_workspace"), nullable=False),
                    sa.Column("at", sa.Text, nullable=False), sa.Column("purpose", sa.Text, nullable=False), sa.Column("tier", sa.Text, nullable=False),
                    sa.Column("model", sa.Text, nullable=False), sa.Column("passed", sa.Boolean, nullable=False), sa.Column("reason", sa.Text))
    op.create_index("ix_model_output_check_workspace_at", "model_output_check", ["workspace_id", "at"])


def downgrade() -> None:
    op.drop_table("model_output_check")
