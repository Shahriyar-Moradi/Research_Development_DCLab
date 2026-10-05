"""The model gateway's usage log (package A1.1).

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("model_request", sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_model_request_workspace_id_workspace"), nullable=False),
                    sa.Column("at", sa.Text, nullable=False), sa.Column("purpose", sa.Text, nullable=False), sa.Column("tier", sa.Text, nullable=False),
                    sa.Column("model", sa.Text, nullable=False), sa.Column("endpoint", sa.Text, nullable=False),
                    sa.Column("project_id", sa.String(64)), sa.Column("draft_id", sa.String(64)),
                    sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"), sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
                    sa.Column("seconds", sa.Float, nullable=False, server_default="0"), sa.Column("attempts", sa.Integer, nullable=False, server_default="1"),
                    sa.Column("outcome", sa.Text, nullable=False), sa.Column("prompt", sa.Text))
    op.create_index("ix_model_request_workspace_at", "model_request", ["workspace_id", "at"])
    op.create_index("ix_model_request_project_id", "model_request", ["project_id"])


def downgrade() -> None:
    op.drop_table("model_request")
