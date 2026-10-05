"""One row per step of an agent run: the trace (package A2.3).

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("agent_step", sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_agent_step_workspace_id_workspace"), nullable=False),
                    sa.Column("run_id", sa.String(64), nullable=False), sa.Column("agent", sa.Text, nullable=False),
                    sa.Column("n", sa.Integer, nullable=False), sa.Column("reply", sa.Integer), sa.Column("at", sa.Text, nullable=False),
                    sa.Column("state", sa.Text), sa.Column("tool", sa.Text, nullable=False), sa.Column("arguments", JSONB, nullable=False),
                    sa.Column("verdict", sa.Text, nullable=False), sa.Column("result", sa.Text),
                    sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
                    sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
                    sa.Column("seconds", sa.Float, nullable=False, server_default="0"))
    op.create_index("ix_agent_step_workspace_id_run_id", "agent_step", ["workspace_id", "run_id"])


def downgrade() -> None:
    op.drop_table("agent_step")
