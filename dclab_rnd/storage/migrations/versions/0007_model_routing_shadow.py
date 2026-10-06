"""Which tier serves or shadows each purpose, and the shadow log (package A6.4).

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("model_routing",
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_model_routing_workspace_id_workspace"), primary_key=True),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")))
    op.create_table("model_shadow", sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_model_shadow_workspace_id_workspace"), nullable=False),
                    sa.Column("at", sa.Text, nullable=False), sa.Column("purpose", sa.Text, nullable=False),
                    sa.Column("primary_tier", sa.Text), sa.Column("primary_model", sa.Text),
                    sa.Column("shadow_tier", sa.Text, nullable=False), sa.Column("shadow_model", sa.Text, nullable=False),
                    sa.Column("project_id", sa.String(64)), sa.Column("draft_id", sa.String(64)),
                    sa.Column("primary_tool", sa.Text), sa.Column("shadow_tool", sa.Text),
                    sa.Column("same_tool", sa.Boolean), sa.Column("same_arguments", sa.Boolean),
                    sa.Column("outcome", sa.Text, nullable=False),
                    sa.Column("seconds", sa.Float, nullable=False, server_default="0"),
                    sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
                    sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"))
    op.create_index("ix_model_shadow_workspace_at", "model_shadow", ["workspace_id", "at"])


def downgrade() -> None:
    op.drop_table("model_shadow")
    op.drop_table("model_routing")
