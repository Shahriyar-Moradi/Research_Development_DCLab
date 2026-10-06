"""Rate limits and quotas (package 10.6): counters per user and workspace, and who each model request was for.

Revision ID: 0014
Revises: 0013
"""

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("usage_counter", sa.Column("scope", sa.Text, primary_key=True), sa.Column("kind", sa.Text, primary_key=True),
                    sa.Column("window", sa.Float, primary_key=True), sa.Column("amount", sa.Float, nullable=False))
    op.add_column("model_request", sa.Column("user_id", sa.String(64)))


def downgrade() -> None:
    op.drop_column("model_request", "user_id")
    op.drop_table("usage_counter")
