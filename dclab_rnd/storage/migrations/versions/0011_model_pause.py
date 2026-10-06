"""The model pause as a row with an expiry, shared by every app instance and worker (package 10.5).

Revision ID: 0011
Revises: 0010
"""

import sqlalchemy as sa
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("model_pause", sa.Column("key", sa.Text, primary_key=True), sa.Column("until", sa.Float, nullable=False),
                    sa.Column("reason", sa.Text, nullable=False))


def downgrade() -> None:
    op.drop_table("model_pause")
