"""What each model request cost, when its price is known (package A1.2).

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_request", sa.Column("cost_eur", sa.Float))
    op.add_column("model_request", sa.Column("cost_basis", sa.Text))


def downgrade() -> None:
    op.drop_column("model_request", "cost_basis")
    op.drop_column("model_request", "cost_eur")
