"""One account per identity at the company's identity provider (package 10.2, part B).

Revision ID: 0013
Revises: 0012
"""

import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("uq_app_user_subject", "app_user", ["subject"], unique=True, postgresql_where=sa.text("subject is not null"))


def downgrade() -> None:
    op.drop_index("uq_app_user_subject", table_name="app_user")
