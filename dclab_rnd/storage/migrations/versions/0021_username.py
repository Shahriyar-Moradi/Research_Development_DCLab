"""A username for an account, and an email that is optional.

``app_user.username``: a handle (3 to 32 characters: letters, digits, dot, dash, underscore; kept lower-case) that is unique
and can sign a person in; ``app_user.email`` may now be empty (a person with a username alone). Existing users keep their
email and have no username until one is set.

Revision ID: 0021
Revises: 0020
"""

import sqlalchemy as sa
from alembic import op

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("app_user", sa.Column("username", sa.Text))
    op.create_unique_constraint("uq_app_user_username", "app_user", ["username"])
    op.alter_column("app_user", "email", existing_type=sa.Text, nullable=True)


def downgrade() -> None:
    op.execute("update app_user set email = username || '@users.invalid' where email is null")  # an email again: a placeholder, never an address
    op.alter_column("app_user", "email", existing_type=sa.Text, nullable=False)
    op.drop_constraint("uq_app_user_username", "app_user", type_="unique")
    op.drop_column("app_user", "username")
