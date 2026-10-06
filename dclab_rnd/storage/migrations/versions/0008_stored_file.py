"""A record of each stored table and artifact: key, size, SHA-256, content type (package 9.3).

Revision ID: 0008
Revises: 0007
"""

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("stored_file",
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_stored_file_workspace_id_workspace"), primary_key=True),
                    sa.Column("key", sa.Text, primary_key=True), sa.Column("size", sa.BigInteger, nullable=False),
                    sa.Column("sha256", sa.String(64), nullable=False), sa.Column("content_type", sa.Text, nullable=False),
                    sa.Column("backend", sa.Text, nullable=False), sa.Column("recorded", sa.Text, nullable=False))


def downgrade() -> None:
    op.drop_table("stored_file")
