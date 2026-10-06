"""Sessions and API tokens for accounts (package 10.2): only the tokens' SHA-256 is stored.

Revision ID: 0012
Revises: 0011
"""

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("user_session",
                    sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("user_id", sa.String(64), sa.ForeignKey("app_user.id", ondelete="CASCADE", name="fk_user_session_user_id_app_user"), nullable=False),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_user_session_workspace_id_workspace"), nullable=False),
                    sa.Column("csrf", sa.Text, nullable=False), sa.Column("started", sa.Float, nullable=False),
                    sa.Column("last_seen", sa.Float, nullable=False), sa.Column("expires", sa.Float, nullable=False))
    op.create_index("ix_user_session_user_id", "user_session", ["user_id"])
    op.create_table("api_token",
                    sa.Column("id", sa.String(32), primary_key=True),
                    sa.Column("user_id", sa.String(64), sa.ForeignKey("app_user.id", ondelete="CASCADE", name="fk_api_token_user_id_app_user"), nullable=False),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_api_token_workspace_id_workspace"), nullable=False),
                    sa.Column("name", sa.Text, nullable=False), sa.Column("token_hash", sa.String(64), nullable=False),
                    sa.Column("prefix", sa.Text, nullable=False), sa.Column("created", sa.Float, nullable=False),
                    sa.Column("last_used", sa.Float), sa.Column("revoked", sa.Boolean, nullable=False, server_default=sa.false()),
                    sa.UniqueConstraint("token_hash", name="uq_api_token_token_hash"))
    op.create_index("ix_api_token_user_id", "api_token", ["user_id"])


def downgrade() -> None:
    op.drop_table("api_token")
    op.drop_table("user_session")
