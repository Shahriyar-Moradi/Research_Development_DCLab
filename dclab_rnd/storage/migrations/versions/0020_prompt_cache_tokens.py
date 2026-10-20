"""The prompt cache's tokens on the model gateway's usage log (prompt caching on Anthropic's Messages API).

``model_request.cache_read_tokens`` and ``cache_write_tokens``: of a request's ``input_tokens``, those read from the
prompt cache and those written to it, so its cost can be checked against its rates. Nullable: rows written before carry
none.

Revision ID: 0020
Revises: 0019
"""

import sqlalchemy as sa
from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_request", sa.Column("cache_read_tokens", sa.Integer))
    op.add_column("model_request", sa.Column("cache_write_tokens", sa.Integer))


def downgrade() -> None:
    op.drop_column("model_request", "cache_write_tokens")
    op.drop_column("model_request", "cache_read_tokens")
