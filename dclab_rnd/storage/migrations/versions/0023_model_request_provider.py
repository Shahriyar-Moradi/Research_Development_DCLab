"""Who served a model request, on the gateway's usage log (2026-11-11: DCLab moved from Claude to OpenRouter Qwen).

``model_request.provider`` (anthropic, openrouter, openai, openai_compatible, local) and ``source_class``
(commercial_model, open_weights_model, unknown), read from the endpoint's host by ``dclab_rnd.models.gateway.source_of``,
so training sets can keep each source apart. Nullable: rows written before carry none.

Revision ID: 0023
Revises: 0022
"""

import sqlalchemy as sa
from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_request", sa.Column("provider", sa.Text))
    op.add_column("model_request", sa.Column("source_class", sa.Text))


def downgrade() -> None:
    op.drop_column("model_request", "source_class")
    op.drop_column("model_request", "provider")
