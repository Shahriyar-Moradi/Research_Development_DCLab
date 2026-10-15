"""A trace id on the session record (package 23.11).

``experience_event.trace_id`` is the W3C trace of the request an event belongs to, so one request's events, its jobs' and its model
requests' are one indexed read. A nullable column: events written before carry none. The append-only trigger refuses UPDATE and
DELETE, not a new column.

Revision ID: 0019
Revises: 0018
"""

import sqlalchemy as sa
from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("experience_event", sa.Column("trace_id", sa.String(32)))
    op.create_index("ix_experience_event_workspace_id_trace_id", "experience_event", ["workspace_id", "trace_id"])


def downgrade() -> None:
    op.drop_index("ix_experience_event_workspace_id_trace_id", table_name="experience_event")
    op.drop_column("experience_event", "trace_id")
