"""Memory items (package 35.2): every version of every item an agent may remember, append-only.

Each ``memory_item_version`` row is one ``dclab_core.memory_item.v1`` document with what finds it in columns (the item's
id, kind, scope and scope id, status). A retirement, a counter moved or a correction is a new row; the last row of an
id is its state. A trigger refuses UPDATE and DELETE: deleting a project retires its items, and the rows stay (the
owner's rule of 2026-10-20: the store keeps what is deleted).

Revision ID: 0022
Revises: 0021
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None

TABLE = "memory_item_version"


def upgrade() -> None:
    op.create_table(TABLE,
                    sa.Column("seq", sa.BigInteger, primary_key=True, autoincrement=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", name="fk_memory_item_version_workspace_id_workspace"), nullable=False),
                    sa.Column("item_id", sa.String(16), nullable=False), sa.Column("kind", sa.Text, nullable=False),
                    sa.Column("scope", sa.Text, nullable=False), sa.Column("scope_id", sa.Text, nullable=False), sa.Column("status", sa.Text, nullable=False),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("written", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")))
    op.create_index("ix_memory_item_version_workspace_id_item_id", TABLE, ["workspace_id", "item_id"])
    op.create_index("ix_memory_item_version_workspace_id_scope", TABLE, ["workspace_id", "scope", "scope_id"])
    op.execute(f"""
        create function {TABLE}_append_only() returns trigger language plpgsql as $$
        begin
            raise exception '{TABLE} is append-only: % is not allowed', tg_op;
        end $$;
    """)
    op.execute(f"create trigger {TABLE}_append_only before update or delete on {TABLE} for each row execute function {TABLE}_append_only()")


def downgrade() -> None:
    op.execute(f"drop trigger if exists {TABLE}_append_only on {TABLE}")
    op.execute(f"drop function if exists {TABLE}_append_only()")
    op.drop_table(TABLE)
