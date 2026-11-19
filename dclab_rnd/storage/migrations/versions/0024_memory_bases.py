"""Problem memory's bases and summaries in the database (finding 10 of the problem-memory review).

They were JSON files under the workspace folder behind a per-host file lock, so two API or worker hosts on one
database disagreed. ``memory_base`` holds one editable Project or Problem base a row: a write takes a transaction-scoped
advisory lock on its scope and then updates only where the row's ``version`` is still the one it read. A deleted scope's
row gets ``purged_at`` (a tombstone, hidden from every read); a trigger refuses DELETE, so the ledger is never removed.
``memory_summary`` holds the derived summaries: a publish is one conditional upsert that replaces a summary only when
its cursor is not older than the stored one.

Revision ID: 0024
Revises: 0023
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None

TIME = sa.DateTime(timezone=True)
NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table("memory_base",
                    sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", name="fk_memory_base_workspace_id_workspace"), nullable=False),
                    sa.Column("scope", sa.Text, nullable=False), sa.Column("scope_id", sa.Text, nullable=False),
                    sa.Column("project_id", sa.Text), sa.Column("problem_id", sa.Text),
                    sa.Column("revision", sa.Integer, nullable=False, server_default="0"),
                    sa.Column("version", sa.BigInteger, nullable=False, server_default="1"),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("updated", TIME, nullable=False, server_default=NOW),
                    sa.Column("purged_at", TIME), sa.Column("purge_reason", sa.Text))
    op.create_index("uq_memory_base_live", "memory_base", ["workspace_id", "scope", "scope_id"], unique=True,
                    postgresql_where=sa.text("purged_at is null"))
    op.create_index("ix_memory_base_workspace_id_project_id", "memory_base", ["workspace_id", "project_id"])
    op.execute("""
        create function memory_base_keep() returns trigger language plpgsql as $$
        begin
            raise exception 'memory_base keeps every ledger: DELETE is not allowed (a purge sets purged_at)';
        end $$;
    """)
    op.execute("create trigger memory_base_keep before delete on memory_base for each row execute function memory_base_keep()")

    op.create_table("memory_summary",
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_memory_summary_workspace_id_workspace"),
                              primary_key=True),
                    sa.Column("key", sa.Text, primary_key=True),
                    sa.Column("project_id", sa.Text), sa.Column("problem_id", sa.Text), sa.Column("visibility_class", sa.Text, nullable=False),
                    sa.Column("cursor", ARRAY(sa.BigInteger), nullable=False),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("updated", TIME, nullable=False, server_default=NOW))
    op.create_index("ix_memory_summary_workspace_id_project_id", "memory_summary", ["workspace_id", "project_id"])


def downgrade() -> None:
    op.drop_table("memory_summary")
    op.execute("drop trigger if exists memory_base_keep on memory_base")
    op.execute("drop function if exists memory_base_keep()")
    op.drop_table("memory_base")
