"""One audit trail: every governed action, append-only (package 10.4).

A trigger refuses UPDATE and DELETE on the table, so no code path of the application can change or remove an
entry. TRUNCATE (the test suite's reset) is a table operation, not a row change, and stays possible for the owner.

Revision ID: 0010
Revises: 0009
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("audit_event",
                    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", name="fk_audit_event_workspace_id_workspace"), nullable=False),
                    sa.Column("at", sa.Text, nullable=False), sa.Column("kind", sa.Text, nullable=False),
                    sa.Column("actor", sa.Text, nullable=False), sa.Column("who", sa.Text, nullable=False),
                    sa.Column("user_id", sa.String(64)), sa.Column("project_id", sa.String(64)), sa.Column("project_name", sa.Text),
                    sa.Column("draft_id", sa.String(64)), sa.Column("move", sa.Text), sa.Column("status", sa.Text),
                    sa.Column("detail", JSONB, nullable=False))
    op.create_index("ix_audit_event_workspace_id_id", "audit_event", ["workspace_id", "id"])
    op.create_index("ix_audit_event_workspace_id_kind", "audit_event", ["workspace_id", "kind"])
    op.create_index("ix_audit_event_workspace_id_project_id", "audit_event", ["workspace_id", "project_id"])
    op.execute("""
        create function audit_event_append_only() returns trigger language plpgsql as $$
        begin
            raise exception 'audit_event is append-only: % is not allowed', tg_op;
        end $$;
    """)
    op.execute("create trigger audit_event_append_only before update or delete on audit_event "
               "for each row execute function audit_event_append_only()")


def downgrade() -> None:
    op.execute("drop trigger if exists audit_event_append_only on audit_event")
    op.execute("drop function if exists audit_event_append_only()")
    op.drop_table("audit_event")
