"""Outcomes linked to decisions (package 23.5).

Each ``decision_outcome`` row is a ``dclab_core.outcome.v1`` document: the decision it follows, the metric, its value
and reference, how it was observed and the verdict. Append-only like the ledger's other tables, with the same
privileged-delete exception (0017) so deleting a project removes its outcomes.

Revision ID: 0018
Revises: 0017
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None

TABLE = "decision_outcome"


def upgrade() -> None:
    op.create_table(TABLE,
                    sa.Column("id", sa.String(80), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", name=f"fk_{TABLE}_workspace_id_workspace"), nullable=False),
                    sa.Column("at", sa.Text, nullable=False), sa.Column("record_id", sa.String(80), nullable=False), sa.Column("point", sa.Text, nullable=False),
                    sa.Column("project_id", sa.String(64)), sa.Column("run_id", sa.String(80)), sa.Column("observed", sa.Text, nullable=False),
                    sa.Column("doc", JSONB, nullable=False))
    for column in ("record_id", "point", "project_id"):
        op.create_index(f"ix_{TABLE}_workspace_id_{column}", TABLE, ["workspace_id", column])
    op.execute(f"""
        create function {TABLE}_append_only() returns trigger language plpgsql as $$
        begin
            if current_setting('dclab.privileged_delete', true) = 'on' then
                if tg_op = 'DELETE' then return old; end if;
                return new;
            end if;
            raise exception '{TABLE} is append-only: % is not allowed', tg_op;
        end $$;
    """)
    op.execute(f"create trigger {TABLE}_no_change before update or delete on {TABLE} for each row execute function {TABLE}_append_only()")


def downgrade() -> None:
    op.execute(f"drop trigger if exists {TABLE}_no_change on {TABLE}")
    op.execute(f"drop function if exists {TABLE}_append_only()")
    op.drop_table(TABLE)
