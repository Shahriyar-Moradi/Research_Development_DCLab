"""Retention and deletion through the append-only record (package 19.10).

The session record and the decision ledger refuse UPDATE and DELETE. Retention (what is kept, for how long) and the
deletion of a project or a session must still remove what they held, and only they may: the triggers now let a
transaction through when it set ``dclab.privileged_delete`` to ``on`` (``SET LOCAL``, so it ends with the
transaction), which only ``dclab_rnd.experience.retention`` does, and every such removal leaves an audit entry.

Revision ID: 0017
Revises: 0016
"""

from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None

TABLES = ("experience_event", "decision_record", "decision_label")


def upgrade() -> None:
    for table in TABLES:
        op.execute(f"""
            create or replace function {table}_append_only() returns trigger language plpgsql as $$
            begin
                if current_setting('dclab.privileged_delete', true) = 'on' then
                    if tg_op = 'DELETE' then return old; end if;
                    return new;
                end if;
                raise exception '{table} is append-only: % is not allowed', tg_op;
            end $$;
        """)


def downgrade() -> None:
    for table in TABLES:
        op.execute(f"""
            create or replace function {table}_append_only() returns trigger language plpgsql as $$
            begin
                raise exception '{table} is append-only: % is not allowed', tg_op;
            end $$;
        """)
