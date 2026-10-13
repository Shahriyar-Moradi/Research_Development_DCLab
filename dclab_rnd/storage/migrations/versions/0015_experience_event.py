"""The record of every session: one append-only log of experience events (package 19.2).

Each row is a ``dclab_core.experience_event.v1`` document with its ids in columns for finding it. A trigger refuses
UPDATE and DELETE, as for the audit trail; retention (package 19.6) will delete through a privileged path.

Revision ID: 0015
Revises: 0014
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("experience_event",
                    sa.Column("id", sa.String(40), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", name="fk_experience_event_workspace_id_workspace"), nullable=False),
                    sa.Column("at", sa.Text, nullable=False), sa.Column("seq", sa.BigInteger, nullable=False),
                    sa.Column("kind", sa.Text, nullable=False), sa.Column("actor", sa.Text, nullable=False),
                    sa.Column("visit", sa.String(40)), sa.Column("user_id", sa.String(64)), sa.Column("conversation", sa.String(64)),
                    sa.Column("project_id", sa.String(64)), sa.Column("run_id", sa.String(64)), sa.Column("job_id", sa.String(64)),
                    sa.Column("request_id", sa.String(80)), sa.Column("model_request", sa.String(80)),
                    sa.Column("doc", JSONB, nullable=False))
    op.create_index("ix_experience_event_workspace_id_at", "experience_event", ["workspace_id", "at"])
    op.create_index("ix_experience_event_workspace_id_visit", "experience_event", ["workspace_id", "visit"])
    op.create_index("ix_experience_event_workspace_id_project_id", "experience_event", ["workspace_id", "project_id"])
    op.create_index("ix_experience_event_workspace_id_conversation", "experience_event", ["workspace_id", "conversation"])
    for column in ("user_id", "run_id", "job_id", "request_id", "model_request", "kind"):
        op.create_index(f"ix_experience_event_workspace_id_{column}", "experience_event", ["workspace_id", column])
    op.execute("""
        create function experience_event_append_only() returns trigger language plpgsql as $$
        begin
            raise exception 'experience_event is append-only: % is not allowed', tg_op;
        end $$;
    """)
    op.execute("create trigger experience_event_append_only before update or delete on experience_event "
               "for each row execute function experience_event_append_only()")


def downgrade() -> None:
    op.execute("drop trigger if exists experience_event_append_only on experience_event")
    op.execute("drop function if exists experience_event_append_only()")
    op.drop_table("experience_event")
