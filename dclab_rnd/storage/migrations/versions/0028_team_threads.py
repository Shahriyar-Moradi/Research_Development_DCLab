"""Team threads (W3 of the collaborative workspace, 2026-11-20): each team's own conversation on a problem.

``team_thread`` is one row a (workspace, project, problem, team): the team is a team's id (storage/teams.py), or a team
agent's name when no team of the workspace has it. ``team_message`` holds its messages whole, numbered 1, 2, … within the
thread (the event stream resumes from that number). A message is inserted, never rewritten: a trigger refuses UPDATE
(a DELETE is left to the cascade from a deleted workspace or thread). A message the delivery outbox posts carries its
delivery's key (``delivery:<id>``), unique within a project, so a delivery delivered twice is posted once.

Revision ID: 0028
Revises: 0027
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None

TIME = sa.DateTime(timezone=True)
NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table("team_thread",
                    sa.Column("id", sa.String(64), nullable=False),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_team_thread_workspace_id_workspace"),
                              nullable=False),
                    sa.Column("project_id", sa.Text, nullable=False), sa.Column("problem_id", sa.Text, nullable=False),
                    sa.Column("team", sa.Text, nullable=False),
                    sa.Column("message_seq", sa.BigInteger, nullable=False, server_default="0"),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("created", TIME, nullable=False, server_default=NOW),
                    sa.PrimaryKeyConstraint("id", name="pk_team_thread"),
                    sa.UniqueConstraint("workspace_id", "project_id", "problem_id", "team", name="uq_team_thread_workspace_id_project_id_problem_id_team"))
    op.create_index("ix_team_thread_workspace_id_team", "team_thread", ["workspace_id", "team"])
    op.create_table("team_message",
                    sa.Column("thread_id", sa.String(64), sa.ForeignKey("team_thread.id", ondelete="CASCADE", name="fk_team_message_thread_id_team_thread"),
                              nullable=False),
                    sa.Column("seq", sa.BigInteger, nullable=False),
                    sa.Column("id", sa.String(64), nullable=False),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_team_message_workspace_id_workspace"),
                              nullable=False),
                    sa.Column("project_id", sa.Text, nullable=False),
                    sa.Column("key", sa.Text),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("created", TIME, nullable=False, server_default=NOW),
                    sa.PrimaryKeyConstraint("thread_id", "seq", name="pk_team_message"),
                    sa.UniqueConstraint("id", name="uq_team_message_id"),
                    sa.UniqueConstraint("workspace_id", "project_id", "key", name="uq_team_message_workspace_id_project_id_key"))
    op.execute("""
        create function team_message_never_rewritten() returns trigger language plpgsql as $$
        begin
            raise exception 'team_message is append-only: UPDATE is not allowed';
        end $$;
    """)
    op.execute("create trigger team_message_never_rewritten before update on team_message for each row execute function team_message_never_rewritten()")


def downgrade() -> None:
    op.execute("drop trigger if exists team_message_never_rewritten on team_message")
    op.execute("drop function if exists team_message_never_rewritten()")
    op.drop_table("team_message")
    op.drop_index("ix_team_thread_workspace_id_team", table_name="team_thread")
    op.drop_table("team_thread")
