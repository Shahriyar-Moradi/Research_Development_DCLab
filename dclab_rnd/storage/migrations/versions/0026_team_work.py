"""The work passed between team agents (W4 of the collaborative workspace, 2026-11-20).

``team_task``, ``team_action`` and ``team_report`` keep dclab_core.team_work's records whole (``doc``) with the columns a
read filters on; ``team_delivery`` is the outbox: a task, a clarification, its answer or a reply is saved with its
delivery in one transaction, and a job delivers a task's rows in order (dclab_rnd/teamspace). Ids are unique within a
project, as the file store keeps one document per project. A retry key has at most one succeeded or simulated action (a
partial unique index), a report is one row per (task, revision), and a delivery is one row per what it carries (``key``).

The next migration must use ``down_revision = "0026"`` (W2's teams migration, 0027, already does).

Revision ID: 0026
Revises: 0024
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0026"
down_revision = "0024"
branch_labels = None
depends_on = None

TIME = sa.DateTime(timezone=True)
NOW = sa.text("now()")


def _workspace(table: str) -> sa.Column:
    return sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name=f"fk_{table}_workspace_id_workspace"),
                     nullable=False)


def upgrade() -> None:
    op.create_table("team_task", _workspace("team_task"),
                    sa.Column("project_id", sa.Text, nullable=False),
                    sa.Column("id", sa.Text, nullable=False),
                    sa.Column("seq", sa.BigInteger, sa.Identity(), nullable=False),
                    sa.Column("problem_id", sa.Text, nullable=False),
                    sa.Column("parent_task_id", sa.Text),
                    sa.Column("owner_team", sa.Text, nullable=False), sa.Column("requester_team", sa.Text, nullable=False),
                    sa.Column("status", sa.Text, nullable=False),
                    sa.Column("version", sa.BigInteger, nullable=False, server_default="1"),
                    sa.Column("initiator", JSONB),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("updated", TIME, nullable=False, server_default=NOW),
                    sa.PrimaryKeyConstraint("workspace_id", "project_id", "id", name="pk_team_task"))
    op.create_index("ix_team_task_workspace_id_project_id_problem_id", "team_task", ["workspace_id", "project_id", "problem_id"])
    op.create_index("ix_team_task_workspace_id_project_id_parent_task_id", "team_task", ["workspace_id", "project_id", "parent_task_id"])
    op.create_index("ix_team_task_workspace_id_owner_team_status", "team_task", ["workspace_id", "owner_team", "status"])

    op.create_table("team_action", _workspace("team_action"),
                    sa.Column("project_id", sa.Text, nullable=False),
                    sa.Column("task_id", sa.Text, nullable=False), sa.Column("action_id", sa.Text, nullable=False),
                    sa.Column("seq", sa.BigInteger, sa.Identity(), nullable=False),
                    sa.Column("retry_key", sa.Text, nullable=False),
                    sa.Column("status", sa.Text, nullable=False),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("updated", TIME, nullable=False, server_default=NOW),
                    sa.PrimaryKeyConstraint("workspace_id", "project_id", "task_id", "action_id", name="pk_team_action"))
    op.create_index("uq_team_action_done_key", "team_action", ["workspace_id", "project_id", "retry_key"], unique=True,
                    postgresql_where=sa.text("status in ('succeeded', 'simulated')"))

    op.create_table("team_report", _workspace("team_report"),
                    sa.Column("project_id", sa.Text, nullable=False),
                    sa.Column("task_id", sa.Text, nullable=False), sa.Column("revision", sa.Integer, nullable=False),
                    sa.Column("seq", sa.BigInteger, sa.Identity(), nullable=False),
                    sa.Column("parent_task_id", sa.Text),
                    sa.Column("status", sa.Text, nullable=False),
                    sa.Column("joined", sa.Boolean, nullable=False, server_default=sa.false()),
                    sa.Column("verdict", JSONB),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("updated", TIME, nullable=False, server_default=NOW),
                    sa.PrimaryKeyConstraint("workspace_id", "project_id", "task_id", "revision", name="pk_team_report"))
    op.create_index("ix_team_report_workspace_id_project_id_parent_task_id", "team_report", ["workspace_id", "project_id", "parent_task_id"])

    op.create_table("team_delivery",
                    sa.Column("id", sa.String(64), nullable=False),
                    _workspace("team_delivery"),
                    sa.Column("seq", sa.BigInteger, sa.Identity(), nullable=False),
                    sa.Column("key", sa.Text, nullable=False),
                    sa.Column("kind", sa.Text, nullable=False),
                    sa.Column("project_id", sa.Text, nullable=False), sa.Column("problem_id", sa.Text, nullable=False),
                    sa.Column("task_id", sa.Text, nullable=False),
                    sa.Column("status", sa.Text, nullable=False),
                    sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
                    sa.Column("next_attempt_at", TIME),
                    sa.Column("version", sa.BigInteger, nullable=False, server_default="1"),
                    sa.Column("doc", JSONB, nullable=False),
                    sa.Column("updated", TIME, nullable=False, server_default=NOW),
                    sa.CheckConstraint("status in ('pending', 'delivered', 'blocked', 'refused')", name="ck_team_delivery_status"),
                    sa.UniqueConstraint("workspace_id", "project_id", "key", name="uq_team_delivery_workspace_id_project_id_key"),
                    sa.PrimaryKeyConstraint("id", name="pk_team_delivery"))
    op.create_index("ix_team_delivery_workspace_id_status", "team_delivery", ["workspace_id", "status"])
    op.create_index("ix_team_delivery_workspace_id_project_id_task_id_seq", "team_delivery", ["workspace_id", "project_id", "task_id", "seq"])
    op.create_index("ix_team_delivery_workspace_id_project_id_problem_id_seq", "team_delivery",
                    ["workspace_id", "project_id", "problem_id", "seq"])


def downgrade() -> None:
    op.drop_table("team_delivery")
    op.drop_table("team_report")
    op.drop_table("team_action")
    op.drop_table("team_task")
