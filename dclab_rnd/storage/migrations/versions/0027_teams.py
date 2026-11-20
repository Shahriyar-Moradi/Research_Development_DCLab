"""Teams and who is in them (W2 of the collaborative workspace, 2026-11-20).

A team is a new thing beside the four roles: a role says what a person may do (``accounts/roles.py``, the CHECK on
``membership``), a team says whom an agent represents (the Business team's agent, the Sales team's agent). ``team`` is
one row a team of a workspace, its name unique there; ``kind`` is configuration (business, data_science, ml_engineering,
sales, marketing, or another), so it has no CHECK. ``team_membership`` is one row a (team, person): leaving the team,
the team or the account removes it.

Revision ID: 0027
Revises: 0026
"""

import sqlalchemy as sa
from alembic import op

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None

TIME = sa.DateTime(timezone=True)
NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table("team",
                    sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_team_workspace_id_workspace"),
                              nullable=False),
                    sa.Column("name", sa.Text, nullable=False),
                    sa.Column("kind", sa.String(64), nullable=False),
                    sa.Column("created", TIME, nullable=False, server_default=NOW),
                    sa.PrimaryKeyConstraint("id", name="pk_team"),
                    sa.UniqueConstraint("workspace_id", "name", name="uq_team_workspace_id_name"))
    op.create_table("team_membership",
                    sa.Column("team_id", sa.String(64), sa.ForeignKey("team.id", ondelete="CASCADE", name="fk_team_membership_team_id_team"),
                              nullable=False),
                    sa.Column("user_id", sa.String(64), sa.ForeignKey("app_user.id", ondelete="CASCADE", name="fk_team_membership_user_id_app_user"),
                              nullable=False),
                    sa.Column("since", TIME, nullable=False, server_default=NOW),
                    sa.PrimaryKeyConstraint("team_id", "user_id", name="pk_team_membership"))
    op.create_index("ix_team_membership_user_id", "team_membership", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_team_membership_user_id", table_name="team_membership")
    op.drop_table("team_membership")
    op.drop_table("team")
