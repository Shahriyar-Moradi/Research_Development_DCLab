"""Decision samples and labels (package A6.5): every typed decision the studio made, with the evidence the rule saw,
and what a person did with it; the training data of DCLab's own decision models.

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
    op.create_table(
        "decision_sample",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_decision_sample_workspace_id_workspace"), nullable=False),
        sa.Column("sample_id", sa.String(32), nullable=False),
        sa.Column("at", sa.Text, nullable=False),
        sa.Column("point", sa.Text, nullable=False),
        sa.Column("subject_kind", sa.Text, nullable=False),
        sa.Column("subject", sa.Text),
        sa.Column("key", sa.Text, nullable=False),
        sa.Column("project_id", sa.String(64)),
        sa.Column("draft_id", sa.String(64)),
        sa.Column("dataset_ref", sa.String(32)),
        sa.Column("stage", sa.Text),
        sa.Column("run_ref", sa.Text),
        sa.Column("features", JSONB, nullable=False),
        sa.Column("state_text", sa.Text, nullable=False),
        sa.Column("rule_answer", JSONB),
        sa.Column("rule_version", sa.Text, nullable=False),
        sa.Column("alternatives", JSONB, nullable=False),
        sa.Column("evidence_partition", sa.Text, nullable=False),
        sa.Column("actor", sa.Text, nullable=False),
        sa.Column("user_id", sa.String(64)),
        sa.Column("schema_version", sa.Integer, nullable=False),
        sa.UniqueConstraint("sample_id", name="uq_decision_sample_sample_id"))
    op.create_index("ix_decision_sample_workspace_id_project_id", "decision_sample", ["workspace_id", "project_id"])
    op.create_index("ix_decision_sample_workspace_id_key", "decision_sample", ["workspace_id", "key"])
    op.create_table(
        "decision_label",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspace.id", ondelete="CASCADE", name="fk_decision_label_workspace_id_workspace"), nullable=False),
        sa.Column("label_id", sa.String(32), nullable=False),
        sa.Column("at", sa.Text, nullable=False),
        sa.Column("sample_id", sa.String(32)),
        sa.Column("key", sa.Text, nullable=False),
        sa.Column("project_id", sa.String(64)),
        sa.Column("label", JSONB),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("who", sa.Text, nullable=False),
        sa.Column("user_id", sa.String(64)),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("move", sa.Text),
        sa.Column("schema_version", sa.Integer, nullable=False),
        sa.UniqueConstraint("label_id", name="uq_decision_label_label_id"))
    op.create_index("ix_decision_label_workspace_id_project_id", "decision_label", ["workspace_id", "project_id"])
    op.create_index("ix_decision_label_workspace_id_key", "decision_label", ["workspace_id", "key"])


def downgrade() -> None:
    op.drop_table("decision_label")
    op.drop_table("decision_sample")
