"""dashboard: alerts, reports, audit log

Revision ID: b3c1d2e4f5a6
Revises: 750ebb8991fd
Create Date: 2026-10-19 22:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b3c1d2e4f5a6'
down_revision: Union[str, Sequence[str], None] = '750ebb8991fd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

AGENT_ROLE = "autotrader_agent"


def upgrade() -> None:
    op.create_table('alerts',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('priority', sa.String(length=16), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_alerts'))
    )
    with op.batch_alter_table('alerts', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_alerts_ts'), ['ts'], unique=False)

    op.create_table('reports',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('date', sa.String(length=10), nullable=False),
    sa.Column('body_md', sa.Text(), nullable=False),
    sa.Column('data', sa.JSON(), nullable=False),
    sa.Column('commentary', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('posted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('commentary_posted_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_reports')),
    sa.UniqueConstraint('date', name=op.f('uq_reports_date'))
    )
    op.create_table('audit_log',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('action', sa.String(length=40), nullable=False),
    sa.Column('target', sa.String(length=120), nullable=False),
    sa.Column('detail', sa.JSON(), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_log'))
    )
    # The supervisor agent reads the report and writes only its commentary.
    if op.get_bind().dialect.name == "postgresql":
        op.execute(f"""DO $$ BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{AGENT_ROLE}') THEN
                GRANT SELECT ON reports TO {AGENT_ROLE};
                GRANT UPDATE (commentary) ON reports TO {AGENT_ROLE};
            END IF; END $$""")


def downgrade() -> None:
    op.drop_table('audit_log')
    op.drop_table('reports')
    with op.batch_alter_table('alerts', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_alerts_ts'))
    op.drop_table('alerts')
