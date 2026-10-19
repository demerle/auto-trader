"""options gauntlet: bot groups, option orders, weekly results

Revision ID: c4d5e6f7a8b9
Revises: b3c1d2e4f5a6
Create Date: 2026-10-19 23:30:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c4d5e6f7a8b9'
down_revision: Union[str, Sequence[str], None] = 'b3c1d2e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

AGENT_ROLE = "autotrader_agent"


def upgrade() -> None:
    with op.batch_alter_table('bots', schema=None) as batch_op:
        batch_op.add_column(sa.Column('group', sa.String(length=16), server_default='arena', nullable=False))
        batch_op.create_index(batch_op.f('ix_bots_group'), ['group'], unique=False)
    with op.batch_alter_table('orders', schema=None) as batch_op:
        batch_op.alter_column('symbol', existing_type=sa.String(length=16), type_=sa.String(length=32))
        batch_op.add_column(sa.Column('legs', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('structure', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('chase', sa.Integer(), server_default='0', nullable=False))
    with op.batch_alter_table('fills', schema=None) as batch_op:
        batch_op.alter_column('symbol', existing_type=sa.String(length=16), type_=sa.String(length=32))
        batch_op.add_column(sa.Column('structure', sa.String(length=64), nullable=True))

    op.create_table('weekly_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('week', sa.String(length=10), nullable=False),
    sa.Column('bot_id', sa.Integer(), nullable=False),
    sa.Column('start_equity', sa.Float(), nullable=False),
    sa.Column('end_equity', sa.Float(), nullable=True),
    sa.Column('ret', sa.Float(), nullable=True),
    sa.Column('max_drawdown', sa.Float(), nullable=True),
    sa.Column('trades', sa.Integer(), nullable=False),
    sa.Column('sessions', sa.Integer(), nullable=False),
    sa.Column('rank', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('busted', sa.Boolean(), nullable=False),
    sa.Column('autopsy', sa.Text(), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('scored_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['bot_id'], ['bots.id'], name=op.f('fk_weekly_results_bot_id_bots')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_weekly_results')),
    sa.UniqueConstraint('week', 'bot_id', name=op.f('uq_weekly_results_week'))
    )
    with op.batch_alter_table('weekly_results', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_weekly_results_week'), ['week'], unique=False)
        batch_op.create_index(batch_op.f('ix_weekly_results_bot_id'), ['bot_id'], unique=False)

    # Options researchers read results, trades and money moves; they never write them.
    if op.get_bind().dialect.name == "postgresql":
        op.execute(f"""DO $$ BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{AGENT_ROLE}') THEN
                GRANT SELECT ON weekly_results, orders, order_allocations, cash_flows, system_state
                    TO {AGENT_ROLE};
            END IF; END $$""")


def downgrade() -> None:
    with op.batch_alter_table('weekly_results', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_weekly_results_bot_id'))
        batch_op.drop_index(batch_op.f('ix_weekly_results_week'))
    op.drop_table('weekly_results')
    with op.batch_alter_table('fills', schema=None) as batch_op:
        batch_op.drop_column('structure')
        batch_op.alter_column('symbol', existing_type=sa.String(length=32), type_=sa.String(length=16))
    with op.batch_alter_table('orders', schema=None) as batch_op:
        batch_op.drop_column('chase')
        batch_op.drop_column('structure')
        batch_op.drop_column('legs')
        batch_op.alter_column('symbol', existing_type=sa.String(length=32), type_=sa.String(length=16))
    with op.batch_alter_table('bots', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_bots_group'))
        batch_op.drop_column('group')
