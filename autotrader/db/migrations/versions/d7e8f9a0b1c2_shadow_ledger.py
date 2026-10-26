"""shadow ledger: real-money fill prices for options

Revision ID: d7e8f9a0b1c2
Revises: c4d5e6f7a8b9
Create Date: 2026-10-26 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd7e8f9a0b1c2'
down_revision: Union[str, Sequence[str], None] = 'c4d5e6f7a8b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('fills', schema=None) as batch_op:
        batch_op.add_column(sa.Column('shadow_price', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('quote', sa.JSON(), nullable=True))
    with op.batch_alter_table('weekly_results', schema=None) as batch_op:
        batch_op.add_column(sa.Column('shadow_ret', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('weekly_results', schema=None) as batch_op:
        batch_op.drop_column('shadow_ret')
    with op.batch_alter_table('fills', schema=None) as batch_op:
        batch_op.drop_column('quote')
        batch_op.drop_column('shadow_price')
