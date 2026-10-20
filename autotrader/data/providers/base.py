from typing import Protocol

import pandas as pd
from returns.result import Result

from autotrader.data.intervals import Interval


class BarProvider(Protocol):
    name: str

    def fetch_bars(
        self,
        symbol: str,
        interval: Interval,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> Result[pd.DataFrame, Exception]:
        """Return a canonical bar frame (see autotrader.data.bars)."""
        ...
