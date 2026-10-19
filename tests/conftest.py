import numpy as np
import pandas as pd
import pytest


def make_bars(n: int = 400, seed: int = 0, start: str = "2020-01-01", drift: float = 0.0003,
              vol: float = 0.015, freq: str = "B") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="ts")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture
def universe() -> dict[str, pd.DataFrame]:
    return {s: make_bars(seed=i) for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"])}
