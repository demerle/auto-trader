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


SYMS = ("AAA", "BBB", "CCC", "DDD")


@pytest.fixture
def history() -> dict[str, pd.DataFrame]:
    """Synthetic daily bars on real NYSE sessions (Jan 2023 - Jun 2024)."""
    from autotrader.data.calendar import schedule

    days = schedule(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-06-30", tz="UTC")).index
    out = {}
    for i, s in enumerate(SYMS):
        df = make_bars(len(days), seed=i)
        df.index = days
        out[s] = df
    return out


@pytest.fixture(autouse=True)
def sent_alerts(monkeypatch) -> list[tuple[str, str]]:
    """Tests never store alerts in the real database or push them to Discord/ntfy."""
    from autotrader import alerts

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(alerts, "_record", lambda title, message, priority: None)
    monkeypatch.setattr(alerts, "_deliver", lambda title, message, priority: sent.append((title, message)))
    monkeypatch.setattr(alerts, "post_discord", lambda url, payload, tries=3: sent.append(("discord", payload)))
    return sent
