import pandas as pd

from autotrader.data.archive import BarArchive
from autotrader.data.calendar import bar_close_time, drop_incomplete
from autotrader.data.providers.yahoo import parse_chart
from autotrader.data.quality import check_bars
from returns.result import Success
from tests.conftest import make_bars

PAYLOAD = {"chart": {"result": [{
    "meta": {"exchangeTimezoneName": "America/New_York"},
    "timestamp": [1704205800, 1704292200, 1704378600],  # 2024-01-02/03/04 09:30 ET
    "indicators": {"quote": [{
        "open": [10.0, None, 12.0], "high": [11.0, 12.0, 13.0], "low": [9.0, 10.0, 11.0],
        "close": [10.5, 11.5, 12.5], "volume": [100, 200, None],
    }]},
}], "error": None}}


def test_parse_chart_drops_null_rows_and_stamps_daily_dates():
    df = parse_chart(PAYLOAD, "1d")
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-04"]
    assert df.index.tz is not None and df.index[0].hour == 0
    assert df.volume.iloc[-1] == 0.0


def test_bar_close_times():
    day = pd.Timestamp("2024-01-02", tz="UTC")
    assert bar_close_time(day, "1d") == pd.Timestamp("2024-01-02 21:00", tz="UTC")
    last_hour = pd.Timestamp("2024-01-02 20:30", tz="UTC")  # 15:30 ET bar is a half hour
    assert bar_close_time(last_hour, "1h") == pd.Timestamp("2024-01-02 21:00", tz="UTC")


def test_drop_incomplete_daily_bar():
    df = parse_chart(PAYLOAD, "1d")
    midday = pd.Timestamp("2024-01-04 17:00", tz="UTC")
    assert len(drop_incomplete(df, "1d", midday)) == 1


def test_quality_flags_jumps_and_staleness():
    df = make_bars(30)
    df.iloc[-1, df.columns.get_loc("open")] = df.close.iloc[-2] * 2
    df.iloc[-1, df.columns.get_loc("high")] = df.close.iloc[-2] * 2.1
    kinds = {i.kind for i in check_bars("X", df, "1d", now=df.index[-1] + pd.Timedelta(days=30))}
    assert {"price_jump", "stale"} <= kinds


class FakeProvider:
    name = "fake"

    def __init__(self, df):
        self.df, self.calls = df, []

    def fetch_bars(self, symbol, interval, start=None, end=None):
        self.calls.append(start)
        df = self.df if start is None else self.df[self.df.index >= start]
        return Success(df)


def test_archive_merges_and_fetches_incrementally(tmp_path):
    df = make_bars(20, start="2024-01-02")
    provider = FakeProvider(df.iloc[:10])
    archive = BarArchive(tmp_path, [provider])
    now = pd.Timestamp("2025-01-01", tz="UTC")
    assert len(archive.update("AAA", "1d", now=now)) == 10
    provider.df = df
    assert len(archive.update("AAA", "1d", now=now)) == 20
    assert provider.calls[1] is not None  # second fetch only asked for the tail
    pd.testing.assert_frame_equal(archive.load("AAA", "1d"), df, check_freq=False)


def test_archive_backfills_older_history(tmp_path):
    df = make_bars(40, start="2024-01-02")
    provider = FakeProvider(df.iloc[20:])
    archive = BarArchive(tmp_path, [provider])
    now = pd.Timestamp("2025-01-01", tz="UTC")
    archive.update("AAA", "1d", now=now)
    provider.df = df
    assert len(archive.update("AAA", "1d", start=df.index[0], now=now)) == 40


def test_failed_backfill_keeps_existing_data(tmp_path):
    from returns.result import Failure

    df = make_bars(20, start="2024-01-02")
    provider = FakeProvider(df)
    archive = BarArchive(tmp_path, [provider])
    now = pd.Timestamp("2025-01-01", tz="UTC")
    archive.update("AAA", "1d", now=now)

    def fail_old(symbol, interval, start=None, end=None):
        if start is not None and start < df.index[0]:
            return Failure(ValueError("404: no data before listing"))
        return Success(df[df.index >= start] if start is not None else df)

    provider.fetch_bars = fail_old
    assert len(archive.update("AAA", "1d", start=pd.Timestamp("2010-01-01", tz="UTC"), now=now)) == 20
