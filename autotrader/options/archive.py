"""Every chain snapshot we fetch is kept: data/options/{UNDERLYING}/{YYYY-MM-DD}/{HHMMSS}.parquet.

From these, backtests replay real chains (with CBOE's IV/greeks/OI/volume) once
enough history has been recorded; until then they use synthetic chains.
"""

from pathlib import Path

import pandas as pd

from autotrader.config import get_settings
from autotrader.options.cboe import ChainSnapshot


class ChainArchive:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or get_settings().chain_dir)

    def _dir(self, underlying: str, day: str) -> Path:
        return self.root / underlying.upper() / day

    def save(self, snap: ChainSnapshot) -> Path:
        et = snap.quote_time.tz_convert("America/New_York")
        d = self._dir(snap.underlying, f"{et:%Y-%m-%d}")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{et:%H%M%S}.parquet"
        if path.exists():
            return path
        df = snap.chain.copy()
        df["quote_time"] = snap.quote_time
        df["fetched_at"] = snap.fetched_at
        df["spot"] = snap.spot
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        return path

    def days(self, underlying: str) -> list[str]:
        base = self.root / underlying.upper()
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []

    def snapshots(self, underlying: str, day: str) -> list[ChainSnapshot]:
        out = []
        for path in sorted(self._dir(underlying, day).glob("*.parquet")):
            df = pd.read_parquet(path)
            if df.empty:
                continue
            first = df.iloc[0]
            chain = df.drop(columns=["quote_time", "fetched_at", "spot"])
            chain["expiry"] = pd.to_datetime(chain["expiry"]).dt.date
            out.append(ChainSnapshot(underlying.upper(), pd.Timestamp(first["fetched_at"]),
                                     pd.Timestamp(first["quote_time"]), float(first["spot"]), chain))
        return out

    def latest(self, underlying: str) -> ChainSnapshot | None:
        for day in reversed(self.days(underlying)):
            snaps = self.snapshots(underlying, day)
            if snaps:
                return snaps[-1]
        return None
