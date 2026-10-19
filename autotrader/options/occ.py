"""OCC option symbols, e.g. SPY261019C00580000 = SPY, 2026-10-19, call, strike 580."""

import re
from dataclasses import dataclass
from datetime import date

OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")
MULTIPLIER = 100


@dataclass(frozen=True)
class Contract:
    underlying: str
    expiry: date
    right: str  # "C" or "P"
    strike: float

    @property
    def symbol(self) -> str:
        return occ_symbol(self.underlying, self.expiry, self.right, self.strike)


def is_option(symbol: str) -> bool:
    return bool(OCC_RE.match(symbol))


def multiplier(symbol: str) -> int:
    return MULTIPLIER if is_option(symbol) else 1


def parse(symbol: str) -> Contract:
    m = OCC_RE.match(symbol)
    if not m:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, yy, mm, dd, right, strike = m.groups()
    return Contract(root, date(2000 + int(yy), int(mm), int(dd)), right, int(strike) / 1000)


def occ_symbol(underlying: str, expiry: date, right: str, strike: float) -> str:
    return f"{underlying}{expiry:%y%m%d}{right}{round(strike * 1000):08d}"
