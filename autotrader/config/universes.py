"""Static symbol universes. Intraday bots should stay on small, liquid lists."""

LIQUID_ETFS = (
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
    "XLI", "XLU", "XLB", "XLRE", "XLC", "TLT", "IEF", "GLD", "SLV", "EFA", "EEM",
)

SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC")

MEGA_CAPS = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "V",
    "LLY", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "NFLX", "AMD",
    "CRM", "ORCL", "BAC", "KO", "PEP", "ADBE", "WMT", "MRK", "CVX", "INTC",
)

UNIVERSES: dict[str, tuple[str, ...]] = {
    "liquid_etfs": LIQUID_ETFS,
    "sector_etfs": SECTOR_ETFS,
    "mega_caps": MEGA_CAPS,
    "core": ("SPY", "QQQ", "AAPL", "NVDA", "TSLA", "INTC", "GOOGL", "SMCI"),
}


def resolve(universe: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(universe, str):
        return UNIVERSES[universe]
    return tuple(universe)
