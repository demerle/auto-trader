from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Trading frictions. Alpaca charges no commission, but regulatory fees and
    slippage are real; defaults are deliberately a bit pessimistic."""

    slippage_bps: float = 5.0
    sec_fee_rate: float = 0.0000278  # $ per $ of sell notional
    taf_per_share: float = 0.000166  # FINRA TAF on sells
    taf_max: float = 8.30
    borrow_rate_annual: float = 0.03  # on short market value

    def fill_price(self, price: float, qty: float) -> float:
        """Adverse slippage: buys pay up, sells receive less."""
        slip = self.slippage_bps / 1e4
        return price * (1 + slip) if qty > 0 else price * (1 - slip)

    def fees(self, qty: float, price: float) -> float:
        if qty >= 0:
            return 0.0
        shares = -qty
        return shares * price * self.sec_fee_rate + min(shares * self.taf_per_share, self.taf_max)


ZERO_COSTS = CostModel(slippage_bps=0.0, sec_fee_rate=0.0, taf_per_share=0.0, borrow_rate_annual=0.0)
