import pytest

from autotrader.risk.limits import (
    AssetInfo, RiskConfigError, RiskLimits, clamp_targets, load_limits, sign,
)

L = RiskLimits(max_position_pct=0.5, min_price=5)


def test_position_clamped_to_pct_of_equity():
    out, notes = clamp_targets({"A": 100}, {}, {"A": 10.0}, 1000, L, {})
    assert out == {"A": 50} and notes


def test_reductions_always_allowed():
    out, _ = clamp_targets({"A": 0}, {"A": 100}, {"A": 1.0}, 1000, L, {})
    assert out == {"A": 0}


def test_penny_stock_and_unshortable_blocked():
    out, _ = clamp_targets({"P": 10, "S": -5}, {}, {"P": 2.0, "S": 50.0}, 10_000, L,
                           {"S": AssetInfo(shortable=False)})
    assert out == {"P": 0, "S": 0}


def test_risk_file_must_be_signed(tmp_path):
    p = tmp_path / "risk.yaml"
    p.write_text("max_gross: 1.0\n")
    with pytest.raises(RiskConfigError):
        load_limits(p)
    sign(p)
    assert load_limits(p).max_gross == 1.0
    p.write_text("max_gross: 5.0\n")  # tampered
    with pytest.raises(RiskConfigError):
        load_limits(p)


def test_repo_risk_file_is_signed():
    from autotrader.config import get_settings

    load_limits(get_settings().risk_file)
