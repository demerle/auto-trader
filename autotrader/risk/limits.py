"""Hard risk limits from the human-only config/risk.yaml, plus pre-trade clamps."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class RiskConfigError(RuntimeError):
    pass


class RiskLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_gross: float = 1.0
    max_position_pct: float = 0.35
    max_order_notional: float = 25_000.0
    min_price: float = 5.0
    allow_short: bool = True
    bot_daily_loss_pct: float = 0.05
    account_max_drawdown_pct: float = 0.20


def _sig_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(path: Path) -> str:
    digest = file_hash(path)
    _sig_path(path).write_text(digest + "\n")
    return digest


def load_limits(path: Path, verify: bool = True) -> RiskLimits:
    if verify:
        sig = _sig_path(path)
        if not sig.exists() or sig.read_text().strip() != file_hash(path):
            raise RiskConfigError(
                f"{path} does not match its signature; review it and run `autotrader risk sign`")
    return RiskLimits.model_validate(yaml.safe_load(path.read_text()) or {})


@dataclass(frozen=True)
class AssetInfo:
    tradable: bool = True
    shortable: bool = True
    easy_to_borrow: bool = True


def clamp_targets(
    target_qty: Mapping[str, float],
    held_qty: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
    limits: RiskLimits,
    assets: Mapping[str, AssetInfo],
) -> tuple[dict[str, float], list[str]]:
    """Apply per-position limits. Reducing risk is always allowed; adding it is clamped."""
    out: dict[str, float] = {}
    notes: list[str] = []
    for s, target in target_qty.items():
        held = held_qty.get(s, 0.0)
        price = prices[s]
        info = assets.get(s, AssetInfo())
        adds_risk = abs(target) > abs(held) or (target * held < 0)
        if not adds_risk:
            out[s] = target
            continue
        if not info.tradable:
            notes.append(f"{s}: not tradable")
            out[s] = held
            continue
        if price < limits.min_price:
            notes.append(f"{s}: price {price:.2f} below min {limits.min_price}")
            out[s] = held if abs(held) < abs(target) else target
            continue
        if target < 0 and not (limits.allow_short and info.shortable and info.easy_to_borrow):
            notes.append(f"{s}: short not allowed")
            target = 0.0
        cap = int(limits.max_position_pct * equity / price)
        if abs(target) > cap:
            notes.append(f"{s}: position clamped {target:+.0f} -> {cap * (1 if target > 0 else -1):+d}")
            target = float(cap if target > 0 else -cap)
        out[s] = target
    return out, notes
