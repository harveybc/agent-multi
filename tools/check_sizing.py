"""Sanity check sizing math for each config."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure we load gym-fx plugins from its working copy if running from agent-multi
sys.path.insert(0, str(Path.home() / "Documents/GitHub/gym-fx"))

from strategy_plugins.direct_atr_sltp import Plugin

MOCK_PRICES = {"btc": 65000.0, "eth": 3200.0, "eurusd": 1.10}


def mock_price(name: str) -> float:
    for k, v in MOCK_PRICES.items():
        if k in name:
            return v
    return 1.0


class _MockBroker:
    @staticmethod
    def getcash() -> float:
        return 10000.0


class _MockData:
    def __init__(self, price: float) -> None:
        self.close = [price]


class _MockStrategy:
    def __init__(self, price: float) -> None:
        self.broker = _MockBroker()
        self.data = _MockData(price)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    cfgs = [
        "examples/config/p4_ppo_btc_1h.json",
        "examples/config/p4_ppo_eth_1h.json",
        "examples/config/p4_ppo_eurusd_1h.json",
        "examples/config/sac_btc_1h_twelve_atr.json",
    ]
    for rel_path in cfgs:
        cfg = json.loads((root / rel_path).read_text())
        p = Plugin()
        resolved = p._resolve(cfg)
        price = mock_price(rel_path)
        mock = _MockStrategy(price)
        size = p._compute_size(mock, resolved)
        notional = size * price
        pct = 100.0 * notional / 10000.0
        rel = resolved.get("rel_volume")
        lev = resolved.get("leverage")
        mode = resolved.get("size_mode")
        print(
            f"{rel_path}\n"
            f"  rel={rel} lev={lev} mode={mode}\n"
            f"  price={price} -> size={size:.6f} units, notional=${notional:,.0f} ({pct:.2f}% of $10k)\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
