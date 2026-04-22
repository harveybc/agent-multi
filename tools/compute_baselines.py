"""Compute P5.4 baselines on d6 (2023-01-01 → 2025-12-31) for each asset.

Three baselines per asset, computed on hourly CLOSE series:
  - buy_and_hold      : enter at bar 0, exit at last bar. Single trade.
  - random_long_flat  : each bar, flip a coin → long or flat (no shorts).
                        Averaged over N_SEEDS to get expected Sharpe.
  - random_3action    : each bar, uniform {long, short, flat}. Averaged.

Metrics: total_return, annualized Sharpe (252*24 hours/yr), max_drawdown_pct,
         trade_count, win_rate (on closed trades).

Output: <agent-multi>/logs/partIII/p5_baselines.json  +  p5_baselines.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "project2"

HOURS_PER_YEAR = 252 * 24  # trading-day convention
N_SEEDS = 50

ASSETS = ["btcusdt_1h", "ethusdt_1h", "eurusd_1h"]


def _load_d6(asset: str) -> pd.DataFrame:
    p = DATA_ROOT / asset / "d6.csv"
    df = pd.read_csv(p, usecols=["DATE_TIME", "CLOSE"])
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    return df


def _max_dd(equity: np.ndarray) -> float:
    roll_max = np.maximum.accumulate(equity)
    dd = (roll_max - equity) / roll_max
    return float(dd.max() * 100.0)


def _sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    mu = returns.mean()
    sd = returns.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(mu / sd * math.sqrt(HOURS_PER_YEAR))


def _metrics_from_positions(close: np.ndarray, position: np.ndarray) -> Dict[str, float]:
    """position[t] is the position held from t→t+1. bar returns = position[t] * pct_change[t+1]."""
    pct = np.diff(close) / close[:-1]
    pos = position[:-1]  # align
    bar_ret = pos * pct
    equity = np.concatenate([[1.0], np.cumprod(1.0 + bar_ret)])
    # Count trades as position flips (non-zero to different non-zero, or zero boundary)
    flips = int(np.sum(np.diff(position) != 0))
    # Win rate: of closed trades, count runs where cumulative return over the run > 0
    wins = 0
    closed = 0
    i = 0
    while i < len(position):
        p = position[i]
        if p == 0:
            i += 1
            continue
        j = i
        while j < len(position) and position[j] == p:
            j += 1
        seg_close = close[i:min(j + 1, len(close))]
        if len(seg_close) >= 2:
            seg_ret = (seg_close[-1] / seg_close[0] - 1.0) * p
            closed += 1
            if seg_ret > 0:
                wins += 1
        i = j
    win_rate = wins / closed if closed > 0 else 0.0
    return {
        "total_return": float(equity[-1] - 1.0),
        "final_equity": float(equity[-1]),
        "sharpe_ratio": _sharpe(bar_ret),
        "max_drawdown_pct": _max_dd(equity),
        "trade_count": int(closed),
        "flip_count": flips,
        "win_rate": float(win_rate),
        "n_bars": int(len(close)),
    }


def buy_and_hold(df: pd.DataFrame) -> Dict[str, float]:
    close = df["CLOSE"].to_numpy(dtype=float)
    pos = np.ones(len(close), dtype=float)
    m = _metrics_from_positions(close, pos)
    m["name"] = "buy_and_hold"
    return m


def random_policy(df: pd.DataFrame, mode: str, n_seeds: int = N_SEEDS) -> Dict[str, float]:
    close = df["CLOSE"].to_numpy(dtype=float)
    results: List[Dict[str, float]] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        if mode == "long_flat":
            pos = rng.choice([0.0, 1.0], size=len(close))
        elif mode == "3action":
            pos = rng.choice([-1.0, 0.0, 1.0], size=len(close))
        else:
            raise ValueError(mode)
        results.append(_metrics_from_positions(close, pos))

    agg: Dict[str, float] = {"name": f"random_{mode}", "n_seeds": n_seeds}
    for k in ("total_return", "sharpe_ratio", "max_drawdown_pct", "trade_count", "win_rate"):
        vals = np.array([r[k] for r in results], dtype=float)
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        agg[f"{k}_min"] = float(vals.min())
        agg[f"{k}_max"] = float(vals.max())
    agg["n_bars"] = results[0]["n_bars"]
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=ASSETS)
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "logs" / "partIII"))
    args = ap.parse_args()

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for asset in args.assets:
        df = _load_d6(asset)
        print(f"[{asset}] n_bars={len(df)}  from {df['DATE_TIME'].iloc[0]} to {df['DATE_TIME'].iloc[-1]}")
        out[asset] = {
            "buy_and_hold": buy_and_hold(df),
            "random_long_flat": random_policy(df, "long_flat", args.n_seeds),
            "random_3action": random_policy(df, "3action", args.n_seeds),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p5_baselines.json").write_text(json.dumps(out, indent=2))

    # MD table
    lines: List[str] = ["# P5.4 Baselines on d6 (2023-01-01 → 2025-12-31)", ""]
    lines.append(f"- n_seeds for random policies: {args.n_seeds}")
    lines.append(f"- Sharpe annualization factor: sqrt({HOURS_PER_YEAR}) hours/yr (252 × 24)")
    lines.append("")
    lines.append("## Buy & Hold")
    lines.append("")
    lines.append("| asset | n_bars | total_return | Sharpe | max_DD% | trades | win_rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in args.assets:
        r = out[a]["buy_and_hold"]
        lines.append(f"| {a} | {r['n_bars']} | {r['total_return']:.4f} | {r['sharpe_ratio']:.3f} | {r['max_drawdown_pct']:.2f} | {r['trade_count']} | {r['win_rate']:.3f} |")
    lines.append("")
    lines.append("## Random policies (mean ± std over seeds)")
    lines.append("")
    lines.append("| asset | policy | ret mean±std | Sharpe mean±std | DD% mean | trades mean | win mean |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in args.assets:
        for key in ("random_long_flat", "random_3action"):
            r = out[a][key]
            lines.append(
                f"| {a} | {key} "
                f"| {r['total_return_mean']:.4f} ± {r['total_return_std']:.4f} "
                f"| {r['sharpe_ratio_mean']:.3f} ± {r['sharpe_ratio_std']:.3f} "
                f"| {r['max_drawdown_pct_mean']:.2f} "
                f"| {r['trade_count_mean']:.0f} "
                f"| {r['win_rate_mean']:.3f} |"
            )
    (out_dir / "p5_baselines.md").write_text("\n".join(lines))

    print(f"wrote {out_dir/'p5_baselines.json'}")
    print(f"wrote {out_dir/'p5_baselines.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
