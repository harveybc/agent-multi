#!/usr/bin/env python3
"""Screen B rule baselines (doc 40 rev3; Musashi post-P1 order §3).

Arms B0 (flat), B1 (buy&hold), B2a/B2b (TSMOM 180/540 H4 bars on
close[t-1]), B3 (vol-scaled TSMOM: sign_B2a * min(1, 0.15/sigma_ann),
sigma from the last 180 per-bar log returns through t-1, annualization
sqrt(2190), leverage cap 1) — executed through the SAME GymFx cost and
action-accounting path the P1/B4 recipe uses (default action path, no
strategy plugin — matching the campaign's effective config; declared in
the return packet). Three causal origins (score 2022/2023/2024);
sealed 2025 structurally absent. Rules are deterministic: no seeds.

Positions are precomputed from the source CSV with STRICT lag (only
close[<=t-1] enters pos[t]) and driven through the env as raw actions:
sign for B0-B2, signed fraction for B3 (fractional_position_sizing).
Per-bar net returns, positions, turnover, costs and close reasons are
persisted per doc 41; trial-ledger rows are pre-registered BEFORE any
result is computed. No G1 claim is emitted (B4 absent).
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/home/harveybc/Documents/GitHub/predictor/examples/data/"
            "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
DATA_SHA = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc"
            "735747628f8d0435ebe440f")
CONTEXT_BARS = 540          # max lookback (B2b) covers the vol window too
BARS_PER_YEAR = 2190
SIGMA_TARGET = 0.15
VOL_WINDOW = 180
ORIGINS = {"o2022": 2022, "o2023": 2023, "o2024": 2024}
ARMS = ("B0", "B1", "B2a", "B2b", "B3")
# Two transparent cost configurations (P1 recipe left broker costs at
# their zero defaults — surfaced to the auditor; the declared set is a
# PROPOSED constant pending ratification, 5 bp/side + 1 bp slippage):
COST_SETS = {
    "recipe_zero_cost": {},
    "declared_5bp": {"commission": 0.0005, "slippage_perc": 0.0001},
}
SEALED_START = "2025-01-01"

# cost-defining config keys: their digest must be identical across arms
COST_KEYS = ("commission", "initial_cash", "leverage",
             "position_size", "slippage_perc")


class ScreenBError(SystemExit):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source() -> pd.DataFrame:
    actual = sha256_file(DATA)
    if actual != DATA_SHA:
        raise ScreenBError(f"REFUSED: source data sha {actual[:12]} != "
                           f"pinned {DATA_SHA[:12]}")
    df = pd.read_csv(DATA, parse_dates=["DATE_TIME"])
    return df


def materialize_origin(df: pd.DataFrame, year: int, out_dir: Path) -> dict:
    """Slice [score_start - CONTEXT_BARS, score_end]; refuse sealed."""
    in_year = df.index[df["DATE_TIME"].dt.year == year]
    if len(in_year) == 0:
        raise ScreenBError(f"REFUSED: no rows for score year {year}")
    start, end = int(in_year[0]), int(in_year[-1])
    if start < CONTEXT_BARS:
        raise ScreenBError(f"REFUSED: not enough context before {year}")
    sl = df.iloc[start - CONTEXT_BARS:end + 1].reset_index(drop=True)
    if (sl["DATE_TIME"] >= SEALED_START).any():
        raise ScreenBError(
            "REFUSED: sealed-period rows in a development origin slice")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"origin_{year}.csv"
    sl.to_csv(csv, index=False)
    scored = list(range(CONTEXT_BARS, len(sl)))
    scored_id = hashlib.sha256(
        ("|".join(sl["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M")
                  .iloc[scored])).encode()).hexdigest()
    return {"year": year, "csv": str(csv), "csv_sha256": sha256_file(csv),
            "rows": len(sl), "scored_rows": len(scored),
            "scored_start_index": CONTEXT_BARS,
            "scored_index_sha256": scored_id,
            "score_start": str(sl["DATE_TIME"].iloc[CONTEXT_BARS]),
            "score_end": str(sl["DATE_TIME"].iloc[-1])}


def rule_positions(close: np.ndarray, arm: str, scored_start: int
                   ) -> np.ndarray:
    """Signed target exposure per bar index (0 outside scored range).

    STRICT LAG: pos[t] uses close[<= t-1] only.
    """
    n = len(close)
    pos = np.zeros(n)
    logret = np.diff(np.log(close), prepend=np.nan)
    for t in range(scored_start, n):
        if arm == "B0":
            pos[t] = 0.0
        elif arm == "B1":
            pos[t] = 1.0
        elif arm in ("B2a", "B2b", "B3"):
            k = 180 if arm in ("B2a", "B3") else 540
            past, ref = close[t - 1], close[t - 1 - k]
            sign = 1.0 if past > ref else (-1.0 if past < ref else 0.0)
            if arm == "B3":
                window = logret[t - VOL_WINDOW:t]  # returns through t-1
                sigma_ann = float(np.std(window, ddof=1)) * math.sqrt(
                    BARS_PER_YEAR)
                frac = 1.0 if sigma_ann <= 0 else min(
                    1.0, SIGMA_TARGET / sigma_ann)
                pos[t] = sign * frac
            else:
                pos[t] = sign
    return pos


def base_config(origin: dict, arm: str, cost_set: str = "recipe_zero_cost"
                ) -> dict:
    launch = json.loads(Path(
        "/home/harveybc/.local/share/agent-multi/"
        "l1_curriculum_campaign_20260823/seed101_N/"
        "normal_report.launch_manifest.json").read_text())
    cfg = dict(launch["effective_config"])
    for k in ("nested_split_contract", "nested_split_dir",
              "nested_split_manifest", "evaluate_test_split"):
        cfg.pop(k, None)
    cfg["input_data_file"] = origin["csv"]
    cfg["train_days"] = 1  # unused: env consumes the full slice
    cfg["quiet_mode"] = True
    cfg["continuous_action_threshold"] = 0.0
    cfg["fractional_position_sizing"] = (arm == "B3")
    cfg.update(COST_SETS[cost_set])
    cfg.pop("env_mode", None)  # inference-mode accounting
    return cfg


def cost_digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(
        {k: cfg.get(k) for k in COST_KEYS}, sort_keys=True,
        default=str).encode()).hexdigest()


def run_arm(origin: dict, arm: str, out_dir: Path,
            cost_set: str = "recipe_zero_cost") -> dict:
    sys.path.insert(0, str(REPO))
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)
    df = pd.read_csv(origin["csv"])
    close = df["CLOSE"].to_numpy(dtype=float)
    pos = rule_positions(close, arm, origin["scored_start_index"])
    cfg = base_config(origin, arm, cost_set)
    env = _load_env_plugin("gym_fx_env", cfg).make_env(cfg)
    obs, _ = env.reset(seed=0)
    rows = []
    equity_prev = None
    for t in range(len(close)):
        action = [float(pos[t])]
        obs, _r, term, trunc, info = env.step(action)
        econ = float(info.get("economic_equity", np.nan))
        if t >= origin["scored_start_index"]:
            ret = (0.0 if equity_prev in (None, 0.0)
                   else econ / equity_prev - 1.0)
            rows.append({
                "bar": t, "datetime": df["DATE_TIME"].iloc[t],
                "target_position": float(pos[t]),
                "net_return": ret,
                "economic_equity": econ,
                "trade_cost": float(info.get("trade_cost") or 0.0),
                "commission_paid_cum": float(
                    info.get("commission_paid") or 0.0),
                "termination_cause": info.get("termination_cause"),
            })
        equity_prev = econ
        if term or trunc:
            break
    per_bar = pd.DataFrame(rows)
    csv = out_dir / f"{arm}_{origin['year']}_{cost_set}_per_bar.csv"
    per_bar.to_csv(csv, index=False)
    r = per_bar["net_return"].to_numpy()
    tp = per_bar["target_position"].to_numpy()
    turnover = float(np.abs(np.diff(tp, prepend=0.0)).sum())
    eq = per_bar["economic_equity"].to_numpy()
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max(1.0 - eq / peak)) if len(eq) else None
    mean, std = float(np.mean(r)), float(np.std(r, ddof=1))
    sharpe = (mean / std * math.sqrt(BARS_PER_YEAR)) if std > 0 else 0.0
    trades = int((np.abs(np.diff(tp, prepend=0.0)) > 1e-12).sum())
    return {
        "arm": arm, "origin": origin["year"], "cost_set": cost_set,
        "scored_bars": int(len(per_bar)),
        "net_total_return": float(eq[-1] / eq[0] - 1.0) if len(eq) else None,
        "net_sharpe_annualized": sharpe,
        "max_drawdown_fraction": mdd,
        "turnover_sum_abs_dpos": turnover,
        "activity_position_changes": trades,
        "total_costs_trade_cost_channel": float(
            per_bar["trade_cost"].sum()),
        "total_commission_paid": float(
            per_bar["commission_paid_cum"].iloc[-1]
            if len(per_bar) else 0.0),
        "per_bar_csv": str(csv), "per_bar_sha256": sha256_file(csv),
        "cost_config_sha256": cost_digest(cfg),
        "scored_index_sha256": origin["scored_index_sha256"],
        "fractional_position_sizing": bool(arm == "B3"),
        "seeds": "deterministic rule (no seeds)",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--origins", default=",".join(str(y) for y in
                                                 ORIGINS.values()))
    args = ap.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df = load_source()
    origins = [materialize_origin(df, int(y), out / "origins")
               for y in args.origins.split(",")]
    arms = args.arms.split(",")
    # pre-register EVERY arm x origin row BEFORE any result exists
    ledger = out / "trial_ledger.jsonl"
    with open(ledger, "a") as fh:
        for o in origins:
            for arm in arms:
                for cost_set in COST_SETS:
                    fh.write(json.dumps({
                        "screen": "B", "arm": arm, "origin": o["year"],
                        "cost_set": cost_set,
                        "registered_before_results": True,
                        "origin_csv_sha256": o["csv_sha256"],
                        "scored_index_sha256": o["scored_index_sha256"],
                    }) + "\n")
    results = []
    for o in origins:
        for arm in arms:
            for cost_set in COST_SETS:
                print(f"== running {arm} origin {o['year']} "
                      f"[{cost_set}]", flush=True)
                results.append(run_arm(o, arm, out, cost_set))
    packet = {"schema": "agent_multi.screen_b_rule_arms.v1",
              "source_data_sha256": DATA_SHA,
              "sealed_2025_used": False,
              "g1_claim": "NOT_EMITTED (B4 absent)",
              "origins": origins, "results": results,
              "trial_ledger": str(ledger)}
    (out / "SCREEN_B_RESULTS.json").write_text(json.dumps(packet,
                                                          indent=1))
    print(json.dumps({f"{r['arm']}@{r['origin']}:{r['cost_set']}": round(
        r["net_sharpe_annualized"], 3) for r in results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
