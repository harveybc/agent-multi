#!/usr/bin/env python3
"""Screen B rule baselines v2 (correction order 2026-08-25, C2-C5).

B0-B3 through the SHARED EXECUTION ENVELOPE (gym-fx
shared_execution_envelope): deployed-geometry SL/TP (stop 1% / take 2%
per entry, inherited from the LIVE mt5_eth_sac_model_runner_v1 config —
no free choice), stop_first_pessimistic collision, portfolio-fraction
sizing units = lagged_equity * min(1,|raw|) / lagged_close (C3), close
taxonomy (envelope_close_sl/tp, policy_close, reversal_close,
data_end_liquidation). Costs come from the versioned evidence-backed
cost manifest (C4): primary governs G1 (pending ratification),
zero_cost is diagnostic, stress descriptive. Evidence carries the C5
run manifest (code identities, digests, timing p50/p95, H4 deadline)
and a deterministic idempotent trial ledger.
"""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/home/harveybc/Documents/GitHub/predictor/examples/data/"
            "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
DATA_SHA = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc"
            "735747628f8d0435ebe440f")
COST_MANIFEST = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                 "cost_manifest_eth_h4_v2.json")
CONTEXT_BARS = 540
BARS_PER_YEAR = 2190
SIGMA_TARGET = 0.15
VOL_WINDOW = 180
H4_DEADLINE_SECONDS = 4 * 3600
ORIGINS = (2022, 2023, 2024)
ARMS = ("B0", "B1", "B2a", "B2b", "B3")
SEALED_START = "2025-01-01"

# Deployed protection geometry — the NAMED DIAGNOSTIC control
# (finding 328: 1% SL sits inside H4 intrabar noise):
FIXED_CONTROL_ENVELOPE = {
    "envelope_mode": "fixed_fraction",
    "sl_fraction": 0.01,
    "tp_fraction": 0.02,
    "collision_rule": "stop_first_pessimistic",
    "sizing_mode": "portfolio_fraction",
    "leverage_cap": 1.0,
    "source": ("deployed live-seat geometry "
               "(mt5_eth_sac_model_runner_v1); DIAGNOSTIC control in "
               "the WP3 calibration grid"),
}

# WP3: predeclared causal calibration grid — every cell is a trial.
ATR_WINDOW = 14
CALIBRATION_GRID = [dict(FIXED_CONTROL_ENVELOPE)] + [
    {"envelope_mode": "atr", "atr_window": ATR_WINDOW,
     "atr_sl_mult": sl, "atr_tp_mult": round(sl * ratio, 3),
     "collision_rule": "stop_first_pessimistic",
     "sizing_mode": "portfolio_fraction", "leverage_cap": 1.0,
     "source": f"WP3 grid SL={sl}xATR TP/SL={ratio}"}
    for sl in (1.5, 2.0, 3.0) for ratio in (1.5, 2.0)]
CALIBRATION_ARMS = ("B1", "B2a", "B2b", "B3")
CALIBRATION_COST_SET = "alpaca_ethusd"  # N2: the SOLE current G1
# economy — geometry must be selected under its own economics
# (finding 330); MT5 stays descriptive and is NOT recalibrated
# activity gates BEFORE economic ranking (declared):
MAX_ENVELOPE_FIRES_PER_YEAR = 1000    # pathological churn refusal
MIN_POSITION_EVENTS_PER_YEAR = 4      # no-activity refusal
RISK_LAMBDA = 1.0                     # hierarchical composite weight


class ScreenBError(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True,
                                     default=str).encode()).hexdigest()


def load_cost_sets() -> dict:
    m = json.loads(COST_MANIFEST.read_text())
    return {
        "alpaca_ethusd": {
            "binding": m["alpaca_ethusd"]["env_binding"],
            "g1_eligible": True,
            "authority": ("alpaca venue primary (pending "
                          "ratification)")},
        "mt5_ethusd": {
            "binding": m["mt5_ethusd"]["env_binding"],
            "g1_eligible": False,   # financing evidence gap blocks G1
            "authority": ("mt5 venue primary — G1-blocked by the "
                          "financing/swap evidence gap")},
        "zero_cost": {"binding": m["zero_cost"]["env_binding"],
                      "g1_eligible": False,
                      "authority": "VENUE_NEUTRAL_DIAGNOSTIC_ONLY"},
    }, _sha_file(COST_MANIFEST)


def load_source() -> pd.DataFrame:
    actual = _sha_file(DATA)
    if actual != DATA_SHA:
        raise ScreenBError(f"REFUSED: source sha {actual[:12]} != pinned")
    return pd.read_csv(DATA, parse_dates=["DATE_TIME"])


def materialize_origin(df: pd.DataFrame, year: int, out_dir: Path) -> dict:
    in_year = df.index[df["DATE_TIME"].dt.year == year]
    if len(in_year) == 0:
        raise ScreenBError(f"REFUSED: no rows for score year {year}")
    start, end = int(in_year[0]), int(in_year[-1])
    if start < CONTEXT_BARS:
        raise ScreenBError(f"REFUSED: not enough context before {year}")
    sl = df.iloc[start - CONTEXT_BARS:end + 1].reset_index(drop=True)
    if (sl["DATE_TIME"] >= SEALED_START).any():
        raise ScreenBError("REFUSED: sealed-period rows in origin slice")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"origin_{year}.csv"
    sl.to_csv(csv, index=False)
    scored = list(range(CONTEXT_BARS, len(sl)))
    scored_id = hashlib.sha256(("|".join(
        sl["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M").iloc[scored]))
        .encode()).hexdigest()
    return {"year": year, "csv": str(csv), "csv_sha256": _sha_file(csv),
            "rows": len(sl), "scored_rows": len(scored),
            "scored_start_index": CONTEXT_BARS,
            "scored_index_sha256": scored_id,
            "score_start": str(sl["DATE_TIME"].iloc[CONTEXT_BARS]),
            "score_end": str(sl["DATE_TIME"].iloc[-1])}


def rule_positions(close: np.ndarray, arm: str, scored_start: int
                   ) -> np.ndarray:
    """Signed TARGET EXPOSURE FRACTION per bar (C3). Strict lag:
    pos[t] uses close[<= t-1] only."""
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
                window = logret[t - VOL_WINDOW:t]
                sigma_ann = float(np.std(window, ddof=1)) * math.sqrt(
                    BARS_PER_YEAR)
                frac = 1.0 if sigma_ann <= 0 else min(
                    1.0, SIGMA_TARGET / sigma_ann)
                pos[t] = sign * frac
            else:
                pos[t] = sign
    return pos


def sigma_series(close: np.ndarray, scored_start: int) -> np.ndarray:
    logret = np.diff(np.log(close), prepend=np.nan)
    out = np.full(len(close), np.nan)
    for t in range(scored_start, len(close)):
        out[t] = float(np.std(logret[t - VOL_WINDOW:t], ddof=1)
                       ) * math.sqrt(BARS_PER_YEAR)
    return out


def base_config(origin: dict, cost_binding: dict,
                envelope: dict) -> dict:
    launch = json.loads(Path(
        "/home/harveybc/.local/share/agent-multi/"
        "l1_curriculum_campaign_20260823/seed101_N/"
        "normal_report.launch_manifest.json").read_text())
    cfg = dict(launch["effective_config"])
    for k in ("nested_split_contract", "nested_split_dir",
              "nested_split_manifest", "evaluate_test_split"):
        cfg.pop(k, None)
    cfg["input_data_file"] = origin["csv"]
    cfg["train_days"] = 1
    cfg["quiet_mode"] = True
    cfg["continuous_action_threshold"] = 0.0
    cfg["continuous_action_contract"] = "target_exposure_hysteresis_v2"
    cfg["strategy_plugin"] = "shared_execution_envelope"
    env_cfg = dict(envelope)
    # entry headroom scales with the COST BINDING (a fixed 0.2% was
    # smaller than alpaca's 30.5 bp/side and margin-rejected every
    # full-exposure long — counted, then refused, now fixed):
    per_side = float(cost_binding.get("commission", 0.0)) + float(
        cost_binding.get("slippage_perc", 0.0))
    # headroom = round-trip cost + decision-to-fill drift floor (H4
    # opens routinely gap ~0.5-1% from the decision close; a rejected
    # entry self-heals next bar at recomputed size and is COUNTED)
    env_cfg["entry_cost_headroom"] = round(2.0 * per_side + 0.006, 6)
    cfg["execution_envelope"] = env_cfg
    cfg.update(cost_binding)
    cfg.pop("env_mode", None)
    return cfg


def code_identity() -> dict:
    def git(repo, *args):
        return subprocess.run(["git", "-C", str(repo)] + list(args),
                              capture_output=True, text=True
                              ).stdout.strip()
    import app.env as gymfx_env
    gymfx_root = Path(gymfx_env.__file__).resolve().parents[1]
    return {
        "agent_multi_commit": git(REPO, "rev-parse", "HEAD"),
        # untracked evidence outputs of THIS run live in-repo; the
        # clean-tree proof covers TRACKED content only
        "agent_multi_clean_tree": git(
            REPO, "status", "--porcelain", "--untracked-files=no") == "",
        "gymfx_origin": str(gymfx_root),
        "gymfx_commit": git(gymfx_root, "rev-parse", "HEAD"),
        "gymfx_clean_tree": git(gymfx_root, "status", "--porcelain",
                                "--untracked-files=no") == "",
    }


def trial_id(arm: str, origin: dict, cost_set: str,
             envelope_sha: str, cost_sha: str) -> str:
    return hashlib.sha256(
        f"B|{arm}|{origin['year']}|{origin['csv_sha256']}|{cost_set}|"
        f"{cost_sha}|{envelope_sha}".encode()).hexdigest()[:32]


def register_trials(ledger: Path, rows: list) -> None:
    """Idempotent (C5): same id + same content -> skip; same id with
    DIFFERENT content -> refuse; new id -> append."""
    existing = {}
    if ledger.is_file():
        for line in ledger.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                existing[d["trial_id"]] = d
    with open(ledger, "a") as fh:
        for row in rows:
            prev = existing.get(row["trial_id"])
            if prev is not None:
                if prev != row:
                    raise ScreenBError(
                        f"REFUSED: trial {row['trial_id']} already "
                        f"registered with DIFFERENT content")
                continue
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def validate_stats_inputs(results: list) -> list:
    """C5: DSR/SPA inputs may include ONLY g1-eligible,
    non-diagnostic arms."""
    ok = [r for r in results if r.get("g1_eligible") is True]
    if not ok:
        raise ScreenBError("REFUSED: no g1-eligible arms among the "
                           "statistics inputs")
    return ok


def run_arm(origin: dict, arm: str, out_dir: Path, cost_set: str,
            cost_binding: dict, cost_authority: str, g1_eligible: bool,
            cost_sha: str, envelope: dict, envelope_sha: str,
            tag: str = "") -> dict:
    sys.path.insert(0, str(REPO))
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)
    df = pd.read_csv(origin["csv"])
    close = df["CLOSE"].to_numpy(dtype=float)
    pos = rule_positions(close, arm, origin["scored_start_index"])
    sig = sigma_series(close, origin["scored_start_index"])
    cfg = base_config(origin, cost_binding, envelope)
    env = _load_env_plugin("gym_fx_env", cfg).make_env(cfg)
    obs, _ = env.reset(seed=0)
    inner = env
    while not hasattr(inner, "bridge") and hasattr(inner, "env"):
        inner = inner.env
    rows, step_times = [], []
    equity_prev = None
    events_seen = 0
    for t in range(len(close)):
        t0 = time.perf_counter()
        obs, _r, term, trunc, info = env.step([float(pos[t])])
        step_times.append(time.perf_counter() - t0)
        econ = float(info.get("economic_equity", np.nan))
        events = list(getattr(inner.bridge, "close_events", []))
        new_events = events[events_seen:]
        events_seen = len(events)
        if t >= origin["scored_start_index"]:
            ret = (0.0 if equity_prev in (None, 0.0)
                   else econ / equity_prev - 1.0)
            units = float(getattr(inner.bridge, "position_units", 0.0)
                          or 0.0)
            rows.append({
                "bar": t, "datetime": df["DATE_TIME"].iloc[t],
                "requested_exposure": float(pos[t]),
                "position_units": units,
                "realized_exposure": (units * close[t] / econ
                                      if econ else 0.0),
                "sigma_ann": (float(sig[t])
                              if not np.isnan(sig[t]) else None),
                "net_return": ret, "economic_equity": econ,
                "commission_paid_cum": float(
                    info.get("commission_paid") or 0.0),
                "close_reasons": ";".join(e["reason"]
                                          for e in new_events),
            })
        equity_prev = econ
        if term or trunc:
            break
    # WP1: any lifecycle failure or residual sweep is a TYPED RUN
    # FAILURE — never accepted evidence.
    failure = getattr(inner.bridge, "envelope_run_failure", None)
    sweeps = int(inner.bridge.execution_diagnostics.get(
        "envelope_residual_sweeps", 0) or 0)
    rejections = int(inner.bridge.execution_diagnostics.get(
        "envelope_order_rejections", 0) or 0)
    if failure or sweeps:
        raise ScreenBError(
            f"REFUSED_RUN: envelope lifecycle failure arm={arm} "
            f"origin={origin['year']} tag={tag!r} failure={failure!r} "
            f"residual_sweeps={sweeps}")
    final_pos = float(getattr(inner.bridge, "position", 0.0) or 0.0)
    if final_pos != 0.0 and rows:
        rows[-1]["close_reasons"] = (
            (rows[-1]["close_reasons"] + ";" if rows[-1]["close_reasons"]
             else "") + "data_end_liquidation")
    per_bar = pd.DataFrame(rows)
    csv = out_dir / f"{arm}_{origin['year']}_{cost_set}{tag}_per_bar.csv"
    per_bar.to_csv(csv, index=False)
    r = per_bar["net_return"].to_numpy()
    eq = per_bar["economic_equity"].to_numpy()
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max(1.0 - eq / peak)) if len(eq) else None
    mean, std = float(np.mean(r)), float(np.std(r, ddof=1))
    sharpe = (mean / std * math.sqrt(BARS_PER_YEAR)) if std > 0 else 0.0
    req = per_bar["requested_exposure"].to_numpy()
    realized = per_bar["realized_exposure"].to_numpy()
    reasons = ";".join(x for x in per_bar["close_reasons"] if x)
    counts = {}
    for token in reasons.split(";"):
        if token:
            counts[token] = counts.get(token, 0) + 1
    st = sorted(step_times)
    def pct(p):
        return st[min(len(st) - 1, int(p * len(st)))] if st else None
    return {
        "arm": arm, "origin": origin["year"], "cost_set": cost_set,
        "cost_authority": cost_authority, "g1_eligible": g1_eligible,
        "scored_bars": int(len(per_bar)),
        "net_total_return": float(eq[-1] / eq[0] - 1.0) if len(eq)
        else None,
        "net_sharpe_annualized": sharpe,
        "realized_strategy_vol_annualized": float(
            std * math.sqrt(BARS_PER_YEAR)),
        "max_drawdown_fraction": mdd,
        "turnover_sum_abs_dreq": float(
            np.abs(np.diff(req, prepend=0.0)).sum()),
        "median_abs_requested_exposure": float(
            np.median(np.abs(req[req != 0])) if (req != 0).any() else 0.0),
        "median_abs_realized_exposure": float(
            np.median(np.abs(realized[realized != 0]))
            if (realized != 0).any() else 0.0),
        "close_reason_counts": counts,
        "entry_order_rejections_healed": rejections,
        "total_commission_paid": float(
            per_bar["commission_paid_cum"].iloc[-1] if len(per_bar)
            else 0.0),
        "per_bar_csv": str(csv), "per_bar_sha256": _sha_file(csv),
        "effective_config_sha256": _sha_obj(cfg),
        "cost_manifest_sha256": cost_sha,
        "execution_envelope_sha256": envelope_sha,
        "scored_index_sha256": origin["scored_index_sha256"],
        "decision_step_seconds_p50": pct(0.50),
        "decision_step_seconds_p95": pct(0.95),
        "h4_deadline_met": (pct(0.95) or 0) < H4_DEADLINE_SECONDS,
        "trial_id": trial_id(arm, origin, cost_set, envelope_sha,
                             cost_sha),
    }


def materialize_calibration_slice(df: pd.DataFrame, year: int,
                                  out_dir: Path) -> dict:
    """WP3: fit/monitor-only calibration window = the 12 months BEFORE
    the origin's score year (context 540 bars). Strictly causal: no
    scored-year bar enters geometry selection."""
    return materialize_origin(df, year - 1, out_dir)


def envelope_criterion(results: list) -> dict:
    """Hierarchical activity/economic criterion: activity gates first,
    then median across arms of (net_return - RISK_LAMBDA * mdd)."""
    fires = [sum(v for k, v in r["close_reason_counts"].items()
                 if k.startswith("envelope_close")) for r in results]
    moves = [sum(r["close_reason_counts"].values()) for r in results]
    med_fires = statistics.median(fires)
    med_moves = statistics.median(moves)
    if med_fires > MAX_ENVELOPE_FIRES_PER_YEAR:
        return {"eligible": False, "refusal": "pathological_churn",
                "median_envelope_fires": med_fires}
    if med_moves < MIN_POSITION_EVENTS_PER_YEAR:
        return {"eligible": False, "refusal": "no_activity",
                "median_position_events": med_moves}
    comp = statistics.median(
        [r["net_total_return"] - RISK_LAMBDA *
         (r["max_drawdown_fraction"] or 0.0) for r in results])
    return {"eligible": True, "composite_median": comp,
            "median_envelope_fires": med_fires,
            "median_position_events": med_moves}


def calibrate_origin_envelope(df: pd.DataFrame, origin: dict,
                              out: Path, cost_sets: dict,
                              cost_sha: str, ledger: Path) -> dict:
    """Run the predeclared grid on the CALIBRATION slice only; freeze
    the winner BEFORE the score year. Every cell is a ledger trial."""
    cal = materialize_calibration_slice(df, origin["year"],
                                        out / "calibration_origins")
    spec = cost_sets[CALIBRATION_COST_SET]
    cells = []
    rows = []
    for gi, env_geom in enumerate(CALIBRATION_GRID):
        esha = _sha_obj(env_geom)
        for arm in CALIBRATION_ARMS:
            rows.append({
                "trial_id": trial_id(arm, cal,
                                     f"cal_{CALIBRATION_COST_SET}",
                                     esha, cost_sha),
                "screen": "B_envelope_calibration", "arm": arm,
                "origin": origin["year"], "geometry_index": gi,
                "calibration_year": cal["year"],
                "registered_before_results": True,
                "origin_csv_sha256": cal["csv_sha256"],
                "scored_index_sha256": cal["scored_index_sha256"],
                "execution_envelope_sha256": esha,
                "cost_manifest_sha256": cost_sha})
    register_trials(ledger, rows)
    for gi, env_geom in enumerate(CALIBRATION_GRID):
        esha = _sha_obj(env_geom)
        results = []
        (out / "calibration_runs").mkdir(parents=True, exist_ok=True)
        for arm in CALIBRATION_ARMS:
            print(f"== cal o{origin['year']} geom{gi} {arm}",
                  flush=True)
            results.append(run_arm(
                cal, arm, out / "calibration_runs",
                f"cal_{CALIBRATION_COST_SET}", spec["binding"],
                "calibration (never scored-origin evidence)", False,
                cost_sha, env_geom, esha, tag=f"_g{gi}"))
        crit = envelope_criterion(results)
        cells.append({"geometry_index": gi, "geometry": env_geom,
                      "envelope_sha256": esha, "criterion": crit})
    eligible = [c for c in cells if c["criterion"]["eligible"]]
    if not eligible:
        raise ScreenBError(
            f"REFUSED: no eligible envelope geometry for origin "
            f"{origin['year']} (all cells churn/no-activity)")
    winner = max(eligible,
                 key=lambda c: c["criterion"]["composite_median"])
    record = {"origin": origin["year"],
              "calibration_year": cal["year"],
              "calibration_csv_sha256": cal["csv_sha256"],
              "grid_cells": cells,
              "frozen_geometry_index": winner["geometry_index"],
              "frozen_geometry": winner["geometry"],
              "frozen_envelope_sha256": winner["envelope_sha256"],
              "frozen_before_score_year": True}
    (out / f"ENVELOPE_CALIBRATION_o{origin['year']}.json").write_text(
        json.dumps(record, indent=1))
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--origins", default=",".join(map(str, ORIGINS)))
    ap.add_argument("--cost-sets",
                    default="alpaca_ethusd,zero_cost")
    args = ap.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cost_sets, cost_sha = load_cost_sets()
    df = load_source()
    origins = [materialize_origin(df, int(y), out / "origins")
               for y in args.origins.split(",")]
    arms = args.arms.split(",")
    wanted = args.cost_sets.split(",")
    identity = code_identity()
    ledger = out / "trial_ledger.jsonl"
    calibrations = {o["year"]: calibrate_origin_envelope(
        df, o, out, cost_sets, cost_sha, ledger) for o in origins}
    run_manifest = {
        "schema": "agent_multi.screen_b_run_manifest.v3",
        "code_identity": identity,
        "source_data_sha256": DATA_SHA,
        "cost_manifest_sha256": cost_sha,
        "calibration_grid": CALIBRATION_GRID,
        "calibration_cost_set": CALIBRATION_COST_SET,
        "frozen_geometry_by_origin": {
            y: {"geometry": c["frozen_geometry"],
                "envelope_sha256": c["frozen_envelope_sha256"],
                "calibrated_on_year": c["calibration_year"]}
            for y, c in calibrations.items()},
        "origins": origins,
        "arms": arms, "cost_sets": wanted,
    }
    (out / "RUN_MANIFEST.json").write_text(json.dumps(run_manifest,
                                                      indent=1))
    pre = []
    for o in origins:
        esha = calibrations[o["year"]]["frozen_envelope_sha256"]
        for arm in arms:
            for cs in wanted:
                pre.append({
                    "trial_id": trial_id(arm, o, cs, esha, cost_sha),
                    "screen": "B", "arm": arm, "origin": o["year"],
                    "cost_set": cs,
                    "registered_before_results": True,
                    "origin_csv_sha256": o["csv_sha256"],
                    "scored_index_sha256": o["scored_index_sha256"],
                    "execution_envelope_sha256": esha,
                    "cost_manifest_sha256": cost_sha,
                })
    register_trials(ledger, pre)
    results = []
    for o in origins:
        geom = calibrations[o["year"]]["frozen_geometry"]
        esha = calibrations[o["year"]]["frozen_envelope_sha256"]
        for arm in arms:
            for cs in wanted:
                spec = cost_sets[cs]
                print(f"== {arm} origin {o['year']} [{cs}]", flush=True)
                results.append(run_arm(
                    o, arm, out, cs, spec["binding"],
                    spec["authority"], spec["g1_eligible"],
                    cost_sha, geom, esha))
    packet = {"schema": "agent_multi.screen_b_rule_arms.v3",
              "run_manifest_sha256": _sha_file(out / "RUN_MANIFEST.json"),
              "sealed_2025_used": False,
              "g1_claim": "NOT_EMITTED (B4 absent)",
              "g1_eligible_arms": [r["trial_id"] for r in
                                   validate_stats_inputs(results)],
              "results": results,
              "trial_ledger": str(ledger)}
    (out / "SCREEN_B_RESULTS.json").write_text(json.dumps(
        packet, indent=1))
    print(json.dumps({f"{r['arm']}@{r['origin']}:{r['cost_set']}":
                      round(r["net_sharpe_annualized"], 3)
                      for r in results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
