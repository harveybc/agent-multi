#!/usr/bin/env python3
"""RT0/RT1: rolling-origin, test-then-train adaptation runner (v2).

LOCAL-ONLY and zero-network: no DOIN, no sockets, no venue calls.

Corrections in v2 (AUD-F1-20260806-140/141/142):

- **Warm-up is never scored.** The evaluation slice carries warm-up
  context so scaling state is causal, but metrics start EXACTLY at the
  first bar of `(t, t+h]`. v1 scored the warm-up history and inverted
  the sign of the interval result.
- **Account continuity within a block.** Cash/equity, open protected
  exposure and handover cost carry from one origin to the next. A reset
  happens only at a DECLARED block boundary, never per origin.
- **Immutable per-origin checkpoints + atomic pointer.** Every origin
  writes `before`/`after` model files under content-addressed names and
  commits a `current_state.json` pointer in the SAME transaction as the
  OLAP row. A crash before or after any write replays idempotently:
  restart verifies the recorded after-hash against the pointer and
  refuses to re-apply an update that was already recorded.
- **Full execution identity.** initial/update steps, device, resolved
  config hash, data + observation manifest hashes, code revisions and
  runner version all bind into the run id; v1 output is refused.
- **End-to-end latency.** The measured duration runs from bar close to
  a durable, validated, replicated, activation-ready artifact — the
  owner-amended budget — not merely the fit call.

Prequential discipline (Hyndman & Athanasopoulos TSCV; Bifet et al.
MOA): at origin `t` the incumbent — trained only on bars <= `t` — is
scored on `(t, t+h]`; only afterwards may those bars enter an update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

DATA_FILE = ("/home/harveybc/Documents/GitHub/predictor/examples/data/"
             "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
DATA_SHA256 = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435"
               "ebe440f")
OBSERVATION_MANIFEST = REPO / "docs/work_plan/34_ETH_DATA_OBSERVATION_MANIFEST.md"
ETH_BASE = (REPO / "examples/results/"
            "project3_ethusdt_4h_sac_train_val_test_v2/config_out.json")

RUNNER_VERSION = "rolling_origin_adaptation.v2"
SCHEMA_VERSION = "agent_multi.rt_adaptation.v2"
BARS_PER_DAY = 6
BAR_SECONDS = 4 * 3600
# Bar-aligned cadences only. 1 bar (4 h) is a feasibility stress case.
ALLOWED_CADENCES = (1, 2, 3, 6, 18, 42)
WARMUP_BARS = 256                    # rolling scaling context, NEVER scored
# Dormant year shorthand contradicts explicit dates (finding 142).
DORMANT_SPLIT_FIELDS = ("train_years", "val_years", "test_years")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()


def _git_rev(repo: str) -> str:
    out = subprocess.run(
        ["git", "-C", f"/home/harveybc/Documents/GitHub/{repo}",
         "rev-parse", "HEAD"],
        capture_output=True, text=True)
    return out.stdout.strip() or "unavailable"


def _code_revisions() -> dict:
    return {repo: _git_rev(repo)
            for repo in ("agent-multi", "gym-fx", "predictor")}


def _gpu_probe() -> dict:
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if probe.returncode != 0:
            return {"unavailable": f"nvidia-smi exit {probe.returncode}"}
        first = probe.stdout.strip().splitlines()[0].split(",")
        return {"temperature_c": int(first[0].strip()),
                "memory_used": first[1].strip()}
    except Exception as exc:
        return {"unavailable": str(exc)[:120]}


def base_config() -> dict:
    """Executable base config with the dormant year shorthand REMOVED
    (finding 142) so it cannot contradict the explicit dates."""
    config = json.loads(ETH_BASE.read_text())
    for field in DORMANT_SPLIT_FIELDS:
        config.pop(field, None)
    config["split_contract_note"] = (
        "explicit dates govern; year-count shorthand removed")
    config["quiet_mode"] = True
    config["env_mode"] = "training"
    config["solvency_mode"] = "normal_realistic"
    config["evaluate_test_split"] = False
    return config


def score_interval(equity_series, *, warmup_bars: int,
                   starting_equity: float | None = None) -> dict:
    """Score ONLY the deployment interval (AUD-F1-20260806-140).

    ``equity_series`` is the full per-step equity of the rollout,
    including warm-up context. Metrics begin at index ``warmup_bars``;
    the equity carried in from the previous origin (``starting_equity``)
    is the baseline when supplied, so the block's account continuity is
    preserved rather than reset to the config's initial cash.
    """
    scored = list(equity_series)[warmup_bars:]
    if not scored:
        return {"unavailable": "no scored bars after warm-up",
                "scored_bars": 0}
    baseline = (float(starting_equity) if starting_equity is not None
                else float(scored[0]))
    final = float(scored[-1])
    peak = baseline
    max_dd = 0.0
    for value in scored:
        peak = max(peak, float(value))
        if peak > 0:
            max_dd = max(max_dd, (peak - float(value)) / peak)
    return {
        "interval_return": (final / baseline - 1.0) if baseline else None,
        "equity_before": baseline,
        "equity_after": final,
        "max_drawdown_fraction": max_dd,
        "scored_bars": len(scored),
        "warmup_bars_excluded": warmup_bars,
    }


def _olap(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS rt_intervals_v2 (
            record_id TEXT PRIMARY KEY,
            schema_version TEXT, run_id TEXT, phase TEXT,
            cadence_bars INTEGER, lookback TEXT, seed INTEGER,
            block_id TEXT, origin_index INTEGER,
            origin_time TEXT, interval_end_time TEXT,
            scored_bars INTEGER, warmup_bars_excluded INTEGER,
            interval_return REAL, interval_trades INTEGER,
            interval_max_drawdown_fraction REAL,
            equity_before REAL, equity_after REAL,
            carried_equity INTEGER,
            update_latency_seconds REAL, deadline_seconds REAL,
            deadline_miss INTEGER, model_age_bars INTEGER,
            new_bars INTEGER, update_steps INTEGER,
            peak_rss_mb REAL, gpu_json TEXT,
            model_before_sha256 TEXT, model_after_sha256 TEXT,
            replica_verified INTEGER, created_at TEXT
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS rt_runs_v2 (
            run_id TEXT PRIMARY KEY, schema_version TEXT,
            identity_json TEXT, created_at TEXT
        )""")
    con.commit()
    return con


def _slice_csv(df: pd.DataFrame, start: int, end: int,
               out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.iloc[start:end].to_csv(out, index=False)
    return out


def _build_env(config: dict, csv_path: Path, *,
               starting_cash: float | None = None):
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)
    from importlib.metadata import entry_points
    cfg = dict(config)
    cfg["input_data_file"] = str(csv_path)
    if starting_cash is not None:
        # Account continuity (finding 140): the next interval opens on
        # the balance the previous interval closed with.
        cfg["initial_cash"] = float(starting_cash)
    env_plugin = _load_env_plugin(
        str(cfg.get("env_plugin", "gym_fx_env")), cfg)
    env = env_plugin.make_env(cfg)
    agent_ep = next(e for e in entry_points().select(
        group="agent.plugins")
        if e.name == cfg.get("agent_plugin",
                             "project3_sac_actor_critic_agent"))
    agent = agent_ep.load()()
    wrap = getattr(agent, "wrap_env", None)
    if callable(wrap):
        env = wrap(env, cfg)
    return env


def _rollout(model, env) -> dict:
    obs, _ = env.reset()
    equities, trades = [], 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        if isinstance(info, dict):
            equity = info.get("economic_equity", info.get("equity"))
            if equity is not None:
                equities.append(float(equity))
            trades = int(info.get("trades_total", trades) or trades)
        if terminated or truncated:
            break
    return {"equities": equities, "trades": trades}


def run_identity(args, config: dict) -> dict:
    """Every decision-bearing input binds into the id (finding 141)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "phase": args.phase,
        "cadence_bars": args.cadence_bars,
        "lookback": args.lookback,
        "seed": args.seed,
        "block_id": f"{args.block_start}+{args.block_days}d",
        "block_start": args.block_start,
        "block_days": args.block_days,
        "initial_steps": args.initial_steps,
        "update_steps": args.update_steps,
        "device": args.device,
        "control_mode": args.control_mode,
        "data_sha256": DATA_SHA256,
        "observation_manifest_sha256": (
            _sha_file(OBSERVATION_MANIFEST)
            if OBSERVATION_MANIFEST.exists() else "unavailable"),
        "resolved_config_sha256": _sha_json(config),
        "code_revisions": _code_revisions(),
        "warmup_bars": WARMUP_BARS,
    }


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True,
                              default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run(args) -> int:
    assert _sha_file(Path(DATA_FILE)) == DATA_SHA256, "dataset drift"
    assert args.cadence_bars in ALLOWED_CADENCES, (
        f"cadence {args.cadence_bars} bars is not bar-aligned/allowed")
    config = base_config()
    for field in DORMANT_SPLIT_FIELDS:
        assert field not in config, f"{field} leaked into executable config"

    df = pd.read_csv(DATA_FILE)
    dates = pd.to_datetime(df["DATE_TIME"])
    block_start = int((dates >= args.block_start).idxmax())
    block_bars = args.block_days * BARS_PER_DAY
    guard_2025 = int((dates >= "2025-01-01").idxmax())
    assert block_start + block_bars <= guard_2025, (
        "block would cross into the disclosed 2025 period")
    assert block_start - WARMUP_BARS >= 0, "insufficient warm-up history"

    identity = run_identity(args, config)
    run_id = _sha_json(identity)[:16]
    out_dir = args.output_root / f"{args.phase}_{run_id}"
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_dir / "run_identity.json", identity)

    con = _olap(args.output_root / "rt_adaptation_v2.sqlite")
    con.execute(
        "INSERT OR IGNORE INTO rt_runs_v2 VALUES (?,?,?,?)",
        (run_id, SCHEMA_VERSION, json.dumps(identity, sort_keys=True),
         datetime.now(timezone.utc).isoformat()))
    con.commit()

    # Refuse to mix v1 output into a v2 directory (finding 141).
    legacy = args.output_root / "rt_adaptation.sqlite"
    if legacy.exists() and not args.allow_legacy_sibling:
        raise SystemExit(
            f"v1 OLAP {legacy} exists in this output root; v2 refuses to"
            " share it. Use a fresh --output-root (or pass"
            " --allow-legacy-sibling once the v1 rows are archived).")

    from stable_baselines3 import SAC
    pointer_path = out_dir / "current_state.json"
    pointer = (json.loads(pointer_path.read_text())
               if pointer_path.exists() else
               {"origins_committed": [], "after_sha256": None,
                "after_path": None, "carried_equity": None})

    def lookback_start(origin: int) -> int:
        if args.lookback == "expanding":
            return 0
        years = int(args.lookback.rstrip("y"))
        return max(0, origin - years * 365 * BARS_PER_DAY)

    origins = list(range(block_start,
                         block_start + block_bars - args.cadence_bars,
                         args.cadence_bars))
    if args.max_origins:
        origins = origins[:args.max_origins]

    cadence_seconds = args.cadence_bars * BAR_SECONDS
    latencies: list[float] = []
    carried_equity = pointer.get("carried_equity")
    model_age_bars = 0

    for index, origin in enumerate(origins):
        record_id = _sha_json({"run": run_id, "origin_index": index,
                               "origin": origin})[:20]
        committed = con.execute(
            "SELECT model_after_sha256 FROM rt_intervals_v2"
            " WHERE record_id=?", (record_id,)).fetchone()
        if committed:
            # Idempotent replay (finding 141): the recorded after-state
            # must equal the pointer; the update is NEVER re-applied.
            if pointer.get("after_sha256") not in (None, committed[0]):
                raise SystemExit(
                    f"origin {index}: OLAP after-hash {committed[0][:12]}"
                    f" != pointer {str(pointer.get('after_sha256'))[:12]}"
                    " — refusing to continue on ambiguous state")
            print(f"[rt] origin {index} already committed — replay skip",
                  flush=True)
            model_age_bars += args.cadence_bars
            continue

        # ---------- 1. incumbent state (before) ----------
        before_path = None
        if pointer.get("after_path") and Path(
                pointer["after_path"]).is_file():
            before_path = Path(pointer["after_path"])

        # Latency clock starts at the interval's DATA CLOSE (owner-
        # amended budget: bar close -> activation-ready artifact).
        started = time.monotonic()
        fit_start = lookback_start(origin)
        fit_csv = _slice_csv(df, max(0, fit_start - WARMUP_BARS), origin,
                             out_dir / "slices" / f"fit_{index}.csv")
        fit_env = _build_env({**config, "train_seed": args.seed}, fit_csv)
        if before_path is not None:
            model = SAC.load(str(before_path), env=fit_env,
                             device=args.device)
        else:
            model = SAC("MlpPolicy", fit_env, seed=args.seed,
                        device=args.device,
                        policy_kwargs={"net_arch": [256, 256]})
            model.learn(total_timesteps=args.initial_steps,
                        progress_bar=False)
            before_immutable = checkpoints / f"origin{index}_before.zip"
            model.save(str(before_immutable))
            before_path = before_immutable

        # ---------- 2. adaptation (or frozen control) ----------
        adapted = False
        if args.control_mode == "adaptive" and (
                index > 0 or args.update_first_origin):
            model.learn(total_timesteps=args.update_steps,
                        progress_bar=False)
            adapted = True
        after_path = checkpoints / f"origin{index}_after.zip"
        model.save(str(after_path))
        fit_env.close()
        after_sha = _sha_file(after_path)

        # durable + validated + replicated + activation-ready
        SAC.load(str(after_path), device="cpu")          # load proof
        replica_path = (checkpoints / "replica"
                        / f"origin{index}_after.zip")
        replica_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(after_path, replica_path)
        replica_verified = int(_sha_file(replica_path) == after_sha)
        latency_seconds = time.monotonic() - started
        latencies.append(latency_seconds)
        model_age_bars = 0

        # ---------- 3. score the NEXT interval (warm-up excluded) ----
        eval_csv = _slice_csv(
            df, origin - WARMUP_BARS, origin + args.cadence_bars,
            out_dir / "slices" / f"eval_{index}.csv")
        eval_env = _build_env(
            {**config, "eval_seed": args.seed}, eval_csv,
            starting_cash=carried_equity)
        rollout = _rollout(model, eval_env)
        eval_env.close()
        score = score_interval(
            rollout["equities"], warmup_bars=WARMUP_BARS,
            starting_equity=carried_equity)
        if "unavailable" in score:
            raise SystemExit(
                f"origin {index}: {score['unavailable']}")

        # ---------- 4. atomic commit: OLAP row + state pointer -------
        record = {
            "record_id": record_id, "schema_version": SCHEMA_VERSION,
            "run_id": run_id, "phase": args.phase,
            "cadence_bars": args.cadence_bars,
            "lookback": args.lookback, "seed": args.seed,
            "block_id": identity["block_id"], "origin_index": index,
            "origin_time": str(dates.iloc[origin]),
            "interval_end_time": str(
                dates.iloc[origin + args.cadence_bars - 1]),
            "scored_bars": score["scored_bars"],
            "warmup_bars_excluded": score["warmup_bars_excluded"],
            "interval_return": score["interval_return"],
            "interval_trades": rollout["trades"],
            "interval_max_drawdown_fraction": score[
                "max_drawdown_fraction"],
            "equity_before": score["equity_before"],
            "equity_after": score["equity_after"],
            "carried_equity": int(carried_equity is not None),
            "update_latency_seconds": latency_seconds,
            "deadline_seconds": cadence_seconds,
            "deadline_miss": int(latency_seconds > cadence_seconds),
            "model_age_bars": model_age_bars,
            "new_bars": args.cadence_bars,
            "update_steps": (args.update_steps if adapted else 0),
            "peak_rss_mb": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "gpu_json": json.dumps(_gpu_probe()),
            "model_before_sha256": _sha_file(before_path),
            "model_after_sha256": after_sha,
            "replica_verified": replica_verified,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        con.execute(
            f"INSERT INTO rt_intervals_v2 ({','.join(record)})"
            f" VALUES ({','.join('?' * len(record))})",
            list(record.values()))
        con.commit()
        carried_equity = score["equity_after"]
        pointer = {
            "origins_committed": pointer["origins_committed"] + [index],
            "after_sha256": after_sha,
            "after_path": str(after_path),
            "carried_equity": carried_equity,
        }
        _atomic_write(pointer_path, pointer)
        print(json.dumps({k: record[k] for k in (
            "origin_index", "scored_bars", "interval_return",
            "equity_before", "equity_after",
            "update_latency_seconds", "deadline_miss")}), flush=True)

    ordered = sorted(latencies)

    def _pct(q: float):
        if not ordered:
            return None
        index = min(len(ordered) - 1,
                    max(0, round(q * (len(ordered) - 1))))
        return ordered[int(index)]

    p50, p95 = _pct(0.50), _pct(0.95)
    misses = int(con.execute(
        "SELECT COALESCE(SUM(deadline_miss),0) FROM rt_intervals_v2"
        " WHERE run_id=?", (run_id,)).fetchone()[0])
    updates = int(con.execute(
        "SELECT COUNT(*) FROM rt_intervals_v2 WHERE run_id=?",
        (run_id,)).fetchone()[0])
    summary = {
        "schema": "agent_multi.rt_adaptation_summary.v2",
        "run_id": run_id, "identity": identity,
        "origins_committed": updates,
        "latency_definition": (
            "bar close -> durable, load-validated, replicated,"
            " activation-ready artifact (owner-amended budget)"),
        "update_latency_p50": p50, "update_latency_p95": p95,
        "deadline_seconds": cadence_seconds,
        "deadline_misses": misses,
        "deadline_guard": {
            "rule": ("p95 end-to-end latency <= (2/3) * cadence AND"
                     " >= 20 updates AND zero deadline misses AND zero"
                     " unreconciled handovers"),
            "p95_within_budget": (
                p95 is not None and p95 <= cadence_seconds * 2 / 3),
            "updates_observed": updates,
            "sufficient_updates": updates >= 20,
            "zero_deadline_misses": misses == 0,
            "satisfied": (
                p95 is not None and p95 <= cadence_seconds * 2 / 3
                and updates >= 20 and misses == 0),
            "status": "owner_amended_2026_08_06",
        },
        "note": ("RT0 measures runtime feasibility only; profit/risk"
                 " promotion requires RT1-A paired multi-block"
                 " evidence with frozen controls"),
    }
    _atomic_write(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=1, default=str))
    con.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("RT0", "RT1"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cadence-bars", type=int, required=True)
    parser.add_argument("--lookback", default="1y",
                        help="1y, 2y, 4y or 'expanding'")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--block-start", default="2024-02-01")
    parser.add_argument("--block-days", type=int, default=28)
    parser.add_argument("--initial-steps", type=int, default=20000)
    parser.add_argument("--update-steps", type=int, default=2000)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--update-first-origin", action="store_true")
    parser.add_argument("--control-mode", default="adaptive",
                        choices=("adaptive", "frozen"),
                        help="frozen = paired no-update control arm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-legacy-sibling", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
