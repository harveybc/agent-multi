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


def block_origins(block_start: int, block_bars: int,
                  cadence_bars: int) -> tuple[list[int], int]:
    """Half-open intervals whose union is exactly the declared block.

    AUD-F1-20260806-157: the previous range stopped one interval short,
    so a 28-day block evaluated 83/84 eight-hour intervals and only 3/4
    weekly intervals — a cadence-dependent bias against long cadences.
    Every interval satisfies ``interval_end <= block_end`` with no gap
    and no overlap; any non-divisible remainder is returned explicitly
    rather than silently dropped.
    """
    if cadence_bars <= 0:
        raise ValueError("cadence must be positive")
    count = block_bars // cadence_bars
    remainder = block_bars - count * cadence_bars
    origins = [block_start + index * cadence_bars
               for index in range(count)]
    return origins, remainder


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


def score_interval(samples, *, warmup_bars: int, cadence_bars: int,
                   starting_equity: float | None = None,
                   handover: dict | None = None) -> dict:
    """Score EXACTLY the deployment interval (140/145).

    ``samples`` is the ordered per-step fact list of the rollout, each
    entry carrying at least ``equity``; warm-up entries are context
    only. Corrections in this version:

    - warm-up entries are dropped BEFORE any metric, and the warm-up
      steps are executed as forced holds so they cannot trade at all;
    - exactly ``cadence_bars`` decision bars are scored (`h`, never
      `h+1`); a terminal duplicate fact is discarded;
    - activity is an interval DELTA (trades/commission at the end minus
      at the warm-up boundary), never a cumulative total;
    - the handover is explicit: any exposure open at the end of the
      interval is closed at the last price and charged the configured
      commission, and the post-close balance is what carries forward.
    """
    if warmup_bars < 0 or cadence_bars <= 0:
        return {"unavailable": "invalid warm-up/cadence"}
    scored = list(samples)[warmup_bars:]
    if len(scored) < cadence_bars:
        return {"unavailable":
                f"only {len(scored)} scored samples for h="
                f"{cadence_bars}", "scored_bars": len(scored)}
    scored = scored[:cadence_bars]                 # exactly h bars
    boundary = (list(samples)[warmup_bars - 1] if warmup_bars > 0
                else {"trades": 0, "commission_paid": 0.0})

    equities = [float(s["equity"]) for s in scored]
    baseline = (float(starting_equity) if starting_equity is not None
                else equities[0])
    last = scored[-1]
    final_equity = equities[-1]

    # ---- explicit flat handover (152) -------------------------------
    # The close is EXECUTED by the simulator and its facts are passed in
    # by the caller; this function never invents a cost from a direction
    # flag. `handover` is required for a training interval.
    if handover is None:
        return {"unavailable":
                "no simulator handover facts supplied; refusing to"
                " assert a flat close (finding 152)"}
    if handover.get("flat_proven") is not True:
        return {"unavailable":
                f"handover not proven flat: {handover.get('reason')}"}
    post_close_equity = float(handover["post_close_equity"])

    peak = baseline
    max_dd = 0.0
    for value in equities:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)

    def _delta(key: str) -> float:
        return (float(last.get(key, 0) or 0)
                - float(boundary.get(key, 0) or 0))

    return {
        "interval_return": (
            (post_close_equity / baseline - 1.0) if baseline else None),
        "equity_before": baseline,
        "equity_at_interval_end": final_equity,
        "equity_after": post_close_equity,
        "handover": handover,
        "max_drawdown_fraction": max_dd,
        "scored_bars": len(scored),
        "warmup_bars_excluded": warmup_bars,
        "interval_trades": _delta("trades"),
        "interval_commission": _delta("commission_paid"),
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
            replica_verified INTEGER,
            interval_commission REAL, handover_json TEXT,
            handover_requested_at TEXT, handover_flat_proven_at TEXT,
            artifact_ready_at TEXT, activated_at TEXT,
            unreconciled_handovers INTEGER, activation_delay_bars REAL,
            rollback_status TEXT, warmup_traded INTEGER,
            anchor_sha256 TEXT, source_tree_digest TEXT,
            expected_before_sha256 TEXT, succession_ok INTEGER,
            created_at TEXT
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS rt_runs_v2 (
            run_id TEXT PRIMARY KEY, schema_version TEXT,
            identity_json TEXT, created_at TEXT
        )""")
    # AUD-F1-20260806-146: the AUTHORITATIVE current state lives in
    # SQLite and is written in the SAME transaction as the interval
    # row. JSON is a derived, read-only export — never the authority.
    con.execute("""
        CREATE TABLE IF NOT EXISTS rt_state_v2 (
            run_id TEXT PRIMARY KEY,
            last_origin_index INTEGER,
            model_after_path TEXT, model_after_sha256 TEXT,
            carried_equity REAL, updated_at TEXT
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


def _rollout(model, env, *, warmup_bars: int,
             cadence_bars: int) -> dict:
    """Roll out warm-up as FORCED HOLDS, then h decision bars (145).

    A forced hold is action 0.0, which the environment maps below the
    action threshold to "hold" — no order is submitted, no position is
    created, no fee is charged and no activity is counted. The policy
    only acts on the scored interval.
    """
    import numpy as np

    obs, _ = env.reset()
    samples: list[dict] = []
    hold = np.zeros(env.action_space.shape, dtype=np.float32)
    total = warmup_bars + cadence_bars
    for step in range(total):
        in_warmup = step < warmup_bars
        if in_warmup:
            action = hold
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        if isinstance(info, dict) and info.get("equity") is not None:
            samples.append({
                "equity": float(
                    info.get("economic_equity", info["equity"])),
                "position": info.get("position", 0.0),
                "price": info.get("price", 0.0),
                "trades": info.get("trades", 0),
                "commission_paid": info.get("commission_paid", 0.0),
                "warmup": in_warmup,
            })
        if terminated or truncated:
            break
    warmup_facts = [s for s in samples if s["warmup"]]
    return {
        "samples": samples,
        "warmup_traded": any(
            float(s.get("trades", 0) or 0) > 0 for s in warmup_facts),
        "warmup_commission": max(
            [float(s.get("commission_paid", 0) or 0)
             for s in warmup_facts] or [0.0]),
    }


def source_tree_digest(repos=("agent-multi", "gym-fx")) -> dict:
    """AUD-F1-20260806-149: Git HEAD alone hides uncommitted changes.

    Records HEAD plus a digest of the tracked working tree; a dirty
    tree is reported explicitly and makes the run diagnostic-only.
    """
    facts = {}
    for repo in repos:
        root = f"/home/harveybc/Documents/GitHub/{repo}"
        head = _git_rev(repo)
        diff = subprocess.run(
            ["git", "-C", root, "diff", "HEAD"],
            capture_output=True, text=True).stdout
        # AUD-F1-20260806-155: untracked files are STILL source. An
        # untracked .py or .json changes what Python executes, so it
        # binds into identity and makes the tree unclean.
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain",
             "--untracked-files=all"],
            capture_output=True, text=True).stdout.strip()
        untracked = [line[3:] for line in status.splitlines()
                     if line.startswith("??")]
        relevant = [name for name in untracked
                    if name.endswith((".py", ".json", ".yaml", ".yml",
                                      ".toml", ".cfg", ".ini"))]
        untracked_digest = None
        if relevant:
            digest = hashlib.sha256()
            for name in sorted(relevant):
                digest.update(name.encode())
                try:
                    digest.update(
                        (Path(root) / name).read_bytes())
                except OSError:
                    digest.update(b"<unreadable>")
            untracked_digest = digest.hexdigest()
        facts[repo] = {
            "head": head,
            "clean": status == "",
            "dirty_diff_sha256": (
                hashlib.sha256(diff.encode()).hexdigest()
                if status else None),
            "untracked_relevant": sorted(relevant),
            "untracked_content_sha256": untracked_digest,
        }
    facts["all_clean"] = all(
        v["clean"] for v in facts.values() if isinstance(v, dict))
    return facts


ANCHOR_MANIFEST_SCHEMA = "agent_multi.champion_anchor_manifest.v1"
ANCHOR_REQUIRED = (
    "schema", "artifact_sha256", "resolved_genome_sha256",
    "observation_manifest_sha256", "preprocessing_sha256",
    "data_sha256", "source_revisions", "selection_evidence",
    "promotion_eligible")


def load_anchor_manifest(anchor_path: str) -> dict:
    """A bare compatible SAC ZIP is NOT an anchor (158).

    A performance run's starting model must carry a versioned champion
    manifest beside it (`<artifact>.anchor.json`) binding the artifact
    hash, resolved genome, observation/preprocessing/data contracts,
    source revisions, the selection evidence that made it a champion
    and an explicit promotion-eligibility decision. Anything missing
    refuses the run.
    """
    artifact = Path(anchor_path)
    if not artifact.is_file():
        raise SystemExit(f"anchor artifact not found: {artifact}")
    manifest_path = artifact.with_suffix(artifact.suffix
                                         + ".anchor.json")
    if not manifest_path.is_file():
        raise SystemExit(
            f"anchor {artifact.name} has NO champion manifest at"
            f" {manifest_path.name}; a compatible SAC file is not a"
            " mature anchor (finding 158)")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"anchor manifest unreadable: {exc}")
    missing = [key for key in ANCHOR_REQUIRED if not manifest.get(key)]
    if missing:
        raise SystemExit(
            f"anchor manifest is incomplete: missing {missing}")
    if manifest["schema"] != ANCHOR_MANIFEST_SCHEMA:
        raise SystemExit(
            f"anchor manifest schema {manifest['schema']!r} !="
            f" {ANCHOR_MANIFEST_SCHEMA!r}")
    actual = _sha_file(artifact)
    if manifest["artifact_sha256"] != actual:
        raise SystemExit(
            f"anchor manifest hash {manifest['artifact_sha256'][:12]}"
            f" != artifact {actual[:12]}")
    if manifest["promotion_eligible"] is not True:
        raise SystemExit(
            "anchor manifest declares promotion_eligible=false; it is"
            " not a mature champion")
    if manifest["data_sha256"] != DATA_SHA256:
        raise SystemExit(
            "anchor was trained on a different dataset than this RT"
            " run's frozen contract")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha_file(manifest_path)
    return manifest


def execute_handover(env, samples) -> dict:
    """Close open exposure through the simulator and PROVE flatness.

    AUD-F1-20260806-152. The previous version multiplied the direction
    flag {-1,0,1} by price and commission — a 100x error at
    position_size=0.01 — and then asserted `flat_after_handover=true`.
    Now the close runs through the environment's own action-3 path
    (cancel resting orders, close position, real configured costs), and
    flatness must be PROVEN by post-close simulator facts. Unavailable
    evidence refuses the origin instead of assuming success.
    """
    last = samples[-1] if samples else {}
    before = {
        "position_units": last.get("position_units"),
        "open_order_count": last.get("open_order_count"),
        "equity": last.get("equity"),
        "trades": last.get("trades"),
        "commission_paid": last.get("commission_paid"),
    }
    flatten = getattr(env, "flatten_step", None)
    if not callable(flatten):
        return {"flat_proven": False,
                "reason": ("environment exposes no flatten_step; the"
                           " simulator close path is unavailable")}
    try:
        info = flatten()
    except Exception as exc:                       # noqa: BLE001
        return {"flat_proven": False,
                "reason": f"close failed: {type(exc).__name__}: {exc}"}
    units_after = info.get("position_units")
    orders_after = info.get("open_order_count")
    equity_after = info.get("economic_equity", info.get("equity"))
    if units_after is None or orders_after is None or \
            equity_after is None:
        return {"flat_proven": False,
                "reason": ("post-close facts unavailable"
                           f" (units={units_after},"
                           f" orders={orders_after},"
                           f" equity={equity_after})")}
    if abs(float(units_after)) > 1e-12 or int(orders_after) != 0:
        return {"flat_proven": False,
                "reason": (f"account NOT flat after close:"
                           f" units={units_after},"
                           f" open_orders={orders_after}")}
    commission_before = float(before.get("commission_paid") or 0.0)
    commission_after = float(info.get("commission_paid") or 0.0)
    return {
        "flat_proven": True,
        "mode": "simulator_executed_close_action_3",
        "position_units_before": before.get("position_units"),
        "position_units_after": float(units_after),
        "open_orders_after": int(orders_after),
        "equity_before_close": before.get("equity"),
        "post_close_equity": float(equity_after),
        "closing_cost": commission_after - commission_before,
        "trades_after_close": info.get("trades"),
    }


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
        "source_tree": source_tree_digest(),
        "anchor_sha256": (
            _sha_file(Path(args.anchor_model))
            if getattr(args, "anchor_model", None) else None),
        "anchor_path": getattr(args, "anchor_model", None),
        "anchor_manifest_sha256": (
            getattr(args, "_anchor_manifest", {}) or {}
        ).get("manifest_sha256"),
        "anchor_genome_sha256": (
            getattr(args, "_anchor_manifest", {}) or {}
        ).get("resolved_genome_sha256"),
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

    # AUD-F1-20260806-158: validate the anchor's championship BEFORE
    # anything else, so an incompatible or unproven anchor cannot even
    # produce a run identity.
    args._anchor_manifest = (
        load_anchor_manifest(args.anchor_model)
        if getattr(args, "anchor_model", None) else None)
    identity = run_identity(args, config)
    if not identity["source_tree"]["all_clean"] and not \
            args.allow_dirty_tree:
        raise SystemExit(
            "decision-bearing RT runs require CLEAN tracked worktrees"
            " (finding 149); pass --allow-dirty-tree for a diagnostic"
            " run, which is ineligible for promotion")
    identity["promotion_eligible"] = bool(
        identity["source_tree"]["all_clean"] and args.anchor_model)
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
    pointer_path = out_dir / "current_state.json"   # derived export
    state_row = con.execute(
        "SELECT last_origin_index, model_after_path,"
        " model_after_sha256, carried_equity FROM rt_state_v2"
        " WHERE run_id=?", (run_id,)).fetchone()
    pointer = {
        "last_origin_index": state_row[0] if state_row else None,
        "after_path": state_row[1] if state_row else None,
        "after_sha256": state_row[2] if state_row else None,
        "carried_equity": state_row[3] if state_row else None,
    }
    if pointer["after_path"]:
        # Verify every artifact byte before continuing (WP4).
        restored = Path(pointer["after_path"])
        if not restored.is_file():
            raise SystemExit(
                f"recorded state points at a missing artifact:"
                f" {restored}")
        if _sha_file(restored) != pointer["after_sha256"]:
            raise SystemExit(
                "recorded state artifact hash mismatch; refusing to"
                " continue on ambiguous state")

    def lookback_start(origin: int) -> int:
        if args.lookback == "expanding":
            return 0
        years = int(args.lookback.rstrip("y"))
        return max(0, origin - years * 365 * BARS_PER_DAY)

    origins, remainder_bars = block_origins(
        block_start, block_bars, args.cadence_bars)
    if remainder_bars and not args.allow_partial_remainder:
        raise SystemExit(
            f"block of {block_bars} bars is not divisible by cadence"
            f" {args.cadence_bars} ({remainder_bars} bars remain);"
            " pass --allow-partial-remainder to DROP the remainder"
            " explicitly (finding 157)")
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
            # Exactly-once replay (finding 146): row and state are one
            # transaction, so a committed origin is never re-applied.
            # The state row describes the LAST committed origin, so the
            # hash cross-check applies to that origin only; earlier
            # origins are simply skipped.
            if index == pointer.get("last_origin_index") and \
                    pointer.get("after_sha256") != committed[0]:
                raise SystemExit(
                    f"origin {index}: OLAP after-hash {committed[0][:12]}"
                    f" != state {str(pointer.get('after_sha256'))[:12]}"
                    " — refusing to continue on ambiguous state")
            carried_equity = pointer.get("carried_equity")
            print(f"[rt] origin {index} already committed — replay skip"
                  f" (carried equity {carried_equity})", flush=True)
            model_age_bars += args.cadence_bars
            continue

        # ---------- 1. incumbent state (before) ----------
        before_path = None
        expected_before_sha = pointer.get("after_sha256")
        if pointer.get("after_path") and Path(
                pointer["after_path"]).is_file():
            before_path = Path(pointer["after_path"])
            if _sha_file(before_path) != expected_before_sha:
                raise SystemExit(
                    f"origin {index}: inherited model hash does not"
                    " match the recorded after-state; refusing to"
                    " continue on a broken succession chain")

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
        elif args.anchor_model:
            # AUD-F1-20260806-147: adaptation is measured FROM a mature
            # champion/anchor, never from a fresh random SAC.
            anchor = Path(args.anchor_model)
            model = SAC.load(str(anchor), env=fit_env,
                             device=args.device)   # load proof
            before_immutable = checkpoints / f"origin{index}_before.zip"
            model.save(str(before_immutable))
            before_path = before_immutable
        elif args.allow_fresh_init:
            model = SAC("MlpPolicy", fit_env, seed=args.seed,
                        device=args.device,
                        policy_kwargs={"net_arch": [256, 256]})
            model.learn(total_timesteps=args.initial_steps,
                        progress_bar=False)
            before_immutable = checkpoints / f"origin{index}_before.zip"
            model.save(str(before_immutable))
            before_path = before_immutable
        else:
            raise SystemExit(
                "no --anchor-model given: a performance RT run must"
                " adapt a mature champion, not a fresh random SAC"
                " (finding 147). Pass --allow-fresh-init ONLY for a"
                " mechanics fixture, which cannot select a cadence.")

        # ---------- 2. adaptation (or frozen control) ----------
        adapted = False
        if args.control_mode == "adaptive" and (
                index > 0 or args.update_first_origin):
            model.learn(total_timesteps=args.update_steps,
                        progress_bar=False)
            adapted = True
        after_path = checkpoints / f"origin{index}_after.zip"
        if os.environ.get("RT_CRASH_BEFORE_ARTIFACT") == str(index):
            raise SystemExit("injected crash BEFORE artifact write")
        model.save(str(after_path))
        if os.environ.get("RT_CRASH_AFTER_ARTIFACT") == str(index):
            raise SystemExit("injected crash AFTER artifact write")
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
        handover_requested_at = datetime.now(timezone.utc).isoformat()
        eval_env = _build_env(
            {**config, "eval_seed": args.seed}, eval_csv,
            starting_cash=carried_equity)
        rollout = _rollout(model, eval_env, warmup_bars=WARMUP_BARS,
                           cadence_bars=args.cadence_bars)
        if rollout["warmup_traded"]:
            eval_env.close()
            raise SystemExit(
                f"origin {index}: warm-up placed trades — the forced"
                " hold contract is broken (finding 145)")
        # AUD-F1-20260806-152: EXECUTE the close through the simulator's
        # own risk-reducing path and PROVE the account is flat from its
        # facts; never compute a cost from a direction flag.
        handover = execute_handover(eval_env, rollout["samples"])
        eval_env.close()
        score = score_interval(
            rollout["samples"], warmup_bars=WARMUP_BARS,
            cadence_bars=args.cadence_bars,
            starting_equity=carried_equity,
            handover=handover)
        if "unavailable" in score:
            raise SystemExit(
                f"origin {index}: {score['unavailable']}")
        handover_flat_proven_at = datetime.now(timezone.utc).isoformat()

        # ---------- 4. atomic commit: OLAP row + state pointer -------
        artifact_ready_at = datetime.now(timezone.utc).isoformat()
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
            "interval_trades": score["interval_trades"],
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
            "expected_before_sha256": expected_before_sha,
            "succession_ok": int(
                expected_before_sha is None
                or _sha_file(before_path) == expected_before_sha),
            "model_after_sha256": after_sha,
            "replica_verified": replica_verified,
            "interval_commission": score["interval_commission"],
            "handover_json": json.dumps(score["handover"]),
            "handover_requested_at": handover_requested_at,
            "handover_flat_proven_at": handover_flat_proven_at,
            "artifact_ready_at": artifact_ready_at,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            # AUD-F1-20260806-148: an unreconciled handover is one where
            # the interval ended with exposure that was not closed and
            # charged. The explicit flat close makes this measurable.
            "unreconciled_handovers": int(
                score["handover"].get("flat_proven") is not True),
            "activation_delay_bars": (
                latency_seconds / BAR_SECONDS),
            "rollback_status": "none",
            "warmup_traded": int(rollout["warmup_traded"]),
            "anchor_sha256": identity.get("anchor_sha256"),
            "source_tree_digest": _sha_json(identity["source_tree"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # AUD-F1-20260806-146: ONE transaction contains both the
        # interval row and the authoritative state. A crash before the
        # commit leaves neither; after it leaves both. JSON is exported
        # afterwards as a derived, read-only view.
        with con:
            con.execute(
                f"INSERT INTO rt_intervals_v2 ({','.join(record)})"
                f" VALUES ({','.join('?' * len(record))})",
                list(record.values()))
            con.execute(
                "INSERT INTO rt_state_v2 (run_id, last_origin_index,"
                " model_after_path, model_after_sha256,"
                " carried_equity, updated_at)"
                " VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(run_id) DO UPDATE SET"
                " last_origin_index=excluded.last_origin_index,"
                " model_after_path=excluded.model_after_path,"
                " model_after_sha256=excluded.model_after_sha256,"
                " carried_equity=excluded.carried_equity,"
                " updated_at=excluded.updated_at",
                (run_id, index, str(after_path), after_sha,
                 score["equity_after"],
                 datetime.now(timezone.utc).isoformat()))
        if os.environ.get("RT_CRASH_AFTER_COMMIT") == str(index):
            raise SystemExit("injected crash AFTER SQL commit")
        # AUD-F1-20260806-153: the in-memory authority MUST advance
        # after each commit, or the next origin silently restarts from
        # the anchor instead of inheriting the adapted model.
        carried_equity = score["equity_after"]
        pointer = {
            "last_origin_index": index,
            "after_path": str(after_path),
            "after_sha256": after_sha,
            "carried_equity": carried_equity,
        }
        _atomic_write(pointer_path, {          # derived export only
            "authority": "rt_state_v2 table in the OLAP database",
            "last_origin_index": index,
            "model_after_path": str(after_path),
            "model_after_sha256": after_sha,
            "carried_equity": carried_equity,
        })
        print(json.dumps({k: record[k] for k in (
            "origin_index", "scored_bars", "interval_return",
            "equity_before", "equity_after",
            "update_latency_seconds", "deadline_miss")}), flush=True)

    # AUD-F1-20260806-154: percentiles come from EVERY committed row of
    # this run, so a restart cannot discard earlier observations and
    # falsely satisfy the deadline budget.
    ordered = [row[0] for row in con.execute(
        "SELECT update_latency_seconds FROM rt_intervals_v2"
        " WHERE run_id=? AND update_latency_seconds IS NOT NULL"
        " ORDER BY update_latency_seconds", (run_id,))]
    process_local = sorted(latencies)

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
    unreconciled = int(con.execute(
        "SELECT COALESCE(SUM(unreconciled_handovers),0) FROM"
        " rt_intervals_v2 WHERE run_id=?", (run_id,)).fetchone()[0])
    handover_rows = int(con.execute(
        "SELECT COUNT(*) FROM rt_intervals_v2 WHERE run_id=? AND"
        " handover_flat_proven_at IS NOT NULL", (run_id,)
    ).fetchone()[0])
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
        "latency_sample_size": len(ordered),
        "latency_source": "all committed OLAP rows for this run_id",
        "process_local_samples": len(process_local),
        "deadline_seconds": cadence_seconds,
        "deadline_misses": misses,
        "deadline_guard": {
            "rule": ("p95 end-to-end latency <= (2/3) * cadence AND"
                     " >= 20 updates AND zero deadline misses AND zero"
                     " unreconciled handovers, each MEASURED"),
            "p95_within_budget": (
                p95 is not None and p95 <= cadence_seconds * 2 / 3),
            "updates_observed": updates,
            "sufficient_updates": updates >= 20,
            "deadline_misses": misses,
            "zero_deadline_misses": misses == 0,
            # AUD-F1-20260806-148: reconciliation is now MEASURED, not
            # asserted. Every origin must carry a proven flat handover.
            "unreconciled_handovers": unreconciled,
            "handover_evidence_rows": handover_rows,
            "reconciliation_evidence_complete": (
                handover_rows == updates and updates > 0),
            "zero_unreconciled_handovers": unreconciled == 0,
            "satisfied": (
                p95 is not None and p95 <= cadence_seconds * 2 / 3
                and updates >= 20 and misses == 0
                and unreconciled == 0 and handover_rows == updates
                and updates > 0),
            "status": "owner_amended_2026_08_06",
        },
        "promotion_eligible": identity.get("promotion_eligible", False),
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
    parser.add_argument(
        "--anchor-model",
        help="load-proven mature champion to adapt (finding 147)")
    parser.add_argument(
        "--allow-fresh-init", action="store_true",
        help="mechanics fixture only; cannot select a cadence")
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help="diagnostic run on a dirty worktree; ineligible for"
             " promotion (finding 149)")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
