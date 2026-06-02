"""P2c.1 — multi-seed sweep driver.

Loads a base config JSON, runs agent-multi once per seed in {0,1,2} by default,
writes each run under logs/partIII/<run_id>/ with config.json, summary.json,
policy.zip, train.log, git_sha.txt.

run_id = <asset>_<algo>_<features>_<strategy>_s<seed>_<utc>

The sweep does NOT re-invoke the CLI; it loads agent_multi.app.main and runs
in-process per seed so logs/summaries land exactly where we write them.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT_DEFAULT = REPO_ROOT / "logs" / "partIII"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _build_run_id(cfg: Dict[str, Any], seed: int, ts: str) -> str:
    asset = cfg.get("asset") or cfg.get("symbol") or _asset_from_path(cfg.get("input_data_file", ""))
    algo = cfg.get("agent_plugin", "agent").replace("_agent", "")
    features = cfg.get("features_preset", "twelve")
    strategy = cfg.get("strategy_plugin", "default_strategy").replace("_strategy", "")
    return f"{asset}_{algo}_{features}_{strategy}_s{seed}_{ts}"


def _asset_from_path(path: str) -> str:
    p = Path(path)
    # e.g. data/project2/btcusdt_1h/d4.csv → btcusdt_1h
    for part in p.parts[::-1]:
        if part and not part.endswith(".csv") and part != "d4" and part != "project2":
            return part
    return "unknown"


def _run_one(base_cfg: Dict[str, Any], seed: int, run_dir: Path, git_sha: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["train_seed"] = seed
    cfg["eval_seed"] = seed
    cfg["save_model"] = str(run_dir / "policy.zip")
    cfg["results_file"] = str(run_dir / "summary.json")
    cfg["save_config"] = str(run_dir / "config_out.json")
    if "return_trace_file" in cfg:
        cfg["return_trace_file"] = str(run_dir / "return_trace.csv")
    if "return_trace_dir" in cfg:
        cfg["return_trace_dir"] = str(run_dir / "return_traces")

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    (run_dir / "git_sha.txt").write_text(git_sha + "\n", encoding="utf-8")

    log_path = run_dir / "train.log"
    print(f"[seed_sweep] running seed={seed} run_dir={run_dir}", flush=True)

    # Run via the installed console-script in a subprocess so stdout/stderr can
    # be captured cleanly and the kernel process state resets per seed.
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            ["agent-multi", "--load_config", str(cfg_path), "--quiet_mode"],
            cwd=REPO_ROOT,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            summary = {"_error": f"summary parse failed: {exc}"}
    return {
        "seed": seed,
        "exit_code": proc.returncode,
        "summary": summary,
        "run_dir": str(run_dir),
    }


def _locked_protocol_reason(cfg: Dict[str, Any]) -> str | None:
    lock = cfg.get("_protocol_lock")
    if not isinstance(lock, dict):
        return None
    if not lock.get("_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"):
        return None
    if os.environ.get("PROJECT3_ALLOW_STAGE_B_LOCKED_PROTOCOL") == "1":
        return None
    reason = lock.get("reason") or "protocol lock requires Stage B approval"
    packet = lock.get("protocol_packet")
    return f"{reason}" + (f" packet={packet}" if packet else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="base config JSON path")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--log_root", default=str(LOG_ROOT_DEFAULT))
    parser.add_argument("--run_tag", default=None, help="optional suffix tag")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open("r", encoding="utf-8") as fh:
        base_cfg = json.load(fh)
    locked_reason = _locked_protocol_reason(base_cfg)
    if locked_reason:
        print(
            json.dumps(
                {
                    "ok": False,
                    "blocked": "stage_b_protocol_lock",
                    "config": str(cfg_path),
                    "reason": locked_reason,
                    "override_env": "PROJECT3_ALLOW_STAGE_B_LOCKED_PROTOCOL=1",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_sha = _git_sha()
    log_root = Path(args.log_root).resolve()

    results: List[Dict[str, Any]] = []
    for seed in args.seeds:
        run_id = _build_run_id(base_cfg, seed, ts)
        if args.run_tag:
            run_id = f"{run_id}_{args.run_tag}"
        run_dir = log_root / run_id
        results.append(_run_one(base_cfg, seed, run_dir, git_sha))

    ok = all(r["exit_code"] == 0 for r in results)
    print(json.dumps({"ok": ok, "results": results}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
