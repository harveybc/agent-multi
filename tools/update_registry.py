"""P2c.2 — append run metadata into logs/partIII/index.csv.

Row columns: run_id, asset, algo, features, strategy, seed, started_at,
finished_at, total_return, sharpe, max_dd, trades, episode_reward,
episode_length, config_hash, git_sha.

Idempotent on run_id: existing rows are not duplicated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH_DEFAULT = REPO_ROOT / "logs" / "partIII" / "index.csv"

COLS = [
    "run_id",
    "asset",
    "algo",
    "features",
    "strategy",
    "seed",
    "started_at",
    "finished_at",
    "total_return",
    "sharpe",
    "max_dd",
    "trades",
    "episode_reward",
    "episode_length",
    "config_hash",
    "git_sha",
    "run_dir",
]


def _hash_config(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _load_index(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_index(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLS})


def _parse_run(run_dir: Path) -> Optional[Dict[str, str]]:
    cfg_path = run_dir / "config.json"
    sum_path = run_dir / "summary.json"
    sha_path = run_dir / "git_sha.txt"
    if not cfg_path.exists() or not sum_path.exists():
        print(f"[warn] skipping {run_dir.name}: missing config or summary", file=sys.stderr)
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    git_sha = sha_path.read_text(encoding="utf-8").strip() if sha_path.exists() else ""

    run_id = run_dir.name
    parts = run_id.split("_")
    # run_id format: <asset>_<algo>_<features>_<strategy>_s<seed>_<utc>[_tag]
    seed = ""
    for tok in parts:
        if tok.startswith("s") and tok[1:].isdigit():
            seed = tok[1:]
            break

    def _get(*keys: str) -> str:
        for k in keys:
            if k in summary:
                return str(summary[k])
        return ""

    return {
        "run_id": run_id,
        "asset": str(cfg.get("asset", "") or cfg.get("symbol", "")),
        "algo": str(cfg.get("agent_plugin", "")),
        "features": str(cfg.get("features_preset", "twelve")),
        "strategy": str(cfg.get("strategy_plugin", "")),
        "seed": seed,
        "started_at": _get("started_at"),
        "finished_at": _get("finished_at"),
        "total_return": _get("total_return"),
        "sharpe": _get("sharpe"),
        "max_dd": _get("max_dd", "max_drawdown"),
        "trades": _get("trades_total", "trades"),
        "episode_reward": _get("episode_reward"),
        "episode_length": _get("episode_length"),
        "config_hash": _hash_config(cfg),
        "git_sha": git_sha,
        "run_dir": str(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", action="append", default=[], help="run directory to ingest (repeat)")
    parser.add_argument("--scan_root", default=None, help="if set, ingest every immediate subdir")
    parser.add_argument("--index", default=str(INDEX_PATH_DEFAULT))
    args = parser.parse_args()

    targets: List[Path] = [Path(p).resolve() for p in args.run_dir]
    if args.scan_root:
        root = Path(args.scan_root).resolve()
        if root.is_dir():
            targets.extend(p for p in root.iterdir() if p.is_dir())

    if not targets:
        parser.error("provide --run_dir or --scan_root")

    index_path = Path(args.index).resolve()
    existing = _load_index(index_path)
    seen = {r["run_id"] for r in existing}

    new_rows: List[Dict[str, str]] = []
    for run_dir in targets:
        if run_dir.name in seen:
            print(f"[skip] already indexed: {run_dir.name}")
            continue
        parsed = _parse_run(run_dir)
        if parsed is None:
            continue
        new_rows.append(parsed)
        seen.add(run_dir.name)

    if not new_rows:
        print("[info] nothing to append")
        return 0

    existing.extend(new_rows)
    _write_index(index_path, existing)
    print(f"[ok] appended {len(new_rows)} row(s) to {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
