"""Quick smoke test helper: load a base config, override key params for a short
1-2k step run, dispatch agent-multi, and print trades/return.

Usage:
    python tools/smoke_run.py examples/config/p4_ppo_btc_1h.json --steps 3000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--learning_starts", type=int, default=300)
    ap.add_argument("--n_steps", type=int, default=256)
    args = ap.parse_args()

    base = json.loads(Path(args.config).read_text())
    base["total_timesteps"] = args.steps
    if "ppo" in args.config:
        base["n_steps"] = args.n_steps
    base["learning_starts"] = args.learning_starts
    base["save_model"] = "/tmp/_smoke.zip"
    base["results_file"] = "/tmp/_smoke_summary.json"
    base["save_config"] = "/tmp/_smoke_cfg_out.json"

    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    tmp.write_text(json.dumps(base))

    print(f"[smoke] {args.config}  steps={args.steps}", flush=True)
    proc = subprocess.run(
        ["agent-multi", "--load_config", str(tmp), "--quiet_mode"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        print("[smoke] FAILED rc=", proc.returncode)
        print(proc.stderr[-2000:])
        return proc.returncode

    try:
        s = json.loads(Path("/tmp/_smoke_summary.json").read_text())
        print(f"[smoke] trades={s.get('trades_total')} "
              f"ret={s.get('total_return'):.5f} "
              f"dd={s.get('max_drawdown_pct'):.2f} "
              f"sharpe={s.get('sharpe_ratio')}")
    except Exception as exc:
        print("[smoke] could not read summary:", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
