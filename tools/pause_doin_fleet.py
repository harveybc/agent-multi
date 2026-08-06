#!/usr/bin/env python3
"""One operator pause command for the whole DOIN fleet (finding 115).

Reads the campaign plan referenced by a supervisor profile, POSTs
``/api/pause`` to every participant supervisor and aggregates their
verification reports. Exit code 0 means every supervisor reported a
verified pause (no worker process, no worker API port, no GPU owner).
Any unreachable supervisor or surviving worker exits 1 with the reason —
an unreachable supervisor with possibly-live workers is a FAILED pause
that requires direct process-group action on that host, never silence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# AUD-F1-20260806-122: supervisor mutation endpoints are loopback-only.
# The fleet command therefore reaches each node through SSH and posts
# to that host's OWN loopback; no network peer can mutate a campaign.
SSH_HOST = {"omega": None, "dragon": "dragon", "gamma": "gamma"}


def _post_pause(node_id: str, port: int, timeout: float) -> dict:
    curl = ["curl", "-s", "--max-time", str(int(timeout)), "-X", "POST",
            f"http://127.0.0.1:{port}/api/pause"]
    host = SSH_HOST.get(node_id, node_id)
    command = curl if host is None else [
        "ssh", "-o", "BatchMode=yes", host, " ".join(curl)]
    done = subprocess.run(command, capture_output=True, text=True,
                          timeout=timeout + 30)
    if not done.stdout.strip():
        raise RuntimeError(
            f"empty response ({done.stderr.strip()[:120]})")
    return json.loads(done.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True,
                        help="any node's supervisor profile (plan source)")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=8795)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())
    plan_path = Path(profile["plan_file"])
    if not plan_path.is_absolute():
        plan_path = args.profile.parent / plan_path
    plan = json.loads(plan_path.read_text())

    all_paused = True
    summary = {}
    for participant in plan["participants"]:
        node_id = participant["node_id"]
        try:
            report = _post_pause(node_id, args.port, args.timeout)
        except Exception as exc:
            summary[node_id] = {
                "paused": False,
                "error": (f"supervisor unreachable ({exc}); workers on"
                          " this host may still be running — FAILED"
                          " pause, act on the host directly"),
            }
            all_paused = False
            continue
        summary[node_id] = report
        if not report.get("paused"):
            all_paused = False

    print(json.dumps({
        "schema": "agent_multi.fleet_pause_report.v1",
        "plan_id": plan.get("plan_id"),
        "fleet_paused": all_paused,
        "nodes": summary,
    }, indent=1, default=str))
    return 0 if all_paused else 1


if __name__ == "__main__":
    raise SystemExit(main())
