#!/usr/bin/env python3
"""Authenticated same-chain fleet resume (finding 121).

Takes the fleet pause report produced by ``pause_doin_fleet.py`` (which
carries each node's pause binding hash) and POSTs ``/api/resume`` with
the matching hash to every participant. Refuses to proceed if any node
is missing a verified pause binding. Exit 0 only when every supervisor
reports ``resumed=true`` against its exact bound campaign identity.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/resume", method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--pause-report", type=Path, required=True,
                        help="JSON output of pause_doin_fleet.py")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())
    plan_path = Path(profile["plan_file"])
    if not plan_path.is_absolute():
        plan_path = args.profile.parent / plan_path
    plan = json.loads(plan_path.read_text())
    pause = json.loads(args.pause_report.read_text())

    bindings = {}
    for node_id, node_report in (pause.get("nodes") or {}).items():
        binding_hash = node_report.get("binding_hash")
        if node_report.get("paused") is not True or not binding_hash:
            print(json.dumps({
                "resumed": False,
                "reason": f"node {node_id} has no VERIFIED pause"
                          " binding; refusing fleet resume"}))
            return 1
        bindings[node_id] = binding_hash

    all_resumed = True
    summary = {}
    for participant in plan["participants"]:
        node_id = participant["node_id"]
        report = _post(participant["supervisor_url"],
                       {"binding_hash": bindings.get(node_id, "")},
                       args.timeout)
        summary[node_id] = report
        if not report.get("resumed"):
            all_resumed = False

    print(json.dumps({
        "schema": "agent_multi.fleet_resume_report.v1",
        "plan_id": plan.get("plan_id"),
        "fleet_resumed": all_resumed,
        "nodes": summary,
    }, indent=1, default=str))
    return 0 if all_resumed else 1


if __name__ == "__main__":
    raise SystemExit(main())
