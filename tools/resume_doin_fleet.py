#!/usr/bin/env python3
"""Authenticated same-chain fleet resume with PROVEN rejoin (122).

Takes the fleet pause report produced by ``pause_doin_fleet.py`` (which
carries each node's pause binding hash), posts ``/api/resume`` to every
node through SSH-to-loopback (mutation endpoints are loopback-only),
then POLLS each supervisor until it PROVES the rejoin from worker-
reported lineage — bound domain, genesis block and generation-zero
population fingerprint — or refutes it.

Acceptance is not resumption: exit 0 requires ``rejoin_proven`` on every
node. A contradiction returns that node to paused and exits 1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

SSH_HOST = {"omega": None, "dragon": "dragon", "gamma": "gamma"}


def _run(node_id: str, command: str, timeout: float) -> str:
    host = SSH_HOST.get(node_id, node_id)
    argv = ["bash", "-lc", command] if host is None else [
        "ssh", "-o", "BatchMode=yes", host, command]
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout)
    return done.stdout.strip()


def _post_resume(node_id: str, port: int, binding_hash: str,
                 timeout: float) -> dict:
    payload = json.dumps({"binding_hash": binding_hash})
    command = (
        f"curl -s --max-time {int(timeout)} -X POST"
        f" -H 'Content-Type: application/json'"
        f" -d '{payload}' http://127.0.0.1:{port}/api/resume")
    out = _run(node_id, command, timeout + 30)
    if not out:
        raise RuntimeError("empty response from supervisor")
    return json.loads(out)


def _status(node_id: str, port: int, timeout: float) -> dict:
    out = _run(node_id,
               f"curl -s --max-time {int(timeout)}"
               f" http://127.0.0.1:{port}/api/status", timeout + 30)
    return json.loads(out) if out else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--pause-report", type=Path, required=True,
                        help="JSON output of pause_doin_fleet.py")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--proof-timeout", type=float, default=900.0,
                        help="seconds to wait for post-rejoin proof")
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
                "fleet_resumed": False,
                "reason": f"node {node_id} has no VERIFIED pause"
                          " binding; refusing fleet resume"}, indent=1))
            return 1
        bindings[node_id] = binding_hash

    accepted = {}
    for participant in plan["participants"]:
        node_id = participant["node_id"]
        try:
            accepted[node_id] = _post_resume(
                node_id, args.port, bindings[node_id], args.timeout)
        except Exception as exc:
            accepted[node_id] = {"resume_accepted": False,
                                 "error": str(exc)}
    if not all(r.get("resume_accepted") for r in accepted.values()):
        print(json.dumps({"fleet_resumed": False,
                          "stage": "acceptance",
                          "nodes": accepted}, indent=1, default=str))
        return 1

    deadline = time.monotonic() + args.proof_timeout
    proofs: dict = {}
    while time.monotonic() < deadline:
        proofs = {}
        for participant in plan["participants"]:
            node_id = participant["node_id"]
            status = _status(node_id, args.port, args.timeout)
            proofs[node_id] = (status.get("resume_report") or {})
        if any(r.get("rejoin_contradictions") for r in proofs.values()):
            print(json.dumps({"fleet_resumed": False,
                              "stage": "refuted", "nodes": proofs},
                             indent=1, default=str))
            return 1
        if all(r.get("rejoin_proven") for r in proofs.values()):
            print(json.dumps({
                "schema": "agent_multi.fleet_resume_report.v2",
                "plan_id": plan.get("plan_id"),
                "fleet_resumed": True,
                "nodes": proofs,
            }, indent=1, default=str))
            return 0
        time.sleep(15)

    print(json.dumps({
        "fleet_resumed": False,
        "stage": "proof_timeout",
        "reason": ("rejoin proof did not arrive within"
                   f" {args.proof_timeout}s; unavailable evidence is"
                   " not success"),
        "nodes": proofs,
    }, indent=1, default=str))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
