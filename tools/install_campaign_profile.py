#!/usr/bin/env python3
"""Atomic, guarded campaign-profile installation (finding 119).

Installing a different systemd profile while a campaign is active must
fail. This tool verifies the local supervisor is either inactive or in
a VERIFIED paused state, then installs the drop-in atomically
(temp file + rename) and prints the installed profile's sha256 so the
fleet transaction can be checked host by host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

DROPIN = Path(
    "~/.config/systemd/user/doin-campaign-supervisor.service.d/"
    "50-phase2-eth.conf").expanduser()
PYTHON = "/home/harveybc/anaconda3/envs/trading-stack/bin/python3.12"


def _supervisor_state(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/status", timeout=4) as r:
            return json.loads(r.read())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument(
        "--force", action="store_true",
        help="bypass the active-campaign guard (operator emergencies"
             " only; the bypass itself is printed)")
    args = parser.parse_args()

    profile = args.profile.resolve()
    if not profile.exists():
        print(f"profile does not exist: {profile}", file=sys.stderr)
        return 1

    active = subprocess.run(
        ["systemctl", "--user", "is-active",
         "doin-campaign-supervisor.service"],
        capture_output=True, text=True).stdout.strip()
    refusal: str | None = None
    if active == "active":
        status = _supervisor_state(args.port)
        if status is None:
            # AUD-F1-20260806-123: unavailable status is NOT permission.
            refusal = (
                "supervisor unit is active but its status is"
                " UNREACHABLE; unavailable evidence is never treated as"
                " a verified pause")
        elif status.get("phase") not in ("paused", "complete"):
            refusal = (
                f"supervisor is ACTIVE on plan {status.get('plan_id')!r}"
                f" phase {status.get('phase')!r}; profile installation"
                " requires a verified pause")
        elif status.get("phase") == "paused" and (
                status.get("pause_report") or {}).get("paused") is not True:
            refusal = (
                "supervisor reports phase 'paused' without a VERIFIED"
                " pause report; refusing profile installation")
    elif active not in ("inactive", "failed", "unknown", ""):
        refusal = f"unrecognized unit state {active!r}; refusing"
    if refusal is not None:
        if not args.force:
            print(json.dumps({"installed": False, "reason": refusal},
                             indent=1))
            return 1
        print(f"WARNING: --force bypassed the guard ({refusal})",
              file=sys.stderr)

    content = (
        "[Service]\nExecStart=\n"
        f"ExecStart={PYTHON} -m app.campaign_supervisor --profile"
        f" {profile}\n")
    DROPIN.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(DROPIN.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        handle.write(content)
    os.replace(tmp, DROPIN)             # atomic on the same filesystem
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    print(json.dumps({
        "installed": True,
        "dropin": str(DROPIN),
        "profile": str(profile),
        "profile_sha256": hashlib.sha256(
            profile.read_bytes()).hexdigest(),
        "dropin_sha256": hashlib.sha256(
            DROPIN.read_bytes()).hexdigest(),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
