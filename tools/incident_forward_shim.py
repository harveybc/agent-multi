#!/usr/bin/env python3
"""Forced-command shim for fleet incident forwarding (findings 096).

Installed on the notification owner as the ``command=`` target of the
dedicated per-host forwarding keys. The forced command carries IMMUTABLE
identity bindings as its own argv — the connecting key can emit only for
its bound machine, its allow-listed producer sources and fronts, and only
through ``incident_ledger.py observe|recover`` with allow-listed options.
Forwarded identity fields are verified against the bindings, never
trusted. Anything else refuses.

authorized_keys entry shape (one line, owner host):

    command="/usr/bin/python3 REPO/tools/incident_forward_shim.py \
--allowed-machine dragon --allowed-sources swarm_watchdog,... \
--allowed-fronts front1,front2,front4",no-port-forwarding,\
no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 AAAA… label
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "incident_ledger.py"
ALLOWED_SUBCOMMANDS = {"observe", "recover", "receipt"}
ALLOWED_OPTIONS = {
    "--source", "--front", "--machine", "--event-code", "--object",
    "--severity", "--evidence-at", "--observed-at", "--payload-json",
    "--payload-stdin", "--evidence-json", "--state-json", "--config",
    "--fingerprint",
}


def _option_value(tokens: list[str], option: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-machine", required=True)
    parser.add_argument("--allowed-sources", required=True,
                        help="comma-separated producer allowlist")
    parser.add_argument("--allowed-fronts", required=True,
                        help="comma-separated front allowlist")
    bindings = parser.parse_args()
    allowed_sources = {token for token in
                       bindings.allowed_sources.split(",") if token}
    allowed_fronts = {token for token in
                      bindings.allowed_fronts.split(",") if token}

    original = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    if not original:
        print("REFUSED: no forwarded command", file=sys.stderr)
        return 2
    try:
        tokens = shlex.split(original)
    except ValueError:
        print("REFUSED: unparseable forwarded command", file=sys.stderr)
        return 2
    if len(tokens) < 3 or tokens[0] not in ("python3", "/usr/bin/python3"):
        print("REFUSED: only the incident ledger CLI is forwardable",
              file=sys.stderr)
        return 2
    if Path(tokens[1]).name != "incident_ledger.py":
        print("REFUSED: only the incident ledger CLI is forwardable",
              file=sys.stderr)
        return 2
    if tokens[2] not in ALLOWED_SUBCOMMANDS:
        print("REFUSED: only observe/recover are forwardable",
              file=sys.stderr)
        return 2
    for token in tokens[3:]:
        if token.startswith("--") and token not in ALLOWED_OPTIONS:
            print(f"REFUSED: option {token!r} is not forwardable",
                  file=sys.stderr)
            return 2

    # Finding 096: forwarded identity must match the key's immutable
    # bindings exactly; a worker key can never impersonate another host,
    # producer or front. Receipt queries (finding 097) are read-only and
    # carry only the machine binding.
    machine = _option_value(tokens, "--machine")
    if machine != bindings.allowed_machine:
        print(f"REFUSED: machine {machine!r} is not this key's bound"
              f" machine", file=sys.stderr)
        return 2
    if tokens[2] in ("observe", "recover"):
        source = _option_value(tokens, "--source")
        if source not in allowed_sources:
            print(f"REFUSED: source {source!r} is not allow-listed for"
                  f" this key", file=sys.stderr)
            return 2
        front = _option_value(tokens, "--front")
        if front not in allowed_fronts:
            print(f"REFUSED: front {front!r} is not allow-listed for this"
                  f" key", file=sys.stderr)
            return 2

    command = ["/usr/bin/python3", str(LEDGER)] + tokens[2:]
    result = subprocess.run(command, stdin=sys.stdin, timeout=60)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
