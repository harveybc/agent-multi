#!/usr/bin/env python3
"""Forced-command shim for fleet incident forwarding.

Installed on the notification owner as the ``command=`` target of the
dedicated per-host forwarding keys, so a compromised worker key can do
exactly one thing: emit incident observations/recoveries into the owner
ledger. Anything else — any other executable, subcommand, option or shell
metacharacter — is refused.

authorized_keys entry shape (one line, owner host):

    command="/usr/bin/python3 REPO/tools/incident_forward_shim.py",\
no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 AAAA… host-incident-forward
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "incident_ledger.py"
ALLOWED_SUBCOMMANDS = {"observe", "recover"}
ALLOWED_OPTIONS = {
    "--source", "--front", "--machine", "--event-code", "--object",
    "--severity", "--evidence-at", "--observed-at", "--payload-json",
    "--payload-stdin", "--evidence-json", "--config",
}


def main() -> int:
    original = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    if not original:
        print("REFUSED: no forwarded command", file=sys.stderr)
        return 2
    try:
        tokens = shlex.split(original)
    except ValueError:
        print("REFUSED: unparseable forwarded command", file=sys.stderr)
        return 2
    # Accept exactly: python3 <ledger path> <subcommand> [allowed options]
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
    command = ["/usr/bin/python3", str(LEDGER)] + tokens[2:]
    result = subprocess.run(command, stdin=sys.stdin, timeout=60)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
