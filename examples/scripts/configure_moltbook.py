#!/usr/bin/env python3
"""Store a Moltbook API key without exposing it in shell history."""

from __future__ import annotations

import getpass
import os
from pathlib import Path


def main() -> int:
    target = Path.home() / ".config/agent-multi/moltbook.env"
    target.parent.mkdir(parents=True, exist_ok=True)
    key = getpass.getpass("Moltbook API key (hidden): ").strip()
    if not key.startswith("moltbook_"):
        raise SystemExit("The key does not have the expected moltbook_ prefix")
    target.write_text(f"MOLTBOOK_API_KEY={key}\n", encoding="utf-8")
    os.chmod(target, 0o600)
    print(f"Credential stored at {target} with mode 0600")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
