#!/usr/bin/env python3
"""Register a Moltbook agent and store its key locally with mode 0600."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


REGISTER_URL = "https://www.moltbook.com/api/v1/agents/register"


def main() -> int:
    name = input("New Moltbook agent name: ").strip()
    description = input("Agent description: ").strip()
    if not name or not description:
        raise SystemExit("Name and description are required")
    request = urllib.request.Request(
        REGISTER_URL,
        data=json.dumps(
            {"name": name, "description": description},
            separators=(",", ":"),
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "agent-multi-moltbook-registration/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read(400).decode("utf-8", errors="replace")
        raise SystemExit(f"Moltbook registration failed: HTTP {exc.code}: {detail}")
    agent = payload.get("agent") or {}
    key = str(agent.get("api_key") or "")
    claim_url = str(agent.get("claim_url") or "")
    verification_code = str(agent.get("verification_code") or "")
    if not key.startswith("moltbook_") or not claim_url.startswith(
        "https://www.moltbook.com/"
    ):
        raise SystemExit("Moltbook response omitted the key or official claim URL")
    target = Path.home() / ".config/agent-multi/moltbook.env"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"MOLTBOOK_API_KEY={key}\n", encoding="utf-8")
    os.chmod(target, 0o600)
    print(f"Credential stored at {target} with mode 0600")
    print(f"Claim URL: {claim_url}")
    print(f"Verification code: {verification_code}")
    print("Open the claim URL and complete the owner verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
