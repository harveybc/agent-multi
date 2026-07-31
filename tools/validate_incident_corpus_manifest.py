#!/usr/bin/env python3
"""Validate the preregistered incident-corpus enumeration rule."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/publications/incident-corpus/manifest.json"


def _git_file(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _extract_lines(text: str, line_start: int, line_end: int) -> str:
    lines = text.splitlines(keepends=True)
    if line_start < 1 or line_end < line_start or line_end > len(lines):
        raise ValueError("invalid preregistration line range")
    return "".join(lines[line_start - 1 : line_end])


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "agent_multi.incident_corpus_manifest.v1":
        raise ValueError("unsupported incident corpus manifest schema")
    preregistration = payload["preregistration"]
    source = _git_file(
        str(preregistration["introducing_commit"]),
        str(preregistration["source_path"]),
    )
    extracted = _extract_lines(
        source,
        int(preregistration["line_start"]),
        int(preregistration["line_end"]),
    )
    observed_hash = hashlib.sha256(extracted.encode("utf-8")).hexdigest()
    if observed_hash != preregistration["sha256"]:
        raise ValueError(
            "preregistered enumeration rule hash mismatch: "
            f"{observed_hash} != {preregistration['sha256']}"
        )
    if extracted != preregistration["rule_text"]:
        raise ValueError("manifest rule_text differs from introducing commit")
    if len(payload["incident_row_fields"]) != len(set(payload["incident_row_fields"])):
        raise ValueError("incident_row_fields contains duplicates")
    return {
        "manifest": str(path),
        "introducing_commit": preregistration["introducing_commit"],
        "rule_sha256": observed_hash,
        "status": "valid",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(validate_manifest(args.manifest.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
