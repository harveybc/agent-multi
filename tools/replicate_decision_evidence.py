#!/usr/bin/env python3
"""Canonical manifest + verified second-host replica (finding 126/114).

Builds a content-addressed manifest of a decision-experiment output
root, copies it to a second host, then VERIFIES the replica by hashing
the copied bytes on that host and comparing them to the manifest. A
manifest pointing at files that were never verified elsewhere does not
satisfy cross-workstation recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "canonical_manifest.json":
            files[str(path.relative_to(root))] = {
                "sha256": _sha(path), "bytes": path.stat().st_size}
    return {
        "schema": "agent_multi.decision_evidence_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replica-host", default="dragon")
    parser.add_argument("--replica-root", required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    manifest = build_manifest(root)
    manifest_path = root / "canonical_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.replica_host,
         f"mkdir -p {args.replica_root}"], check=True)
    subprocess.run(
        ["rsync", "-a", "--delete", f"{root}/",
         f"{args.replica_host}:{args.replica_root}/"], check=True)

    # Verify by hashing on the REMOTE host, never by trusting rsync.
    remote = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.replica_host,
         f"cd {args.replica_root} && find . -type f ! -name"
         " canonical_manifest.json -print0 | sort -z |"
         " xargs -0 sha256sum"],
        capture_output=True, text=True, check=True).stdout

    remote_hashes = {}
    for line in remote.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            remote_hashes[parts[1].strip().lstrip("./")] = parts[0]

    mismatches, missing = [], []
    for name, fact in manifest["files"].items():
        seen = remote_hashes.get(name)
        if seen is None:
            missing.append(name)
        elif seen != fact["sha256"]:
            mismatches.append(name)

    verified = not mismatches and not missing
    report = {
        "schema": "agent_multi.replica_verification.v1",
        "replica_verified": verified,
        "primary_root": str(root),
        "replica": f"{args.replica_host}:{args.replica_root}",
        "file_count": manifest["file_count"],
        "remote_file_count": len(remote_hashes),
        "missing_on_replica": missing[:20],
        "hash_mismatches": mismatches[:20],
        "manifest_sha256": _sha(manifest_path),
    }
    print(json.dumps(report, indent=1))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
