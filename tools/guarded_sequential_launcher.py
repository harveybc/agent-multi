"""Finding 315: launch-identity guard for sequential arm wrappers.

A sequential wrapper must REFUSE to launch its next arm when the
worktree HEAD, tree cleanliness, or any governed executable-file hash
differs from the frozen launch-identity manifest. Provenance may be
reported (a docs-only commit), but executable identity is never
silently substituted.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


class LaunchIdentityDrift(RuntimeError):
    pass


def verify_launch_identity(manifest_path: Path, worktree: Path,
                           expected_manifest_sha256: str | None = None
                           ) -> dict:
    raw = Path(manifest_path).read_bytes()
    if expected_manifest_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_manifest_sha256:
            raise LaunchIdentityDrift(
                "manifest drift: the launch-identity manifest no "
                f"longer matches its pinned literal digest "
                f"({actual[:16]} != {expected_manifest_sha256[:16]})")
    manifest = json.loads(raw)
    head = subprocess.run(["git", "-C", str(worktree), "rev-parse",
                           "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if head != manifest["full_commit"]:
        raise LaunchIdentityDrift(
            f"HEAD {head[:12]} != frozen {manifest['full_commit'][:12]}")
    dirty = subprocess.run(["git", "-C", str(worktree), "status",
                            "--porcelain"], capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise LaunchIdentityDrift("worktree is dirty")
    for rel, expected in manifest["file_sha256"].items():
        actual = hashlib.sha256(
            (Path(worktree) / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise LaunchIdentityDrift(
                f"executable-file hash drift: {rel}")
    return {"identity_ok": True, "full_commit": head,
            "files_verified": len(manifest["file_sha256"])}
