from __future__ import annotations

import copy
import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from trading_contracts import TradingRuntimeOverlay

from app.canonical_config import ConfigResolutionError


_PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class RuntimeResolution:
    overlay: TradingRuntimeOverlay | None
    runtime: dict[str, Any]
    manifest: dict[str, Any]


def _absolute_local_path(value: str, *, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve(strict=False))


def _resolve_value(value: Any, roots: Mapping[str, str], *, location: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_value(item, roots, location=f"{location}/{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_value(item, roots, location=f"{location}/{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _resolve_value(item, roots, location=f"{location}/{index}")
            for index, item in enumerate(value)
        )
    if not isinstance(value, str):
        return copy.deepcopy(value)

    names = _PLACEHOLDER.findall(value)
    missing = sorted({name for name in names if name not in roots})
    if missing:
        raise ConfigResolutionError(
            f"unresolved runtime roots at {location}: " + ", ".join(missing)
        )
    resolved = value
    for name in names:
        resolved = resolved.replace(f"${{{name}}}", roots[name])
    return resolved


def _git_text(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.PIPE,
        ).rstrip("\n")
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise ConfigResolutionError(f"git {' '.join(args)} failed for {repo}: {detail.strip()}") from exc


def _git_stream_hash(repo: Path, *args: str) -> str:
    try:
        process = subprocess.Popen(
            ["git", "-C", str(repo), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ConfigResolutionError(f"cannot execute git for {repo}: {exc}") from exc

    digest = hashlib.sha256()
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise ConfigResolutionError(
            f"git {' '.join(args)} failed for {repo}: {stderr.strip()}"
        )
    return "sha256:" + digest.hexdigest()


def collect_git_snapshot(
    name: str,
    path: str,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    repo = Path(path)
    if not repo.is_dir():
        raise ConfigResolutionError(f"repository path for {name!r} does not exist: {repo}")
    root = Path(_git_text(repo, "rev-parse", "--show-toplevel"))
    commit = _git_text(root, "rev-parse", "HEAD")
    branch = _git_text(root, "branch", "--show-current") or "DETACHED"
    status_text = _git_text(root, "status", "--porcelain=v1", "--untracked-files=normal")
    status_lines = status_text.splitlines() if status_text else []
    declared = expected_commit if expected_commit else None
    matches_expected = commit == declared if declared and _COMMIT.fullmatch(declared) else None
    return {
        "name": name,
        "path": str(root.resolve()),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
        "status_sample": status_lines[:20],
        "status_hash": "sha256:" + hashlib.sha256(status_text.encode("utf-8")).hexdigest(),
        "tracked_diff_hash": _git_stream_hash(root, "diff", "--no-ext-diff", "--binary", "HEAD"),
        "expected_commit": declared,
        "matches_expected_commit": matches_expected,
    }


def resolve_runtime_overlay(
    runtime: Mapping[str, Any],
    *,
    overlay_payload: Mapping[str, Any] | None,
    overlay_base_dir: Path,
    expected_repositories: Mapping[str, Any] | None = None,
) -> RuntimeResolution:
    try:
        overlay = (
            TradingRuntimeOverlay.model_validate(overlay_payload)
            if overlay_payload is not None
            else None
        )
    except Exception as exc:
        raise ConfigResolutionError(f"invalid runtime overlay: {exc}") from exc

    roots = {
        name: _absolute_local_path(path, base_dir=overlay_base_dir)
        for name, path in (overlay.roots.items() if overlay else [])
    }
    repositories = {
        name: _absolute_local_path(path, base_dir=overlay_base_dir)
        for name, path in (overlay.repositories.items() if overlay else [])
    }
    resolved_runtime = _resolve_value(
        copy.deepcopy(dict(runtime)),
        roots,
        location="runtime",
    )

    snapshots: dict[str, dict[str, Any]] = {}
    expected = dict(expected_repositories or {})
    for name, path in sorted(repositories.items()):
        snapshots[name] = collect_git_snapshot(
            name,
            path,
            expected_commit=str(expected[name]) if name in expected else None,
        )

    if overlay is not None:
        default_device = overlay.devices.get("training") or overlay.devices.get("default")
        if default_device and resolved_runtime.get("device") in (None, "", "auto"):
            resolved_runtime["device"] = default_device
        resolved_runtime["runtime_machine_id"] = overlay.machine_id
        resolved_runtime["runtime_roots"] = roots
        resolved_runtime["runtime_repositories"] = repositories
        resolved_runtime["runtime_devices"] = dict(overlay.devices)
        resolved_runtime["runtime_resource_limits"] = dict(overlay.resource_limits)
        resolved_runtime["runtime_environment_refs"] = dict(overlay.environment_refs)

    manifest = {
        "schema_version": "runtime_resolution_manifest.v1",
        "runtime_overlay_hash": overlay.canonical_hash if overlay else None,
        "machine_id": overlay.machine_id if overlay else None,
        "resolved_roots": roots,
        "devices": dict(overlay.devices) if overlay else {},
        "resource_limits": dict(overlay.resource_limits) if overlay else {},
        "environment_refs": dict(overlay.environment_refs) if overlay else {},
        "repository_snapshots": snapshots,
    }
    return RuntimeResolution(overlay=overlay, runtime=resolved_runtime, manifest=manifest)
