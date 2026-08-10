"""Exact system manifests and materialized cell configs (order §3/WP2).

A system manifest freezes every decision-bearing fact of one asset
system — data identity, resolved base configuration, observation
contract, cost/margin/SL-TP contract, plugin surface, anchors and the
source identities present at manifest creation. The L1 runner
materializes each cell's config exclusively THROUGH the manifest
(`materialize_system_config`); it never resolves the legacy base
config directly, and every binding is verified fail-closed before any
model is constructed.

Source-tree identity derives from the ACTUAL checkout that executes
(git toplevel of this file / the named import root), never from a
hard-coded sibling directory, and includes a digest over dirty and
untracked files so an unclean tree can never masquerade as a commit.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

MANIFEST_SCHEMA = "agent_multi.system_manifest.v1"

_VOLATILE_CONFIG_KEYS = (
    # attempt/output-local paths are tokenized so the resolved-config
    # hash is stable across attempts of the same cell
    "save_model", "return_trace_dir", "nested_split_dir",
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------
# source identity — actual executing tree, dirty-aware
# ---------------------------------------------------------------------------

def _git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo_root), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo_root}: "
            f"{proc.stderr.strip()[:200]}")
    return proc.stdout


def resolve_repo_root(anchor_file: Path) -> Path:
    """Toplevel of the checkout that actually contains ``anchor_file``."""
    out = _git(Path(anchor_file).resolve().parent,
               "rev-parse", "--show-toplevel")
    return Path(out.strip())


def source_tree_identity(repo_root: Path) -> Dict[str, Any]:
    """Full commit + digest over every dirty/untracked file's content."""
    repo_root = Path(repo_root).resolve()
    commit = _git(repo_root, "rev-parse", "HEAD").strip()
    porcelain = _git(repo_root, "status", "--porcelain",
                     "--untracked-files=all")
    entries = []
    digest = hashlib.sha256()
    for line in sorted(porcelain.splitlines()):
        if not line.strip():
            continue
        status, _, rel = line[:2], line[2], line[3:]
        target = repo_root / rel
        file_sha = (sha_file(target) if target.is_file() else "absent")
        entries.append({"status": status.strip() or "??",
                        "path": rel, "sha256": file_sha})
        digest.update(f"{status}\0{rel}\0{file_sha}\n".encode())
    return {
        "repo_root": str(repo_root),
        "commit": commit,
        "dirty": bool(entries),
        "dirty_entries": entries,
        "dirty_untracked_digest": (digest.hexdigest() if entries else None),
    }


def assert_source_identity_unmoved(before: Dict[str, Any],
                                   after: Dict[str, Any]) -> None:
    """Fail closed when the executing tree or its digest moved mid-cell."""
    for key in ("repo_root", "commit", "dirty_untracked_digest"):
        if before.get(key) != after.get(key):
            raise RuntimeError(
                "executing source tree moved during the cell: "
                f"{key} {before.get(key)!r} -> {after.get(key)!r}")


# ---------------------------------------------------------------------------
# manifest loading
# ---------------------------------------------------------------------------

def load_system_manifest(path: Path) -> Dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unknown system manifest schema {manifest.get('schema')!r}")
    manifest["_manifest_sha256"] = sha_file(path)
    manifest["_manifest_path"] = str(path)
    return manifest


# ---------------------------------------------------------------------------
# materialization — the ONLY path from manifest to a runnable config
# ---------------------------------------------------------------------------

def resolved_config_sha256(config: Dict[str, Any]) -> str:
    """Hash of the resolved config with attempt-local paths tokenized."""
    canon = dict(config)
    for key in _VOLATILE_CONFIG_KEYS:
        if key in canon:
            canon[key] = f"<{key}>"
    return canonical_sha(canon)


def observation_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    columns = list(config.get("feature_columns") or [])
    window = int(config.get("window_size") or 0)
    count = len(columns)
    obs = {
        "feature_columns": columns,
        "feature_count": count,
        "window_size": window,
        "include_price_window": bool(config.get("include_price_window")),
        "include_agent_state": bool(config.get("include_agent_state")),
        "flattened_shape": None,
    }
    obs["observation_manifest_sha256"] = canonical_sha(
        {k: obs[k] for k in ("feature_columns", "feature_count",
                             "window_size", "include_price_window",
                             "include_agent_state")})
    return obs


def materialize_system_config(contract: Dict[str, Any],
                              system_manifest: Dict[str, Any],
                              cell: str, seed: int,
                              output_dir: Path) -> Dict[str, Any]:
    """Build one cell's exact config from the frozen manifest.

    Every binding is verified before return; the caller receives the
    config plus the identity facts (`_identity`) it must persist.
    """
    repo_root = resolve_repo_root(Path(__file__))
    spec = contract["cells"][cell]

    base_rel = system_manifest["base_config"]["path"]
    base_path = repo_root / base_rel
    actual_base_sha = sha_file(base_path)
    if actual_base_sha != system_manifest["base_config"]["sha256"]:
        raise RuntimeError(
            "base config drifted from the system manifest binding: "
            f"{actual_base_sha[:12]}… != "
            f"{system_manifest['base_config']['sha256'][:12]}…")

    data_path = Path(system_manifest["data"]["path"])
    actual_data_sha = sha_file(data_path)
    if actual_data_sha != system_manifest["data"]["sha256"]:
        raise RuntimeError("input CSV drifted from the system manifest")

    nested_rel = contract["nested_split_contract"]
    nested_path = repo_root / nested_rel
    nested_sha = sha_file(nested_path)
    if nested_sha != system_manifest["nested_split_contract"]["sha256"]:
        raise RuntimeError(
            "nested split contract drifted from the system manifest")
    nested = json.loads(nested_path.read_text())
    if nested.get("source_sha256") != actual_data_sha:
        raise RuntimeError(
            "nested split contract pins a different CSV than the "
            "system manifest data binding")

    config = json.loads(base_path.read_text())
    for field in ("train_years", "val_years", "test_years"):
        config.pop(field, None)
    config["split_contract_note"] = (
        "explicit split dates govern; year-count shorthand removed")
    config.update(system_manifest["splits"]["dates"])
    config["input_data_file"] = str(data_path)
    config["env_mode"] = "training"
    config["eval_seed"] = seed
    config["train_seed"] = seed
    config["ga_seed"] = seed
    config["l1_min_checkpoint_timesteps"] = 1
    config["evaluate_test_split"] = False
    config["quiet_mode"] = True
    config["save_model"] = str(Path(output_dir) / "model.zip")
    config["return_trace_dir"] = str(Path(output_dir) / "return_traces")

    if bool(system_manifest["splits"].get("evaluate_test_split", False)):
        raise RuntimeError(
            "system manifest enables the sealed test split — refused")

    expected_asset = str(system_manifest.get("env_asset") or "")
    if str(config.get("asset")) != expected_asset:
        raise RuntimeError(
            f"materialized asset {config.get('asset')!r} != system "
            f"manifest env_asset {expected_asset!r}")
    if str(contract.get("env_asset")) != expected_asset:
        raise RuntimeError("contract env_asset disagrees with the "
                           "system manifest")

    obs = observation_manifest(config)
    bound_obs = system_manifest["observation"]
    if obs["observation_manifest_sha256"] != \
            bound_obs["observation_manifest_sha256"]:
        raise RuntimeError(
            "observation contract drifted from the system manifest")
    obs["flattened_shape"] = bound_obs.get("flattened_shape")

    costs = system_manifest["costs"]
    for key, bound in costs["config_bindings"].items():
        actual = config.get(key)
        if actual != bound:
            raise RuntimeError(
                f"cost/margin contract drifted: config[{key!r}]="
                f"{actual!r} != manifest {bound!r}")

    anchor_entry = contract["anchors"][str(seed)]
    manifest_anchor = system_manifest["anchors"].get(str(seed))
    if manifest_anchor is None:
        raise RuntimeError(f"seed {seed} absent from manifest anchors")
    if manifest_anchor["sha256"] != anchor_entry["sha256"]:
        raise RuntimeError(
            "contract anchor sha disagrees with the system manifest")

    config["_identity"] = {
        "system_manifest_sha256": system_manifest["_manifest_sha256"],
        "system_manifest_path": system_manifest.get("_manifest_path"),
        "base_config_sha256": actual_base_sha,
        "data_sha256": actual_data_sha,
        "data_rows": system_manifest["data"]["rows"],
        "data_time_bounds": system_manifest["data"]["time_bounds"],
        "nested_split_contract_sha256": nested_sha,
        "observation": obs,
        "env_asset": expected_asset,
        "asset_label": system_manifest.get("asset"),
        "initial_cash": float(config.get("initial_cash")),
        "cost_contract": dict(costs["config_bindings"]),
        "anchor_sha256": anchor_entry["sha256"],
        "anchor_policy_tensor_sha256": manifest_anchor.get(
            "policy_tensor_sha256"),
        "cell": cell,
        "cell_factors": {
            "phase1_mode": spec["phase1_mode"],
            "phase2_lr_multiplier": spec["phase2_lr_multiplier"],
        },
        "seed": seed,
        "metric_schema": contract.get("selection_metric"),
    }
    return config
