#!/usr/bin/env python3
"""Content-addressed collection + independent replica (order §5/WP4).

Pulls each assigned seed subtree from its declared source host into a
staged collection root, verifies every referenced file against its
recorded hash, rejects duplicate seed/cell identities (even when bytes
differ after a retry), enforces exact experiment/system/code/data
identity uniformity across all records, publishes a source-host
manifest and a collection-tree digest atomically, replicates records +
model artifacts to one independent host, verifies the replica by
rehash + real load, and only then allows aggregation — exclusively
from the SEALED collection root.

Transports are injectable (tests run socket-free); the defaults use
rsync/ssh against the contract's assignments.

Typed outcomes: COLLECTION_SEALED | COLLECTION_REFUSED (exit 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools import aggregate_l1_factorial as agg  # noqa: E402
from tools import l1_factorial_screen as runner  # noqa: E402

COLLECTION_SCHEMA = "agent_multi.l1_collection_manifest.v1"


def default_fetch(host: str, remote_dir: Path, stage_dir: Path) -> None:
    """rsync one seed subtree from its source host (local = copy)."""
    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    if host in ("localhost", os.uname().nodename):
        subprocess.run(["rsync", "-a", f"{remote_dir}/",
                        f"{stage_dir}/"], check=True)
    else:
        subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes",
                        f"{host}:{remote_dir}/", f"{stage_dir}/"],
                       check=True)


def default_replicate(replica_host: str, sealed_root: Path,
                      replica_root: Path) -> None:
    subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes",
                    f"{sealed_root}/",
                    f"{replica_host}:{replica_root}/"], check=True)


REPLICA_VERIFIER_VERSION = "l1_replica_verifier.v2"


def default_replica_verify(replica_host: str, replica_root: Path,
                           expectations: List[dict]) -> dict:
    """On the REPLICA host (order §5/WP10): compute the whole-tree
    digest over the replica root with the same algorithm as the sealed
    source, then rehash + really load every terminal. The digest covers
    records, results, traces and manifests — any missing or modified
    file changes it."""
    script = (
        "import json,sys,hashlib\n"
        "from pathlib import Path\n"
        "payload=json.loads(sys.stdin.read())\n"
        "root=Path(payload['replica_root'])\n"
        "digest=hashlib.sha256()\n"
        "for p in sorted(root.rglob('*')):\n"
        "    if p.is_file():\n"
        "        digest.update(str(p.relative_to(root)).encode())\n"
        "        digest.update(hashlib.sha256("
        "p.read_bytes()).hexdigest().encode())\n"
        "out={'tree_digest':digest.hexdigest(),"
        "'verifier_version':'" + REPLICA_VERIFIER_VERSION + "',"
        "'terminals':[]}\n"
        "for e in payload['expectations']:\n"
        "    p=Path(e['replica_path']); r={'cell':e['cell'],"
        "'seed':e['seed']}\n"
        "    try:\n"
        "        r['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "        from stable_baselines3 import SAC\n"
        "        m=SAC.load(str(p), device='cpu')\n"
        "        r['loads']=True; r['n_updates']=int(getattr("
        "m,'_n_updates',0))\n"
        "    except Exception as ex:\n"
        "        r['loads']=False; r['error']=f'{type(ex).__name__}:{ex}'\n"
        "    out['terminals'].append(r)\n"
        "print(json.dumps(out))\n")
    # ssh re-parses the joined argument string on the remote shell, so
    # the multi-line script MUST be shell-quoted as one argument
    # (observed 2026-08-09: unquoted newlines executed line-by-line).
    remote_command = (
        "/home/harveybc/anaconda3/envs/trading-stack/bin/python -c "
        + shlex.quote(script))
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", replica_host, remote_command],
        input=json.dumps({"replica_root": str(replica_root),
                          "expectations": expectations}),
        capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"replica verify failed: "
                           f"{proc.stderr.strip()[-300:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def tree_digest(root: Path) -> str:
    # Single canonical implementation shared with the aggregator's
    # envelope gate.
    return agg.tree_digest(root)


def collect(*, contract: dict, experiment_id: str, collection_root: Path,
            source_root: Path | None = None,
            fetch_fn: Callable[[str, Path, Path], None] = default_fetch,
            replica_host: str | None = None,
            replica_root: Path | None = None,
            replicate_fn: Callable[[str, Path, Path], None] =
            default_replicate,
            replica_verify_fn: Callable[[str, Path, List[dict]],
                                        List[dict]] =
            default_replica_verify) -> dict:
    refusals: List[str] = []
    source_root = source_root or \
        Path(contract["output_root"]).expanduser()
    stage = collection_root / "staged" / experiment_id
    sealed_target = collection_root / "sealed" / experiment_id
    if stage.exists() or sealed_target.exists():
        return {"outcome": "COLLECTION_REFUSED",
                "refusals": [
                    f"collection for {experiment_id} already staged or "
                    "sealed under this root — staging and sealing never "
                    "overwrite; use a fresh collection root"]}

    assignments = contract.get("assignments") or {}
    source_hosts: Dict[str, str] = {}
    for seed, assignment in sorted(assignments.items()):
        host = assignment.get("hostname")
        source_hosts[seed] = host
        remote_dir = source_root / experiment_id / f"seed{seed}"
        stage_dir = stage / f"seed{seed}"
        try:
            fetch_fn(host, remote_dir, stage_dir)
        except Exception as exc:
            refusals.append(f"seed {seed}: fetch from {host} failed: "
                            f"{type(exc).__name__}: {exc}")

    # Validate every staged record and its referenced files.
    seen_cell_identities: Dict[str, str] = {}
    seen_cells: Dict[str, str] = {}
    records: Dict[str, dict] = {}
    for rec_path in sorted(stage.rglob("l1_cell_record.json")):
        try:
            record = json.loads(rec_path.read_text())
        except Exception as exc:
            refusals.append(f"unreadable staged record {rec_path}: {exc}")
            continue
        name = f"seed{record.get('seed')}/{record.get('cell')}"
        cell_id = str(record.get("cell_identity"))
        if name in seen_cells:
            refusals.append(f"duplicate seed/cell {name} staged twice")
            continue
        if cell_id in seen_cell_identities:
            refusals.append(
                f"duplicate cell identity {cell_id} ({name} vs "
                f"{seen_cell_identities[cell_id]}) — retry duplicates "
                "are rejected even when bytes differ")
            continue
        seen_cells[name] = str(rec_path)
        seen_cell_identities[cell_id] = name
        records[name] = record
        # Referenced-file verification through the sealed-root path
        # authority (finding 189): the recorded absolute source path is
        # resolved by its experiment-relative suffix, never read as-is.
        stage_resolver = agg.SealedPathResolver(stage.parent,
                                                experiment_id)
        try:
            staged_terminal = stage_resolver.resolve(
                record.get("terminal_model_path"))
        except ValueError as exc:
            refusals.append(f"{name}: terminal path unresolvable: {exc}")
            staged_terminal = None
        if staged_terminal is None or not staged_terminal.is_file():
            refusals.append(f"{name}: staged terminal artifact missing")
        elif sysid.sha_file(staged_terminal) != \
                record.get("terminal_model_sha256"):
            refusals.append(f"{name}: staged terminal artifact does not "
                            "match its recorded sha")

    expected = {f"seed{s}/{c}" for s in contract["anchors"]
                for c in contract["cells"]}
    for missing in sorted(expected - set(records)):
        refusals.append(f"missing staged record {missing}")

    refusals.extend(agg.cross_record_uniformity(records))

    manifest = {
        "schema": COLLECTION_SCHEMA,
        "experiment_id": experiment_id,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "source_hosts": source_hosts,
        "records": {name: {"cell_identity": rec.get("cell_identity"),
                           "terminal_model_sha256":
                               rec.get("terminal_model_sha256")}
                    for name, rec in sorted(records.items())},
        "refusals": refusals,
        "collector_identity": sysid.source_tree_identity(
            sysid.resolve_repo_root(Path(__file__))),
    }
    if refusals:
        manifest["outcome"] = "COLLECTION_REFUSED"
        _atomic(collection_root / "collection_manifest.json", manifest)
        return manifest

    # Seal: digest over the staged tree, then atomic rename to sealed/.
    manifest["collection_tree_digest"] = tree_digest(stage)
    sealed = collection_root / "sealed" / experiment_id
    sealed.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, sealed)
    manifest["sealed_root"] = str(sealed)

    # Independent whole-tree replica (order §5/WP10, finding 190).
    # Without a verified replica the seal is a typed PARTIAL outcome
    # that aggregation refuses.
    if not replica_host:
        manifest["outcome"] = "COLLECTION_SEALED_WITHOUT_REPLICA"
        manifest["replica"] = None
        manifest["refusals"] = [
            "independent replica is mandatory before aggregation "
            "(finding 190); re-run with --replica-host"]
        _atomic(collection_root / "collection_manifest.json", manifest)
        return manifest

    replica_root = replica_root or Path(
        f"/home/harveybc/.local/share/agent-multi/"
        f"l1_replica_{experiment_id}")
    resolver = agg.SealedPathResolver(sealed.parent, experiment_id)
    expectations = []
    for name, rec in sorted(records.items()):
        staged_terminal = resolver.resolve(rec["terminal_model_path"])
        expectations.append({
            "cell": rec["cell"], "seed": rec["seed"],
            "replica_path": str(replica_root /
                                staged_terminal.relative_to(sealed)),
            "expected_sha256": rec["terminal_model_sha256"],
        })
    try:
        replicate_fn(replica_host, sealed, replica_root)
        verification = replica_verify_fn(replica_host, replica_root,
                                         expectations)
    except Exception as exc:
        manifest["outcome"] = "COLLECTION_REFUSED"
        manifest["refusals"] = [f"replication failed: "
                                f"{type(exc).__name__}: {exc}"]
        _atomic(collection_root / "collection_manifest.json", manifest)
        return manifest
    replica_digest = verification.get("tree_digest")
    if replica_digest != manifest["collection_tree_digest"]:
        refusals.append(
            f"replica tree digest {str(replica_digest)[:16]}… does not "
            "equal the sealed source tree digest — a missing or "
            "modified file on the replica refuses the collection")
    terminals = verification.get("terminals") or []
    by_key = {(r.get("seed"), r.get("cell")): r for r in terminals}
    for exp in expectations:
        got = by_key.get((exp["seed"], exp["cell"]))
        if not got:
            refusals.append(f"replica verify missing seed"
                            f"{exp['seed']}/{exp['cell']}")
        elif got.get("sha256") != exp["expected_sha256"]:
            refusals.append(f"replica sha mismatch seed"
                            f"{exp['seed']}/{exp['cell']}")
        elif got.get("loads") is not True:
            refusals.append(f"replica load failed seed{exp['seed']}/"
                            f"{exp['cell']}: {got.get('error')}")
    manifest["replica"] = {
        "host": replica_host,
        "root": str(replica_root),
        "source_tree_digest": manifest["collection_tree_digest"],
        "replica_tree_digest": replica_digest,
        "digests_match": replica_digest ==
        manifest["collection_tree_digest"],
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_version": verification.get("verifier_version"),
        "verifier_collector_identity": manifest["collector_identity"],
        "verification": terminals,
    }
    if refusals:
        manifest["outcome"] = "COLLECTION_REFUSED"
        manifest["refusals"] = refusals
    else:
        manifest["outcome"] = "COLLECTION_SEALED"
    _atomic(collection_root / "collection_manifest.json", manifest)
    return manifest


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True,
                              default=str) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--collection-root", required=True)
    parser.add_argument("--replica-host", default=None)
    parser.add_argument("--aggregate", action="store_true",
                        help="run aggregation from the sealed root "
                             "after a successful seal")
    args = parser.parse_args()
    contract = runner.load_contract()
    manifest = collect(contract=contract,
                       experiment_id=args.experiment_id,
                       collection_root=Path(args.collection_root),
                       replica_host=args.replica_host)
    print(json.dumps({"outcome": manifest["outcome"],
                      "refusals": manifest.get("refusals", []),
                      "digest": manifest.get("collection_tree_digest")},
                     default=str), flush=True)
    if manifest["outcome"] != "COLLECTION_SEALED":
        return 3
    if args.aggregate:
        # Findings 190/196/197: BOTH public CLIs go through the same
        # envelope authority — sealed manifest, matching replica proof,
        # fresh sealed rehash, publication OUTSIDE the seal.
        try:
            result = agg.aggregate_from_collection(
                Path(args.collection_root), args.experiment_id,
                contract=contract)
        except RuntimeError as exc:
            print(json.dumps({"outcome": "AGGREGATION_REFUSED",
                              "reason": str(exc)}), flush=True)
            return 3
        print(json.dumps({"outcome": result["outcome"],
                          "sealed_digest_unchanged": True}), flush=True)
        if result["outcome"] == "INCONCLUSIVE" or result["refusals"]:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
