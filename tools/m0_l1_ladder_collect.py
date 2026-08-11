#!/usr/bin/env python3
"""Sealed collection + replica + contrast table for the mechanism
ladder (order §3.5-§3.6, finding 220).

Pulls each arm subtree from its contract-assigned host into a staged
collection root, refuses identity fragmentation (every record must
carry the SAME diagnostic identity), refuses duplicate arm identities
and missing/hash-mismatched terminal artifacts, seals the tree with a
per-file manifest + whole-tree digest, replicates the sealed tree to
one independent host with the digest recomputed ON the replica, and
only after seal+replica allows the five-row contrast table to be
published OUTSIDE the sealed root, with a post-write digest re-proof.

Transports are injectable (tests run socket-free); the defaults reuse
the factorial collector's rsync/ssh primitives.

Typed outcomes: COLLECTION_SEALED | COLLECTION_REFUSED (exit 3);
publish adds TABLE_PUBLISHED | PUBLISH_REFUSED (exit 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import collect_l1_factorial as fact  # noqa: E402
from tools import m0_l1_mechanism_ladder as runner  # noqa: E402


def default_replicate(replica_host: str, sealed_root: Path,
                      replica_root: Path) -> None:
    """The factorial replicate assumes the remote parent exists; a
    fresh ladder collection root does not (rsync exit 11, observed
    2026-08-11) — create it first, then delegate."""
    import subprocess
    subprocess.run(["ssh", "-o", "BatchMode=yes", replica_host,
                    f"mkdir -p {shlex.quote(str(replica_root))}"],
                   check=True, timeout=60, stdin=subprocess.DEVNULL)
    fact.default_replicate(replica_host, sealed_root, replica_root)

LADDER_COLLECTION_SCHEMA = "agent_multi.m0_l1_ladder_collection.v1"
CONTRAST_SCHEMA = "agent_multi.m0_l1_ladder_contrast_table.v1"
RECORD_NAME = "ladder_arm_record.json"
D1_RECORD_NAME = "d1_evaluator_record.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True,
                              default=str) + "\n")
    tmp.replace(path)


def collect(*, contract: dict, diagnostic_identity: str,
            collection_root: Path,
            fetch_fn: Callable[[str, Path, Path], None] =
            fact.default_fetch,
            replica_host: str | None = None,
            replica_root: Path | None = None,
            replicate_fn: Callable[[str, Path, Path], None] =
            default_replicate,
            replica_verify_fn: Callable[[str, Path, List[dict]], dict] =
            fact.default_replica_verify) -> dict:
    refusals: List[str] = []
    source_root = Path(contract["output_root"]).expanduser()
    stage = collection_root / "staged" / diagnostic_identity
    sealed_root = collection_root / "sealed" / diagnostic_identity
    if stage.exists() or sealed_root.exists():
        return {"outcome": "COLLECTION_REFUSED",
                "refusals": [
                    f"collection for {diagnostic_identity} already "
                    "staged or sealed under this root — staging and "
                    "sealing never overwrite; use a fresh root"]}

    assignments = contract.get("assignments") or {}
    source_hosts: Dict[str, str] = {}
    for arm, assignment in sorted(assignments.items()):
        host = assignment.get("hostname")
        source_hosts[arm] = host
        remote_dir = source_root / diagnostic_identity / arm
        try:
            fetch_fn(host, remote_dir, stage / arm)
        except Exception as exc:
            refusals.append(f"arm {arm}: fetch from {host} failed: "
                            f"{type(exc).__name__}: {exc}")

    # Validate every staged record: one diagnostic identity, unique
    # arm identities, uniform contract hash, referenced terminals.
    records: Dict[str, dict] = {}
    seen_arm_ids: Dict[str, str] = {}
    contract_hashes = set()
    for arm in sorted(assignments):
        rec_path = stage / arm / RECORD_NAME
        if not rec_path.is_file():
            refusals.append(f"arm {arm}: staged record missing")
            continue
        try:
            record = json.loads(rec_path.read_text())
        except Exception as exc:
            refusals.append(f"arm {arm}: unreadable record: {exc}")
            continue
        if record.get("arm") != arm:
            refusals.append(f"arm {arm}: record claims "
                            f"{record.get('arm')!r}")
            continue
        if record.get("diagnostic_identity") != diagnostic_identity:
            refusals.append(
                f"arm {arm}: identity fragmentation — record carries "
                f"{record.get('diagnostic_identity')!r}, collection "
                f"demands {diagnostic_identity!r}")
            continue
        arm_id = str(record.get("arm_identity"))
        if arm_id in seen_arm_ids:
            refusals.append(f"duplicate arm identity {arm_id} "
                            f"({arm} vs {seen_arm_ids[arm_id]})")
            continue
        seen_arm_ids[arm_id] = arm
        contract_hashes.add(record.get("contract_sha256"))
        model_sha = record.get("best_model_sha256")
        model_path = record.get("best_model_path")
        if model_sha and model_path:
            staged_model = stage / arm / Path(model_path).name
            if not staged_model.is_file():
                found = list((stage / arm).rglob(Path(model_path).name))
                staged_model = found[0] if found else staged_model
            if not staged_model.is_file():
                refusals.append(f"arm {arm}: terminal artifact "
                                f"{Path(model_path).name} not staged")
            elif _sha256(staged_model) != model_sha:
                refusals.append(f"arm {arm}: staged terminal hash "
                                "differs from record")
        records[arm] = record
    if len(contract_hashes) > 1:
        refusals.append(f"contract hash not uniform: "
                        f"{sorted(map(str, contract_hashes))}")
    d1_path = stage / "D0_M0_EXACT" / D1_RECORD_NAME
    if "D0_M0_EXACT" in records and not d1_path.is_file():
        refusals.append("D1 evaluator record missing from the staged "
                        "D0 subtree — run tools/m0_l1_d1_evaluator.py "
                        "before collection")

    if refusals:
        return {"outcome": "COLLECTION_REFUSED", "refusals": refusals}

    # Seal: per-file manifest + whole-tree digest, manifest OUTSIDE.
    sealed_root.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(sealed_root)
    files = {str(p.relative_to(sealed_root)): _sha256(p)
             for p in sorted(sealed_root.rglob("*")) if p.is_file()}
    digest = fact.tree_digest(sealed_root)
    manifest = {
        "schema": LADDER_COLLECTION_SCHEMA,
        "outcome": "COLLECTION_SEALED",
        "diagnostic_identity": diagnostic_identity,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "source_hosts": source_hosts,
        "arms": sorted(records),
        "arm_identities": {a: str(r.get("arm_identity"))
                           for a, r in records.items()},
        "contract_sha256": next(iter(contract_hashes), None),
        "files": files,
        "collection_tree_digest": digest,
        "refusals": [],
    }
    if replica_host:
        replica_root = replica_root or sealed_root
        expectations = []
        for arm, record in records.items():
            model_path = record.get("best_model_path")
            if not model_path:
                continue
            rel = None
            for candidate in sealed_root.joinpath(arm).rglob(
                    Path(model_path).name):
                rel = candidate.relative_to(sealed_root)
                break
            if rel is not None:
                expectations.append(
                    {"cell": arm, "seed": contract.get("seed"),
                     "replica_path": str(replica_root / rel)})
        try:
            replicate_fn(replica_host, sealed_root, replica_root)
            proof = replica_verify_fn(replica_host, replica_root,
                                      expectations)
        except Exception as exc:
            # A transport failure is a typed refusal, never a crash —
            # the sealed tree and manifest must still land.
            manifest["outcome"] = "COLLECTION_REFUSED"
            manifest["refusals"].append(
                f"replication to {replica_host} failed: "
                f"{type(exc).__name__}: {exc}")
            proof = None
        if proof is not None and proof.get("tree_digest") != digest:
            manifest["outcome"] = "COLLECTION_REFUSED"
            manifest["refusals"].append(
                f"replica digest {proof.get('tree_digest')} != sealed "
                f"{digest} — replica is not proof")
        manifest["replica"] = {"host": replica_host,
                               "root": str(replica_root),
                               "proof": proof}
    _atomic(collection_root /
            f"ladder_collection_manifest_{diagnostic_identity}.json",
            manifest)
    return manifest


def publish_contrast_table(*, collection_root: Path,
                           diagnostic_identity: str,
                           out_json: Path) -> dict:
    """Five-row table from the SEALED collection only (order §3.5),
    published OUTSIDE the seal, with a post-write digest re-proof."""
    manifest_path = (collection_root /
                     f"ladder_collection_manifest_"
                     f"{diagnostic_identity}.json")
    if not manifest_path.is_file():
        return {"outcome": "PUBLISH_REFUSED",
                "refusals": ["no sealed collection manifest"]}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("outcome") != "COLLECTION_SEALED":
        return {"outcome": "PUBLISH_REFUSED",
                "refusals": [f"manifest outcome is "
                             f"{manifest.get('outcome')!r}"]}
    if not (manifest.get("replica") or {}).get("proof"):
        return {"outcome": "PUBLISH_REFUSED",
                "refusals": ["no replica proof — replicate before "
                             "interpretation (order §3.6)"]}
    sealed_root = collection_root / "sealed" / diagnostic_identity
    fresh = fact.tree_digest(sealed_root)
    if fresh != manifest["collection_tree_digest"]:
        return {"outcome": "PUBLISH_REFUSED",
                "refusals": [f"sealed tree digest changed: {fresh} != "
                             f"{manifest['collection_tree_digest']}"]}
    if out_json.resolve().is_relative_to(sealed_root.resolve()):
        return {"outcome": "PUBLISH_REFUSED",
                "refusals": ["publication target is inside the sealed "
                             "root"]}

    rows = []
    activity: Dict[str, bool] = {}
    for arm in sorted(manifest["arms"]):
        record = json.loads(
            (sealed_root / arm / RECORD_NAME).read_text())
        row = runner.contrast_row(record)
        facts = record.get("m0_activity_facts") or {}
        row["activity_survived_normal"] = facts.get(
            "activity_survived_normal")
        row["phase1_handoff_semantics"] = (
            (record.get("curriculum") or {}).get("post_easy") or {}
        ).get("phase1_handoff_semantics")
        row["best_easy_epoch"] = ((record.get("curriculum") or {})
                                  .get("post_easy") or {}
                                  ).get("best_easy_epoch")
        row["stop_reason"] = record.get("stop_reason")
        activity[arm] = bool(row["activity_survived_normal"])
        rows.append(row)
    d1 = json.loads((sealed_root / "D0_M0_EXACT" /
                     D1_RECORD_NAME).read_text())
    rows.insert(1, {
        "arm": "D1_EVALUATOR_ONLY",
        "label_under_m0_definition": d1.get("label_under_m0_definition"),
        "label_under_l1_definition": d1.get("label_under_l1_definition"),
        "labels_agree": d1.get("labels_agree"),
        "note": "CPU relabel of the D0 terminal — no training",
    })

    # Fail-closed interpretation, exactly the contract's rules (§3.4).
    if not activity.get("D0_M0_EXACT"):
        verdict = ("LADDER_INVALID: D0 failed to reproduce activity — "
                   "diagnose anchor/data/code before any new compute")
    elif not d1.get("labels_agree"):
        verdict = ("DEFINITION_DEFECT: D1 relabels the active D0 — the "
                   "defect is the activity definition, not learning")
    else:
        first_inactive = next(
            (a for a in ("D2_BOUNDARY_ONLY", "D3_COST_PROTECTION",
                         "D4_FULL_L1") if not activity.get(a)), None)
        if first_inactive is None:
            verdict = ("NO_TRANSITION: all arms active — the prior "
                       "collapse came from an uncontrolled identity "
                       "difference; diff every bound manifest field")
        else:
            verdict = (f"MECHANISM_NAMED: first active-to-inactive "
                       f"transition at {first_inactive}")
    table = {
        "schema": CONTRAST_SCHEMA,
        "outcome": "TABLE_PUBLISHED",
        "diagnostic_identity": diagnostic_identity,
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "collection_tree_digest": manifest["collection_tree_digest"],
        "replica_host": manifest["replica"]["host"],
        "replica_digest_equal": True,
        "rows": rows,
        "verdict": verdict,
        "scope_limits": [
            "seed 101 only — locates a deterministic mechanism, "
            "supports no superiority decision (contract rule 5)",
            "never relabels the sealed L1 factorial result "
            "2de49ea9225e2baf: that result remains INCONCLUSIVE",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    _atomic(out_json, table)
    post = fact.tree_digest(sealed_root)
    if post != manifest["collection_tree_digest"]:
        return {"outcome": "PUBLISH_REFUSED",
                "refusals": [f"post-write digest re-proof failed: "
                             f"{post}"]}
    table["post_write_digest_reproof"] = post
    _atomic(out_json, table)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-identity", required=True)
    parser.add_argument("--collection-root", required=True)
    parser.add_argument("--replica-host", default=None)
    parser.add_argument("--contract", type=Path,
                        default=runner.CONTRACT_PATH)
    parser.add_argument("--publish-table", type=Path, default=None,
                        help="after a successful seal (or against an "
                             "existing sealed root), write the "
                             "five-row contrast table to this path "
                             "OUTSIDE the sealed root")
    args = parser.parse_args()
    contract = runner.load_contract(args.contract)
    root = Path(args.collection_root).expanduser()
    sealed = (root / "sealed" / args.diagnostic_identity).exists()
    if not sealed:
        manifest = collect(contract=contract,
                           diagnostic_identity=args.diagnostic_identity,
                           collection_root=root,
                           replica_host=args.replica_host)
        print(json.dumps({"outcome": manifest["outcome"],
                          "refusals": manifest.get("refusals", []),
                          "digest": manifest.get(
                              "collection_tree_digest")},
                         default=str), flush=True)
        if manifest["outcome"] != "COLLECTION_SEALED":
            return 3
    if args.publish_table:
        table = publish_contrast_table(
            collection_root=root,
            diagnostic_identity=args.diagnostic_identity,
            out_json=args.publish_table)
        print(json.dumps({"outcome": table["outcome"],
                          "refusals": table.get("refusals", []),
                          "verdict": table.get("verdict")},
                         default=str), flush=True)
        if table["outcome"] != "TABLE_PUBLISHED":
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
