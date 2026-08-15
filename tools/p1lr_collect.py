#!/usr/bin/env python3
"""P1LR sealed collection + typed replica proof (finding 225).

The screen verdict of ``tools/p1_difficulty_lr_factorial.py`` REFUSES
without a typed replica proof; THIS collector is the only producer of
that proof. It reuses the corrected deterministic custody primitives of
``tools/m0_l1_ladder_collect.py`` (deterministic relative-path
resolution, fresh rehash, per-entry (identity, seed, cell, relative
path, sha256, loads) binding, whole-tree digest recomputed ON the
replica) and the factorial collector's rsync/ssh transports — no ad hoc
rsync or hash logic is duplicated here.

Flow (stage + replicate atomically per seed batch):

1. fetch each seed's subtree ``<output_root>/<exp_id>/seed<seed>`` from
   its CONTRACT-ASSIGNED host into a fresh staging root (staging and
   sealing never overwrite);
2. validate exactly 16 (seed, cell) records: uniform experiment
   identity and contract sha, 16 DISTINCT cell identities, finding-223
   terminal custody (non-empty path+sha, deterministic relative
   resolution — basename-first-match forbidden — fresh hash equality);
3. seal (atomic rename), per-file manifest + whole-tree digest;
4. replicate to the independent host, recompute the digest THERE, and
   really load every terminal (stable_baselines3 SAC.load);
5. bind each of the 16 replica loads to (experiment_identity,
   contract_sha256, seed, cell, terminal_relative_path,
   terminal_model_sha256, loads=true) and write the typed proof file
   ``agent_multi.p1lr_replica_proof.v1`` — the ONLY artifact the screen
   verdict accepts via ``--replica-proof``.

Zero, missing, duplicate, swapped, foreign, hash-altered or
``loads=false`` entries refuse the collection; source records and
terminals are never mutated.

Typed outcomes: COLLECTION_SEALED | COLLECTION_REFUSED (exit 3).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import collect_l1_factorial as fact  # noqa: E402
from tools import m0_l1_ladder_collect as ladder_collect  # noqa: E402
from tools import p1_difficulty_lr_factorial as p1  # noqa: E402

COLLECTION_SCHEMA = "agent_multi.p1lr_collection.v1"
RECORD_NAME = "cell_record.json"

# Reused custody primitives — ONE implementation (finding 223/225).
_sha256 = ladder_collect._sha256
_atomic = ladder_collect._atomic
_resolve_relative = ladder_collect._resolve_arm_relative
_verify_terminal_proof = ladder_collect._verify_terminal_proof
_custody_expectations = ladder_collect._custody_expectations
default_fetch = fact.default_fetch
default_replicate = ladder_collect.default_replicate
default_replica_verify = ladder_collect.default_replica_verify
tree_digest = fact.tree_digest


def _batch_key(seed: int, cell: str) -> str:
    """Composite expectation key: the ladder proof binder keys on one
    'cell' string, and the p1lr factorial repeats cell names across
    seeds — so the binding key carries both."""
    return f"seed{seed}:{cell}"


def _cell_custody(record: dict, seed: int, cell: str,
                  cell_dir: Path) -> tuple:
    """Finding 223 custody for one (seed, cell) record, via the reused
    deterministic relative resolution (the component after the CELL
    name in the recorded absolute path; zero/multiple staged matches
    refuse)."""
    refusals: List[str] = []
    t_path = record.get("terminal_model_path")
    t_sha = record.get("terminal_model_sha256")
    if not t_path or not t_sha:
        return [f"seed{seed}/{cell}: terminal custody incomplete — "
                "non-empty terminal_model_path AND "
                "terminal_model_sha256 are mandatory (finding 223)"], \
            None
    rel, errors = _resolve_relative(str(t_path), cell, cell_dir)
    if errors:
        return [f"seed{seed}/{cell}: terminal: {e}" for e in errors], \
            None
    fresh = _sha256(cell_dir / rel)
    if fresh != t_sha:
        return [f"seed{seed}/{cell}: staged terminal {rel} hash "
                f"{fresh[:16]}… differs from the recorded "
                "terminal_model_sha256"], None
    relative = str(Path(f"seed{seed}") / cell / rel)
    expected = p1.expected_terminal_relative(record)
    if expected != relative:
        refusals.append(
            f"seed{seed}/{cell}: staged relative path {relative!r} != "
            f"the verdict-side derivation {expected!r} — producer and "
            "consumer must bind the SAME path")
        return refusals, None
    return refusals, {"seed": int(seed), "relative_path": relative,
                      "sha256": t_sha}


def collect(*, contract: dict, experiment_identity: str,
            collection_root: Path,
            fetch_fn: Callable[[str, Path, Path], None] = default_fetch,
            replica_host: str | None = None,
            replica_root: Path | None = None,
            replicate_fn: Callable[[str, Path, Path], None] =
            default_replicate,
            replica_verify_fn: Callable[[str, Path, List[dict]], dict] =
            default_replica_verify,
            proof_out: Path | None = None,
            mode: str = "screen") -> dict:
    refusals: List[str] = []
    source_root = p1.output_root_for_mode(contract, mode)
    stage = collection_root / "staged" / experiment_identity
    sealed_root = collection_root / "sealed" / experiment_identity
    if stage.exists() or sealed_root.exists():
        return {"outcome": "COLLECTION_REFUSED",
                "refusals": [
                    f"collection for {experiment_identity} already "
                    "staged or sealed under this root — staging and "
                    "sealing never overwrite; use a fresh root"]}

    assignments = contract.get("assignments") or {}
    source_hosts: Dict[str, str] = {}
    for seed in p1.SEEDS:
        host = (assignments.get(str(seed)) or {}).get("hostname")
        if not host:
            refusals.append(f"seed {seed}: no contract-assigned host")
            continue
        source_hosts[str(seed)] = host
        remote_dir = source_root / experiment_identity / f"seed{seed}"
        try:
            fetch_fn(host, remote_dir, stage / f"seed{seed}")
        except Exception as exc:
            refusals.append(f"seed {seed}: fetch from {host} failed: "
                            f"{type(exc).__name__}: {exc}")

    records: Dict[str, dict] = {}
    custody: Dict[str, dict] = {}
    cell_identities: Dict[str, str] = {}
    contract_hashes = set()
    # v1 records carry the v1 schema, v2 records the v2 schema — the
    # loaded contract decides which one this collection may seal.
    expected_record_schema = p1.record_schema_for(contract)
    for seed in p1.SEEDS:
        for cell in p1.CELLS:
            name = f"seed{seed}/{cell}"
            rec_path = stage / f"seed{seed}" / cell / RECORD_NAME
            if not rec_path.is_file():
                refusals.append(f"{name}: staged record missing")
                continue
            try:
                record = json.loads(rec_path.read_text())
            except Exception as exc:
                refusals.append(f"{name}: unreadable record: {exc}")
                continue
            if record.get("schema") != expected_record_schema \
                    or record.get("seed") != seed \
                    or record.get("cell") != cell:
                refusals.append(f"{name}: wrong schema or seed/cell")
                continue
            if record.get("experiment_identity") != \
                    experiment_identity:
                refusals.append(
                    f"{name}: identity fragmentation — record carries "
                    f"{record.get('experiment_identity')!r}, "
                    f"collection demands {experiment_identity!r}")
                continue
            cell_id = str(record.get("cell_identity"))
            if cell_id in cell_identities:
                refusals.append(
                    f"duplicate cell identity {cell_id} ({name} vs "
                    f"{cell_identities[cell_id]})")
                continue
            cell_identities[cell_id] = name
            contract_hashes.add(record.get("contract_sha256"))
            cell_refusals, cell_custody = _cell_custody(
                record, seed, cell, stage / f"seed{seed}" / cell)
            refusals.extend(cell_refusals)
            if cell_custody is not None:
                custody[_batch_key(seed, cell)] = cell_custody
            records[name] = record
    if len(contract_hashes) > 1:
        refusals.append(f"contract hash not uniform: "
                        f"{sorted(map(str, contract_hashes))}")
    if contract_hashes and \
            contract_hashes != {contract["_contract_sha256"]}:
        refusals.append("records bind a different contract sha than "
                        "the loaded contract")
    if not refusals and len(records) != 16:
        refusals.append(f"{len(records)}/16 records staged — the "
                        "proof demands exactly 16 source records")

    if refusals:
        return {"outcome": "COLLECTION_REFUSED", "refusals": refusals}

    # Seal atomically; digest the sealed tree.
    sealed_root.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(sealed_root)
    files = {str(p.relative_to(sealed_root)): _sha256(p)
             for p in sorted(sealed_root.rglob("*")) if p.is_file()}
    digest = tree_digest(sealed_root)
    manifest = {
        "schema": COLLECTION_SCHEMA,
        "outcome": "COLLECTION_SEALED",
        "experiment_identity": experiment_identity,
        "mode": mode,
        "contract_sha256": contract["_contract_sha256"],
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "source_hosts": source_hosts,
        "cell_identities": cell_identities,
        "terminals": custody,
        "files": files,
        "collection_tree_digest": digest,
        "refusals": [],
    }

    if replica_host:
        replica_root = replica_root or sealed_root
        expectations = _custody_expectations(custody, replica_root)
        try:
            replicate_fn(replica_host, sealed_root, replica_root)
            proof = replica_verify_fn(replica_host, replica_root,
                                      expectations)
        except Exception as exc:
            manifest["outcome"] = "COLLECTION_REFUSED"
            manifest["refusals"].append(
                f"replication to {replica_host} failed: "
                f"{type(exc).__name__}: {exc}")
            proof = None
        if proof is not None:
            if proof.get("tree_digest") != digest:
                manifest["outcome"] = "COLLECTION_REFUSED"
                manifest["refusals"].append(
                    f"replica digest {proof.get('tree_digest')} != "
                    f"sealed {digest} — replica is not proof")
            binding_refusals = _verify_terminal_proof(
                expectations, proof.get("terminals"))
            if binding_refusals:
                manifest["outcome"] = "COLLECTION_REFUSED"
                manifest["refusals"].extend(binding_refusals)
        manifest["replica"] = {"host": replica_host,
                               "root": str(replica_root),
                               "expectations": expectations,
                               "proof": proof}
        if manifest["outcome"] == "COLLECTION_SEALED":
            manifest["replica_proof_file"] = str(_write_proof_file(
                contract=contract,
                experiment_identity=experiment_identity,
                collection_root=collection_root,
                records=records, custody=custody,
                replica_host=replica_host, digest=digest,
                proof_out=proof_out))
    else:
        manifest["outcome"] = "COLLECTION_REFUSED"
        manifest["refusals"].append(
            "no replica host — the screen gate demands 16 replica "
            "load proofs; a seal without a replica proves nothing "
            "(finding 225)")

    _atomic(collection_root /
            f"p1lr_collection_manifest_{experiment_identity}.json",
            manifest)
    return manifest


def _write_proof_file(*, contract: dict, experiment_identity: str,
                      collection_root: Path, records: Dict[str, dict],
                      custody: Dict[str, dict], replica_host: str,
                      digest: str, proof_out: Path | None) -> Path:
    """The typed 16-entry replica proof the screen verdict consumes.
    Written ONLY after the replica digest matched and every terminal
    load bound; each entry carries the full six-fact binding."""
    proofs = []
    for seed in p1.SEEDS:
        for cell in p1.CELLS:
            record = records[f"seed{seed}/{cell}"]
            bound = custody[_batch_key(seed, cell)]
            proofs.append({
                "experiment_identity": experiment_identity,
                "contract_sha256": contract["_contract_sha256"],
                "seed": int(seed),
                "cell": cell,
                "cell_identity": record.get("cell_identity"),
                "terminal_relative_path": bound["relative_path"],
                "terminal_model_sha256": bound["sha256"],
                "loads": True,
            })
    payload = {
        "schema": p1.REPLICA_PROOF_SCHEMA,
        "experiment_identity": experiment_identity,
        "contract_sha256": contract["_contract_sha256"],
        "replica_host": replica_host,
        "collection_tree_digest": digest,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "proofs": proofs,
    }
    out = proof_out or (collection_root /
                        f"p1lr_replica_proof_{experiment_identity}"
                        ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic(out, payload)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--collection-root", type=Path, default=None,
                        help="staging/seal root; a v2 contract "
                             "defaults this to its replica."
                             "collection_root (the NEW v2 root, "
                             "order §6)")
    parser.add_argument("--replica-host", default=None,
                        help="independent replica host; the typed "
                             "proof is only written after 16 bound "
                             "loads succeed THERE (a v2 contract "
                             "defaults this to replica.replica_host)")
    parser.add_argument("--mode", choices=list(p1.MODES),
                        default="screen")
    parser.add_argument("--contract", type=Path,
                        default=p1.CONTRACT_PATH)
    parser.add_argument("--proof-out", type=Path, default=None,
                        help="replica proof file path (default: "
                             "<collection-root>/p1lr_replica_proof_"
                             "<exp-id>.json)")
    args = parser.parse_args()
    contract = p1.load_contract(args.contract)
    replica_defaults = contract.get("replica") or {}
    collection_root = args.collection_root or \
        replica_defaults.get("collection_root")
    if not collection_root:
        parser.error("--collection-root is required (this contract "
                     "declares no replica.collection_root)")
    replica_host = args.replica_host or \
        replica_defaults.get("replica_host")
    if not replica_host:
        parser.error("--replica-host is required (this contract "
                     "declares no replica.replica_host)")
    manifest = collect(
        contract=contract, experiment_identity=args.experiment_id,
        collection_root=Path(collection_root).expanduser(),
        replica_host=replica_host, proof_out=args.proof_out,
        mode=args.mode)
    print(json.dumps({
        "outcome": manifest["outcome"],
        "refusals": manifest.get("refusals", []),
        "digest": manifest.get("collection_tree_digest"),
        "replica_proof_file": manifest.get("replica_proof_file"),
    }, default=str), flush=True)
    return 0 if manifest["outcome"] == "COLLECTION_SEALED" else 3


if __name__ == "__main__":
    sys.exit(main())
