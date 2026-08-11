#!/usr/bin/env python3
"""Sealed collection + replica + contrast table for the mechanism
ladder (order §3.5-§3.6, finding 220; terminal custody per finding
223 / order WP1 §5).

Pulls each arm subtree from its contract-assigned host into a staged
collection root, refuses identity fragmentation (every record must
carry the SAME diagnostic identity), refuses duplicate arm identities
and missing/hash-mismatched terminal artifacts, seals the tree with a
per-file manifest + whole-tree digest, replicates the sealed tree to
one independent host with the digest recomputed ON the replica, and
only after seal+replica allows the five-row contrast table to be
published OUTSIDE the sealed root, with a post-write digest re-proof.

Terminal custody (finding 223): every training arm record MUST carry
non-empty terminal_model_path + terminal_model_sha256 regardless of
activity status; the staged terminal is resolved by a deterministic
RELATIVE path derived from the record's absolute path (basename-only
rglob first-match is forbidden), freshly hashed, and required to equal
the recorded sha. Replica expectations are built from the terminal
fields for EVERY arm (best-model fields are checked additionally when
present, never substituted), and the replica proof must contain
exactly ONE successful load per arm bound to (arm, seed, relative
path, sha256). Publication REPEATS the cardinality/binding check.

`--verify-terminals-only` is a READ-ONLY mode that re-runs the
terminal-custody checks against an existing sealed root + manifest
without mutating anything.

Transports are injectable (tests run socket-free); the defaults reuse
the factorial collector's rsync/ssh primitives.

Typed outcomes: COLLECTION_SEALED | COLLECTION_REFUSED (exit 3);
publish adds TABLE_PUBLISHED | PUBLISH_REFUSED (exit 3); verify adds
TERMINAL_CUSTODY_VERIFIED | TERMINAL_CUSTODY_REFUSED (exit 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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


LADDER_REPLICA_VERIFIER_VERSION = "ladder_replica_verifier.v3"


def default_replica_verify(replica_host: str, replica_root: Path,
                           expectations: List[dict]) -> dict:
    """Finding 223: same transport and digest algorithm as
    tools/collect_l1_factorial.py default_replica_verify, extended so
    every terminal entry ECHOES the replica-relative path it actually
    hashed/loaded. The collector binds each proof entry to
    (arm, seed, relative path, sha256), so a swapped, foreign or
    unloaded file can never satisfy another arm's expectation."""
    import subprocess
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
        "'verifier_version':'" + LADDER_REPLICA_VERIFIER_VERSION + "',"
        "'terminals':[]}\n"
        "for e in payload['expectations']:\n"
        "    p=Path(e['replica_path']); r={'cell':e['cell'],"
        "'seed':e['seed']}\n"
        "    try:\n"
        "        r['relative_path']=str(p.relative_to(root))\n"
        "    except ValueError:\n"
        "        r['relative_path']=None\n"
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


LADDER_COLLECTION_SCHEMA = "agent_multi.m0_l1_ladder_collection.v2"
CONTRAST_SCHEMA = "agent_multi.m0_l1_ladder_contrast_table.v1"
VERIFY_SCHEMA = "agent_multi.m0_l1_ladder_terminal_custody_verify.v1"
RECORD_NAME = "ladder_arm_record.json"
D1_RECORD_NAME = "d1_evaluator_record.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True,
                              default=str) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# terminal custody (finding 223)
# ---------------------------------------------------------------------------

def _resolve_arm_relative(abs_path: str, arm: str,
                          arm_dir: Path) -> Tuple[Path | None, List[str]]:
    """Deterministic relative resolution: derive candidate relative
    paths from the record's absolute path — everything after each path
    component equal to the arm name — and demand exactly ONE staged
    match. Zero or multiple matches refuse. Basename-only rglob
    first-match is forbidden (finding 223)."""
    parts = Path(abs_path).parts
    indices = [i for i, part in enumerate(parts) if part == arm]
    if not indices:
        return None, [f"recorded path {abs_path!r} does not contain "
                      f"the arm component {arm!r} — no deterministic "
                      "relative path can be derived"]
    candidates: List[Path] = []
    for i in indices:
        rel = Path(*parts[i + 1:])
        if rel.parts and (arm_dir / rel).is_file():
            candidates.append(rel)
    unique = sorted({str(c) for c in candidates})
    if not unique:
        return None, [f"no staged file at any relative path derived "
                      f"from {abs_path!r} — terminal artifact not "
                      "staged"]
    if len(unique) > 1:
        return None, [f"ambiguous staged terminal: {len(unique)} "
                      f"relative-path candidates {unique} — exactly "
                      "one match is required"]
    return Path(unique[0]), []


def _terminal_custody(record: dict, arm: str,
                      arm_dir: Path) -> Tuple[List[str], dict | None]:
    """Requirements 1-4 (finding 223): non-empty terminal fields,
    deterministic relative resolution, fresh hash equality, and
    best-model fields checked ADDITIONALLY when present — never as a
    substitute for the terminal."""
    refusals: List[str] = []
    t_path = record.get("terminal_model_path")
    t_sha = record.get("terminal_model_sha256")
    if not t_path or not t_sha:
        refusals.append(
            f"arm {arm}: terminal custody incomplete — every training "
            "arm must record non-empty terminal_model_path AND "
            "terminal_model_sha256 regardless of activity status "
            "(finding 223)")
        return refusals, None
    seed = record.get("seed")
    if seed is None:
        refusals.append(f"arm {arm}: record carries no seed — the "
                        "replica proof cannot be bound")
        return refusals, None
    rel, errors = _resolve_arm_relative(str(t_path), arm, arm_dir)
    if errors:
        refusals.extend(f"arm {arm}: terminal: {e}" for e in errors)
        return refusals, None
    fresh = _sha256(arm_dir / rel)
    if fresh != t_sha:
        refusals.append(
            f"arm {arm}: staged terminal {rel} hash {fresh[:16]}… "
            "differs from the recorded terminal_model_sha256 "
            f"{str(t_sha)[:16]}…")
        return refusals, None
    custody = {"seed": seed, "relative_path": str(Path(arm) / rel),
               "sha256": t_sha}
    b_path = record.get("best_model_path")
    b_sha = record.get("best_model_sha256")
    if b_path and b_sha:
        b_rel, b_errors = _resolve_arm_relative(str(b_path), arm,
                                                arm_dir)
        if b_errors:
            refusals.extend(f"arm {arm}: best-model: {e}"
                            for e in b_errors)
        elif _sha256(arm_dir / b_rel) != b_sha:
            refusals.append(f"arm {arm}: staged best model {b_rel} "
                            "hash differs from the recorded "
                            "best_model_sha256")
        else:
            custody["best_model_relative_path"] = \
                str(Path(arm) / b_rel)
            custody["best_model_sha256"] = b_sha
    return refusals, custody


def _verify_terminal_proof(expectations: List[dict],
                           terminals) -> List[str]:
    """Requirement 5 (finding 223): exactly ONE replica load proof per
    expected arm, bound to (arm, seed, relative path, sha256).
    Duplicate, missing, foreign, unbound or loads!=True entries refuse."""
    refusals: List[str] = []
    expected = {e["cell"]: e for e in expectations}
    by_arm: Dict[str, List[dict]] = {}
    for entry in list(terminals or []):
        cell = entry.get("cell")
        if cell not in expected:
            refusals.append(f"replica proof carries a foreign "
                            f"terminal entry for {cell!r}")
            continue
        by_arm.setdefault(cell, []).append(entry)
    for arm in sorted(expected):
        exp = expected[arm]
        got = by_arm.get(arm, [])
        if not got:
            refusals.append(
                f"replica proof has NO terminal load for arm {arm} — "
                "one bound successful load per arm is mandatory "
                "(finding 223)")
            continue
        if len(got) > 1:
            refusals.append(f"replica proof has {len(got)} terminal "
                            f"entries for arm {arm} — exactly one "
                            "bound load is required")
            continue
        entry = got[0]
        if entry.get("seed") != exp["seed"]:
            refusals.append(f"arm {arm}: proof seed "
                            f"{entry.get('seed')!r} != expected "
                            f"{exp['seed']!r}")
        if entry.get("relative_path") != exp["relative_path"]:
            refusals.append(
                f"arm {arm}: proof path {entry.get('relative_path')!r} "
                "is not the expected terminal relative path "
                f"{exp['relative_path']!r}")
        if entry.get("sha256") != exp["expected_sha256"]:
            refusals.append(f"arm {arm}: proof sha256 does not match "
                            "the recorded terminal_model_sha256")
        if entry.get("loads") is not True:
            refusals.append(f"arm {arm}: replica terminal did not "
                            f"load: {entry.get('error')!r}")
    return refusals


def _custody_expectations(custody_by_arm: Dict[str, dict],
                          replica_root: Path | None = None
                          ) -> List[dict]:
    expectations = []
    for arm in sorted(custody_by_arm):
        custody = custody_by_arm[arm]
        exp = {"cell": arm, "seed": custody["seed"],
               "relative_path": custody["relative_path"],
               "expected_sha256": custody["sha256"]}
        if replica_root is not None:
            exp["replica_path"] = str(
                replica_root / custody["relative_path"])
        expectations.append(exp)
    return expectations


def collect(*, contract: dict, diagnostic_identity: str,
            collection_root: Path,
            fetch_fn: Callable[[str, Path, Path], None] =
            fact.default_fetch,
            replica_host: str | None = None,
            replica_root: Path | None = None,
            replicate_fn: Callable[[str, Path, Path], None] =
            default_replicate,
            replica_verify_fn: Callable[[str, Path, List[dict]], dict] =
            default_replica_verify) -> dict:
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
    # arm identities, uniform contract hash, terminal custody for
    # EVERY training arm (finding 223).
    records: Dict[str, dict] = {}
    custody_by_arm: Dict[str, dict] = {}
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
        custody_refusals, custody = _terminal_custody(record, arm,
                                                      stage / arm)
        refusals.extend(custody_refusals)
        if custody is not None:
            custody_by_arm[arm] = custody
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
        # Finding 223: per-arm terminal custody — relative path, sha
        # and seed for EVERY training arm — recorded in the manifest
        # so publication and read-only verification can repeat the
        # binding check.
        "terminals": custody_by_arm,
        "files": files,
        "collection_tree_digest": digest,
        "refusals": [],
    }
    if replica_host:
        replica_root = replica_root or sealed_root
        # Requirement 4 (finding 223): replica expectations come from
        # the TERMINAL custody of every arm, never from best-model
        # fields alone.
        expectations = _custody_expectations(custody_by_arm,
                                             replica_root)
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
    _atomic(collection_root /
            f"ladder_collection_manifest_{diagnostic_identity}.json",
            manifest)
    return manifest


def publish_contrast_table(*, collection_root: Path,
                           diagnostic_identity: str,
                           out_json: Path) -> dict:
    """Five-row table from the SEALED collection only (order §3.5),
    published OUTSIDE the seal, with a post-write digest re-proof.
    Requirement 6 (finding 223): publication REPEATS the terminal
    custody + proof cardinality/binding check — recomputed fresh from
    the sealed records against the stored manifest's replica proof —
    rather than trusting mere presence of a replica proof."""
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

    # Finding 223 repeat check: fresh terminal custody from the SEALED
    # records, bound against the stored replica proof.
    custody_by_arm: Dict[str, dict] = {}
    custody_refusals: List[str] = []
    for arm in sorted(manifest.get("arms") or []):
        record = json.loads(
            (sealed_root / arm / RECORD_NAME).read_text())
        arm_refusals, custody = _terminal_custody(record, arm,
                                                  sealed_root / arm)
        custody_refusals.extend(arm_refusals)
        if custody is not None:
            custody_by_arm[arm] = custody
    if custody_refusals:
        return {"outcome": "PUBLISH_REFUSED",
                "refusals": custody_refusals}
    binding_refusals = _verify_terminal_proof(
        _custody_expectations(custody_by_arm),
        (manifest["replica"]["proof"] or {}).get("terminals"))
    if binding_refusals:
        return {"outcome": "PUBLISH_REFUSED",
                "refusals": binding_refusals}

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
        "terminal_proof_binding_repeated": True,
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


def verify_terminals_only(*, collection_root: Path,
                          diagnostic_identity: str) -> dict:
    """READ-ONLY terminal-custody verification (finding 223) against
    an existing sealed root + its manifest: deterministic relative
    paths, fresh hashes, per-arm proof binding, whole-tree digest
    freshness. Never writes anywhere. A binding check that legitimately
    cannot pass against a pre-finding-223 manifest (no terminal
    custody map, best-model-only proof) is reported as a typed
    legacy-format fact, not a failure of the artifacts."""
    report: dict = {
        "schema": VERIFY_SCHEMA,
        "mode": "READ_ONLY",
        "collection_root": str(collection_root),
        "diagnostic_identity": diagnostic_identity,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "arms": {},
        "typed_facts": [],
        "failures": [],
    }
    manifest_path = (collection_root /
                     f"ladder_collection_manifest_"
                     f"{diagnostic_identity}.json")
    sealed_root = collection_root / "sealed" / diagnostic_identity
    if not manifest_path.is_file():
        report["outcome"] = "TERMINAL_CUSTODY_REFUSED"
        report["failures"].append("no sealed collection manifest")
        return report
    manifest = json.loads(manifest_path.read_text())
    report["manifest_schema"] = manifest.get("schema")
    report["manifest_outcome"] = manifest.get("outcome")
    if not sealed_root.is_dir():
        report["outcome"] = "TERMINAL_CUSTODY_REFUSED"
        report["failures"].append("sealed root missing")
        return report
    legacy = "terminals" not in manifest
    report["manifest_predates_terminal_custody_schema"] = legacy
    if legacy:
        report["typed_facts"].append(
            "MANIFEST_PREDATES_TERMINAL_CUSTODY_SCHEMA: manifest "
            f"schema {manifest.get('schema')!r} carries no per-arm "
            "terminal custody map — a property of the pre-finding-223 "
            "collector that sealed it, not of the artifacts")
    fresh_digest = fact.tree_digest(sealed_root)
    report["sealed_tree_digest_recorded"] = \
        manifest.get("collection_tree_digest")
    report["sealed_tree_digest_fresh"] = fresh_digest
    if fresh_digest != manifest.get("collection_tree_digest"):
        report["failures"].append(
            f"sealed tree digest changed: fresh {fresh_digest} != "
            f"recorded {manifest.get('collection_tree_digest')}")
    custody_by_arm: Dict[str, dict] = {}
    arms = sorted(manifest.get("arms") or [])
    for arm in arms:
        entry: dict = {}
        rec_path = sealed_root / arm / RECORD_NAME
        if not rec_path.is_file():
            entry["custody_refusals"] = ["sealed record missing"]
            report["failures"].append(f"arm {arm}: sealed record "
                                      "missing")
            report["arms"][arm] = entry
            continue
        record = json.loads(rec_path.read_text())
        refusals, custody = _terminal_custody(record, arm,
                                              sealed_root / arm)
        entry["custody_refusals"] = refusals
        report["failures"].extend(refusals)
        if custody is not None:
            custody_by_arm[arm] = custody
            entry["seed"] = custody["seed"]
            entry["terminal_relative_path"] = custody["relative_path"]
            entry["terminal_sha256"] = custody["sha256"]
            entry["fresh_hash_matches_record"] = True
            files_hash = (manifest.get("files") or {}).get(
                custody["relative_path"])
            entry["manifest_files_entry_matches"] = (
                files_hash == custody["sha256"]
                if files_hash is not None else None)
            if files_hash is not None and \
                    files_hash != custody["sha256"]:
                report["failures"].append(
                    f"arm {arm}: manifest files-map hash differs from "
                    "the fresh terminal hash")
        report["arms"][arm] = entry
    if len(custody_by_arm) == len(arms) and arms:
        terminals = ((manifest.get("replica") or {}).get("proof")
                     or {}).get("terminals")
        binding = _verify_terminal_proof(
            _custody_expectations(custody_by_arm), terminals)
        if legacy:
            report["legacy_manifest_binding_gaps"] = binding
            if binding:
                report["typed_facts"].append(
                    "LEGACY_REPLICA_PROOF_UNBOUND: the stored replica "
                    "proof cannot satisfy the per-arm (arm, seed, "
                    "relative path, sha256) binding introduced by "
                    "finding 223 — recorded as a legacy manifest-"
                    "format fact, not an artifact failure; the staged "
                    "terminals above verify fresh by path and hash")
        else:
            report["proof_binding_refusals"] = binding
            report["failures"].extend(binding)
    report["outcome"] = ("TERMINAL_CUSTODY_VERIFIED"
                         if not report["failures"]
                         else "TERMINAL_CUSTODY_REFUSED")
    return report


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
    parser.add_argument("--verify-terminals-only", action="store_true",
                        help="READ-ONLY: re-run the finding-223 "
                             "terminal-custody checks (deterministic "
                             "relative paths, fresh hashes, per-arm "
                             "proof binding, digest freshness) against "
                             "an existing sealed collection root and "
                             "its manifest; mutates nothing")
    args = parser.parse_args()
    root = Path(args.collection_root).expanduser()
    if args.verify_terminals_only:
        report = verify_terminals_only(
            collection_root=root,
            diagnostic_identity=args.diagnostic_identity)
        print(json.dumps(report, indent=1, sort_keys=True,
                         default=str), flush=True)
        return 0 if report["outcome"] == "TERMINAL_CUSTODY_VERIFIED" \
            else 3
    contract = runner.load_contract(args.contract)
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
