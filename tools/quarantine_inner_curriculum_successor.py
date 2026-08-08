#!/usr/bin/env python3
"""WP0 — atomic quarantine of the invalid M0 mechanism_pass successor.

AUD-F1-20260808-159 containment, corrected per findings 166-168
(MUSASHI_TO_GENERAL_SATOSHI_III_WP0_QUARANTINE_CORRECTION_ORDER):

- ONE validator proves the complete quarantined state on BOTH the
  first-run postcondition and every retry: exact supersession schema,
  launch_eligible is False, exact reason/observation, lowercase sha
  syntax, retired path confined beneath queue/retired/<sha>/ with no
  traversal or symlink escape, retired bytes hashing to the recorded
  sha, and a correction envelope whose five bindings all recompute.
  A schema string alone is never ALREADY_QUARANTINED (finding 167).
- Containment survives incomplete evidence: launch ineligibility is
  installed first; missing canonical evidence yields the typed nonzero
  QUARANTINED_EVIDENCE_INCOMPLETE with historical_evidence_immutable
  'unavailable', and a later retry may complete the envelope
  (finding 168).
- Consumer status comes from a typed per-source inventory — canonical
  JSON, line-aware JSONL, read-only SQLite over the known HistoryStore
  tables, raw text for logs/CSV — plus a code-level consumer search
  bound to the audited revision. Unknown or unreadable relevant sources
  make the claim 'unavailable', never false. A root is listed as
  scanned only when every file in it was classified (finding 166).

Historical M0 evidence is never edited: not m0_aggregation.json, not
the 16 records, not any model ZIP. Corrupted supersession states are
preserved append-only under queue/retired/corrupt-<sha>/ before repair.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SUPERSESSION_SCHEMA = "agent_multi.inner_curriculum_successor_supersession.v1"
ENVELOPE_SCHEMA = "agent_multi.m0_correction_envelope.v1"
REASON_FINDING = "AUD-F1-20260808-159"
PRESERVED_OBSERVATION = ("reduced normal LR/duration retained activity;"
                         " easy contribution unmeasured")
SUCCESSOR_NAME = "m0_successor_mechanism_pass.json"
DEFAULT_ROOT = Path.home() / (
    ".local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1")
LEDGER_CANDIDATES = (
    Path.home() / ".local/state/agent-multi/doin-campaigns",
    Path.home() / ".local/share/agent-multi/eth_curriculum_decision_20260807_v2",
)
LEDGER_ROOTS = LEDGER_CANDIDATES        # back-compat alias
CANONICAL_EVIDENCE = ("m0_aggregation.json", "m0_final_table.csv",
                      "m0_fleet_manifest.json")
SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}
TEXT_SUFFIXES = {".log", ".csv", ".txt", ".jsonl"}
# Binary adjunct formats where a raw byte scan IS the declared,
# complete inspection method for our ASCII needles (the same method
# the auditor's independent broader scan used): model archives,
# SQLite WAL/SHM side files (WAL may hold uncommitted pages), lock
# files, keys and recovery snapshots.
RAW_SCAN_SUFFIXES = (".zip", ".lock", ".pem", "-wal", "-shm")
RAW_SCAN_MARKERS = (".recovery-", ".before-")
# SQLite-internal bookkeeping; never a consumer ledger.
SQLITE_SYSTEM_TABLES = {"sqlite_sequence"}
KNOWN_SQLITE_TABLES = {
    "campaigns": ("job_id", "domain_id", "plan_hash", "status",
                  "artifact_sha256"),
    "worker_events": ("job_id", "node_id", "worker_id", "event",
                      "detail_json"),
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    dir_fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


# --------------------------------------------------- state validation


def _envelope_bindings(root: Path, target: Path,
                       retired: Path | None) -> tuple[dict, list[str]]:
    """Recompute the five bindings; a missing file yields a None
    binding and a named problem — never silently."""
    problems: list[str] = []
    bindings: dict = {}
    for name, key in (
        ("m0_aggregation.json", "m0_aggregation_sha256"),
        ("m0_final_table.csv", "m0_final_table_csv_sha256"),
        ("m0_fleet_manifest.json", "m0_fleet_manifest_sha256"),
    ):
        path = root / name
        if path.is_file():
            bindings[key] = _sha(path)
        else:
            bindings[key] = None
            problems.append(f"canonical evidence missing: {name}")
    bindings["successor_supersession_sha256"] = (
        _sha(target) if target.is_file() else None)
    if retired is not None and retired.is_file():
        bindings["retired_successor_sha256"] = _sha(retired)
    else:
        bindings["retired_successor_sha256"] = None
        problems.append("retired original missing")
    return bindings, problems


def validate_quarantined_state(root: Path) -> tuple[list[str], dict]:
    """The single validator used by first-run postconditions AND every
    retry (finding 167). Empty problem list == fully proven state."""
    problems: list[str] = []
    queue = root / "queue"
    target = queue / SUCCESSOR_NAME
    facts: dict = {"target": str(target)}

    if not target.is_file():
        return ["active successor path missing"], facts
    try:
        current = json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        return [f"active successor unparseable: {exc}"], facts

    if current.get("schema") != SUPERSESSION_SCHEMA:
        problems.append("active path does not carry the supersession schema")
        return problems, facts
    if current.get("launch_eligible") is not False:
        problems.append("supersession is NOT launch-ineligible")
    if current.get("reason_finding") != REASON_FINDING:
        problems.append("reason_finding differs from the audited finding")
    if current.get("preserved_observation") != PRESERVED_OBSERVATION:
        problems.append("preserved_observation text differs")
    supersedes = current.get("supersedes_sha256")
    retired = None
    if not (isinstance(supersedes, str) and _SHA_RE.fullmatch(supersedes)):
        problems.append("supersedes_sha256 is not a lowercase sha256")
    else:
        expected_dir = (queue / "retired" / supersedes).resolve()
        retired = expected_dir / SUCCESSOR_NAME
        recorded = current.get("retired_path")
        if recorded:
            resolved = Path(recorded).resolve()
            if resolved != retired or expected_dir not in resolved.parents:
                problems.append(
                    "retired_path escapes queue/retired/<sha>/ or does"
                    " not match the content address")
        if not retired.is_file():
            problems.append("retired original missing")
        elif _sha(retired) != supersedes:
            problems.append("retired bytes do not hash to supersedes_sha256")

    envelope_path = root / "m0_correction_envelope_v1.json"
    facts["envelope_path"] = str(envelope_path)
    if not envelope_path.is_file():
        problems.append("correction envelope missing")
    else:
        try:
            envelope = json.loads(envelope_path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"correction envelope unparseable: {exc}")
            envelope = {}
        if envelope.get("schema") != ENVELOPE_SCHEMA:
            problems.append("correction envelope schema differs")
        expected, binding_problems = _envelope_bindings(root, target, retired)
        problems.extend(binding_problems)
        recorded = envelope.get("bindings") or {}
        for key, value in expected.items():
            if value is None:
                continue
            if recorded.get(key) != value:
                problems.append(
                    f"envelope binding {key} does not recompute")
        if any(recorded.get(k) is None for k in expected):
            problems.append("envelope carries a null binding")
        inspection = envelope.get("consumer_inspection") or {}
        if "sources" not in inspection or "code_level" not in inspection:
            problems.append(
                "envelope consumer inspection is not the typed"
                " per-source inventory (finding 166)")
    facts["supersedes_sha256"] = supersedes
    return problems, facts


# ------------------------------------------------- consumer inventory


def _scan_text(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _inspect_sqlite(path: Path, needles: tuple[str, ...]) -> dict:
    record = {"path": str(path), "format": "sqlite",
              "sha256": _sha(path), "outcome": "no_reference",
              "inspected": [], "parse_error": None}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            for table, columns in KNOWN_SQLITE_TABLES.items():
                if table not in tables:
                    continue
                present = [row[1] for row in connection.execute(
                    f"PRAGMA table_info({table})")]
                usable = [c for c in columns if c in present]
                record["inspected"].append(
                    {"table": table, "columns": usable})
                for row in connection.execute(
                        f"SELECT {', '.join(usable)} FROM {table}"):
                    if _scan_text(" ".join(str(v) for v in row), needles):
                        record["outcome"] = "reference_found"
            unknown = (tables - set(KNOWN_SQLITE_TABLES)
                       - SQLITE_SYSTEM_TABLES)
            for table in sorted(unknown):
                # read-only generic full scan: every column of every
                # row, stringified — STRONGER than known-column
                # inspection, so an unknown table never hides a claim
                record["inspected"].append(
                    {"table": table, "columns": "ALL (generic full scan)"})
                for row in connection.execute(f'SELECT * FROM "{table}"'):
                    if _scan_text(" ".join(str(v) for v in row), needles):
                        record["outcome"] = "reference_found"
        finally:
            connection.close()
    except sqlite3.Error as exc:
        record["outcome"] = "unavailable"
        record["parse_error"] = str(exc)[:160]
    return record


def _inspect_file(path: Path, needles: tuple[str, ...]) -> dict:
    suffix = path.suffix.lower()
    record = {"path": str(path), "format": suffix or "none",
              "sha256": None, "outcome": "no_reference",
              "parse_error": None}
    try:
        record["sha256"] = _sha(path)
        if suffix == ".json":
            record["format"] = "json"
            json.loads(path.read_text(errors="strict"))
            if _scan_text(path.read_text(), needles):
                record["outcome"] = "reference_found"
        elif suffix == ".jsonl":
            record["format"] = "jsonl"
            for line in path.read_text().splitlines():
                if line.strip():
                    json.loads(line)
                if _scan_text(line, needles):
                    record["outcome"] = "reference_found"
        elif suffix in SQLITE_SUFFIXES:
            return _inspect_sqlite(path, needles)
        elif suffix in TEXT_SUFFIXES or suffix == "":
            record["format"] = "text"
            if _scan_text(path.read_text(errors="replace"), needles):
                record["outcome"] = "reference_found"
        elif (path.name.endswith(RAW_SCAN_SUFFIXES)
              or any(marker in path.name for marker in RAW_SCAN_MARKERS)):
            record["format"] = "binary_raw_scan"
            data = path.read_bytes()
            if any(n.encode() in data for n in needles):
                record["outcome"] = "reference_found"
        else:
            # genuinely unrecognized format: a raw byte scan still
            # runs, but the source counts as not-fully-understood
            data = path.read_bytes()
            if any(n.encode() in data for n in needles):
                record["outcome"] = "reference_found"
            else:
                record["outcome"] = "unavailable"
                record["parse_error"] = "unknown format; raw scan only"
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        record["outcome"] = "unavailable"
        record["parse_error"] = str(exc)[:160]
    return record


def _repo_consumer_search(needles: tuple[str, ...]) -> dict:
    """Code-level fact: which tracked executables reference the
    successor filename/schema, bound to the audited revision."""
    repo = Path(__file__).resolve().parent.parent
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unavailable"
    matches: list[str] = []
    for needle in needles:
        out = subprocess.run(
            ["git", "-C", str(repo), "grep", "-l", needle,
             "--", "app", "tools"],
            capture_output=True, text=True).stdout
        matches.extend(line for line in out.splitlines() if line)
    consumers = sorted({
        m for m in matches
        if not m.endswith("quarantine_inner_curriculum_successor.py")
        and not m.endswith("aggregate_eth_sac_inner_curriculum.py")})
    return {
        "revision": revision,
        "search_needles": list(needles),
        "producer_and_self_excluded": True,
        "executable_consumers": consumers,
        "no_executable_consumer": not consumers,
    }


def inspect_consumers(successor_sha: str,
                      roots: tuple[Path, ...] | None = None) -> dict:
    if roots is None:
        # resolved at call time so audit harnesses may monkeypatch
        # the module-level constant
        roots = LEDGER_CANDIDATES
    needles = (SUCCESSOR_NAME, successor_sha,
               "agent_multi.m0_successor_job.v1")
    findings = {
        "claimed": "unavailable",
        "code_level": _repo_consumer_search(
            (SUCCESSOR_NAME, "agent_multi.m0_successor_job.v1")),
        "sources": [],
        "roots_fully_scanned": [],
        "roots_missing": [],
    }
    any_reference = False
    any_unavailable = False
    for root in roots:
        if not root.exists():
            findings["roots_missing"].append(str(root))
            continue
        complete = True
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            record = _inspect_file(path, needles)
            findings["sources"].append(record)
            if record["outcome"] == "reference_found":
                any_reference = True
            elif record["outcome"] == "unavailable":
                any_unavailable = True
                complete = False
        if complete:
            findings["roots_fully_scanned"].append(str(root))
    if any_reference:
        findings["claimed"] = True
    elif any_unavailable or findings["roots_missing"]:
        findings["claimed"] = "unavailable"
    else:
        findings["claimed"] = False
    return findings


# ------------------------------------------------------------- repair


def _write_envelope(root: Path, target: Path, retired: Path | None,
                    consumers: dict) -> tuple[Path, bool]:
    bindings, problems = _envelope_bindings(root, target, retired)
    complete = not problems
    envelope = {
        "schema": ENVELOPE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason_findings": [REASON_FINDING, "AUD-F1-20260808-160"],
        "historical_evidence_immutable": (
            True if complete else "unavailable"),
        "complete": complete,
        "bindings": bindings,
        "consumer_inspection": consumers,
        "withdrawn_claim": (
            "mechanism_pass as an easy-vs-normal conclusion is"
            " WITHDRAWN; the preserved narrower observation is normal"
            " fine-tuning rate/duration evidence only"),
    }
    path = root / "m0_correction_envelope_v1.json"
    _atomic_write(path, envelope)
    return path, complete


def quarantine(root: Path,
               ledger_roots: tuple[Path, ...] | None = None) -> dict:
    queue = root / "queue"
    target = queue / SUCCESSOR_NAME
    queue.mkdir(parents=True, exist_ok=True)
    lock_path = queue / ".quarantine.lock"

    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        if not target.exists():
            return {"outcome": "REFUSED",
                    "reason": f"successor not found at {target}"}

        before_bytes = target.read_bytes()
        try:
            current = json.loads(before_bytes)
        except json.JSONDecodeError:
            current = {}

        if current.get("schema") != SUPERSESSION_SCHEMA:
            # first containment of a live successor
            original_sha = hashlib.sha256(before_bytes).hexdigest()
            retired_dir = queue / "retired" / original_sha
            retired_dir.mkdir(parents=True, exist_ok=True)
            retired = retired_dir / SUCCESSOR_NAME
            if not retired.exists():
                retired.write_bytes(before_bytes)
                with retired.open("rb") as handle:
                    os.fsync(handle.fileno())
                _fsync_dir(retired_dir)
            if _sha(retired) != original_sha:
                return {"outcome": "REFUSED",
                        "reason": "retired copy hash mismatch; aborting"}
            _atomic_write(target, {
                "schema": SUPERSESSION_SCHEMA,
                "launch_eligible": False,
                "supersedes_sha256": original_sha,
                "reason_finding": REASON_FINDING,
                "preserved_observation": PRESERVED_OBSERVATION,
                "superseded_at_utc": datetime.now(timezone.utc).isoformat(),
                "retired_path": str(retired),
            })
        else:
            supersedes = current.get("supersedes_sha256")
            retired = (queue / "retired" / supersedes / SUCCESSOR_NAME
                       if isinstance(supersedes, str)
                       and _SHA_RE.fullmatch(supersedes or "") else None)
            problems, _ = validate_quarantined_state(root)
            state_bytes_ok = target.read_bytes() == before_bytes
            if not problems and state_bytes_ok:
                return {
                    "outcome": "ALREADY_QUARANTINED",
                    "supersession_sha256": _sha(target),
                    "retired_original_sha256": (
                        _sha(retired) if retired and retired.is_file()
                        else None),
                    "bytes_changed": 0,
                }
            # corrupted or incomplete state: preserve it, then repair
            malformed_supersession = (
                current.get("launch_eligible") is not False
                or current.get("reason_finding") != REASON_FINDING
                or retired is None or not retired.is_file()
                or _sha(retired) != supersedes)
            if malformed_supersession:
                corrupt_sha = hashlib.sha256(before_bytes).hexdigest()
                corrupt_dir = queue / "retired" / f"corrupt-{corrupt_sha}"
                corrupt_dir.mkdir(parents=True, exist_ok=True)
                corrupt_copy = corrupt_dir / SUCCESSOR_NAME
                if not corrupt_copy.exists():
                    corrupt_copy.write_bytes(before_bytes)
                    _fsync_dir(corrupt_dir)
                if retired is None or not retired.is_file() or (
                        isinstance(supersedes, str)
                        and retired.is_file()
                        and _sha(retired) != supersedes):
                    # cannot prove original lineage; contain fail-closed
                    if retired is not None and retired.is_file():
                        supersedes = _sha(retired)
                    elif not (isinstance(supersedes, str)
                              and _SHA_RE.fullmatch(supersedes or "")):
                        supersedes = "unavailable"
                _atomic_write(target, {
                    "schema": SUPERSESSION_SCHEMA,
                    "launch_eligible": False,
                    "supersedes_sha256": supersedes,
                    "reason_finding": REASON_FINDING,
                    "preserved_observation": PRESERVED_OBSERVATION,
                    "superseded_at_utc": datetime.now(
                        timezone.utc).isoformat(),
                    "retired_path": (str(retired) if retired else None),
                    "repaired_from_corrupt_state": str(corrupt_copy),
                })

        # -------- envelope + consumer inventory (both code paths)
        current = json.loads(target.read_text())
        supersedes = current.get("supersedes_sha256")
        retired = (queue / "retired" / supersedes / SUCCESSOR_NAME
                   if isinstance(supersedes, str)
                   and _SHA_RE.fullmatch(supersedes) else None)
        consumers = inspect_consumers(
            supersedes if isinstance(supersedes, str) else "unavailable",
            roots=ledger_roots)
        envelope_path, envelope_complete = _write_envelope(
            root, target, retired, consumers)

        problems, facts = validate_quarantined_state(root)
        result = {
            "supersession_sha256": _sha(target),
            "retired_path": str(retired) if retired else None,
            "envelope_path": str(envelope_path),
            "envelope_sha256": _sha(envelope_path),
            "consumer_inspection_claimed": consumers["claimed"],
            "validation_problems": problems,
        }
        if not problems:
            result["outcome"] = "QUARANTINED"
        else:
            result["outcome"] = "QUARANTINED_EVIDENCE_INCOMPLETE"
            result["historical_evidence_immutable"] = "unavailable"
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = quarantine(args.root)
    print(json.dumps(result, indent=1, sort_keys=True))
    return {"QUARANTINED": 0, "ALREADY_QUARANTINED": 0,
            "QUARANTINED_EVIDENCE_INCOMPLETE": 3}.get(
        result["outcome"], 2)


if __name__ == "__main__":
    sys.exit(main())
