#!/usr/bin/env python3
"""T2-R18/R19 (order 2026-09-12): a read-only readjudication has its OWN
review record, and never repins history.

The historical MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD stays exactly what
it is: the statement of which code executed the campaign. Its commit and
its seven digests are never changed. A readjudication is a new,
read-only act with a record of its own,

    agent_multi.t2_readjudication_review_record.v1

at a fixed external pathname under the reviewer-authority root. That
record binds, at once: the historical execution record's digest, commit
and tree; the seven historical code digests; the clean commit and tree
of the ONE reproducer checkout; the exact readjudication surface with a
digest per file; the preserved root's identity and inventory; the
candidate adjudication digest; the decision, a canonical date, the scope
READ_ONLY_READJUDICATION, and retraining, downloads and grants_execution
all false.

This module is the candidate's side only. It can build a TEMPLATE, which
is visibly unreviewed and refuses to be written inside the authority
root, and it can VERIFY a record that already exists at the fixed path.
It never writes a record. Without one, `require_record` stops with
READJUDICATION_REVIEW_RECORD_REQUIRED before any evidence is opened.

The gate does NOT require HEAD to be the historical commit — that is the
executor's gate and it is not reused here. It requires the seven
historical digests to be present byte for byte in the reproducer
checkout AND the reproducer's own commit and surface to be the ones the
new record reviewed.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

SCHEMA = "agent_multi.t2_readjudication_review_record.v1"
RECORD_NAME = "MUSASHI_T2_READJUDICATION_REVIEW_RECORD.json"
REVIEWER = "General Musashi"
DECISION = "OPEN_T2_READ_ONLY_READJUDICATION"
SCOPE = "READ_ONLY_READJUDICATION"
TEMPLATE_REVIEWER = "UNREVIEWED_TEMPLATE"
TEMPLATE_DECISION = "PENDING_EXTERNAL_REVIEW"
#: an isolated review FIXTURE lets the mechanics run end to end before
#: a real record exists (R20). It never carries the reviewer's name, is
#: refused at the real authority root, and marks everything it touches.
FIXTURE_REVIEWER = "ISOLATED_TEST_FIXTURE"
FIXTURE_DECISION = "FIXTURE_READ_ONLY_READJUDICATION"

#: what the candidate adjudication digest covers: the scientific result
#: only. Code identity, timestamps and custody facts change with the
#: reproducer and are bound separately by the record.
SCIENTIFIC_KEYS = ("final_adjudication_counts", "screen_adjudication")

READJUDICATION_SURFACE = (
    "tools/t2_confirmatory.py",
    "tools/t2_confirmatory_executor.py",
    "tools/t2_assay_harness.py",
    "tools/t2_bank.py",
    "tools/t2_bank_census.py",
    "tools/t2_fresh_verifier.py",
    "tools/t2_public_data_census.py",
    "tools/t2_completion_reconstruction.py",
    "tools/t2_campaign_closure.py",
    "tools/descriptor_custody.py",
    "tools/t2_readjudication_gate.py",
)

KEYS = {
    "schema": str, "reviewer": str, "decision": str,
    "reviewed_at_date": str, "scope": str,
    "historical_execution_record_sha256": str,
    "historical_pinned_commit": str, "historical_pinned_tree": str,
    "historical_code_identity": dict,
    "reproducer_commit": str, "reproducer_tree": str,
    "readjudication_surface": dict, "readjudication_surface_sha256": str,
    "preserved_root_logical_id": str,
    "preserved_root_inventory_sha256": str,
    "candidate_adjudication_sha256": str,
    "retraining": bool, "downloads": bool, "grants_execution": bool,
}
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")


class ReadjudicationRefusal(SystemExit):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_obj(o) -> str:
    return sha_bytes(json.dumps(o, sort_keys=True,
                                separators=(",", ":")).encode())


def _git(repo: Path, *args) -> str:
    r = subprocess.run(("git", "-C", str(repo), *args),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise ReadjudicationRefusal("REPRODUCER_CHECKOUT_MISMATCH",
                                    f"git {' '.join(args)} failed")
    return r.stdout.strip()


def checkout_facts(checkout: Path) -> dict:
    checkout = Path(checkout)
    status = _git(checkout, "status", "--porcelain=v1",
                  "--untracked-files=all")
    return {"commit": _git(checkout, "rev-parse", "HEAD"),
            "tree": _git(checkout, "rev-parse", "HEAD^{tree}"),
            "clean": status == "",
            "dirty_paths": sorted(ln[3:] for ln in status.splitlines())[:20]}


def readjudication_surface(checkout: Path) -> dict:
    files, absent = {}, []
    for rel in READJUDICATION_SURFACE:
        p = Path(checkout) / rel
        if not p.is_file() or p.is_symlink():
            absent.append(rel)
            continue
        files[rel] = sha_bytes(p.read_bytes())
    if absent:
        raise ReadjudicationRefusal(
            "SURFACE_INCOMPLETE",
            f"the reproducer checkout lacks {absent}; an identity that "
            "skips a file is not one")
    return {"files": files, "surface_sha256": sha_obj(files)}


def preserved_root_identity(custody_module, root: Path) -> dict:
    """Photograph the preserved root and its units directory with the
    shared custody layer; the inventory digest covers every leaf's
    identity facts, which each later read must still match."""
    root = Path(root)
    c = custody_module.Custody(root)
    try:
        top = c.root_snapshot()
        entries = [("", n, top.leaf_inventory(n)) for n in top.files]
        if "units" in top.dirs:
            units = top.subdir("units")
            entries += [("units", n, units.leaf_inventory(n))
                        for n in units.files]
    finally:
        c.close()
    inv = sorted([d, n, {k: f[k] for k in ("type", "size", "mtime_ns",
                                           "inode", "device")}]
                 for d, n, f in entries)
    return {"logical_id": root.name,
            "inventory_sha256": sha_obj(inv),
            "leaves": len(inv)}


def scientific_adjudication_digest(closure: dict) -> str:
    screen = closure["screen_adjudication"]
    sign = closure.get("sign_test_supersession") or {}
    inv = closure["inventory"]
    body = {
        "final_adjudication_counts": closure["final_adjudication_counts"],
        "screen_adjudication": screen,
        "sign_test_corrected": {k: sign.get(k) for k in (
            "corrected_table_0_to_6", "corrected_value", "signs_positive")},
        "inventory": {k: inv.get(k) for k in ("exact", "sealed_units",
                                              "total_artifacts")},
    }
    return sha_obj(body)


def build_template(*, historical_record_raw: bytes, checkout: Path,
                   preserved: dict, candidate_adjudication_sha256: str,
                   ) -> dict:
    hist = json.loads(historical_record_raw.decode("utf-8"))
    facts = checkout_facts(checkout)
    surface = readjudication_surface(checkout)
    return {
        "schema": SCHEMA,
        "reviewer": TEMPLATE_REVIEWER,
        "decision": TEMPLATE_DECISION,
        "reviewed_at_date": "YYYY-MM-DD",
        "scope": SCOPE,
        "historical_execution_record_sha256":
            sha_bytes(historical_record_raw),
        "historical_pinned_commit": hist["pinned_commit"],
        "historical_pinned_tree": hist["pinned_tree"],
        "historical_code_identity": dict(sorted(
            hist["executor_code_identity"].items())),
        "reproducer_commit": facts["commit"],
        "reproducer_tree": facts["tree"],
        "readjudication_surface": surface["files"],
        "readjudication_surface_sha256": surface["surface_sha256"],
        "preserved_root_logical_id": preserved["logical_id"],
        "preserved_root_inventory_sha256": preserved["inventory_sha256"],
        "candidate_adjudication_sha256": candidate_adjudication_sha256,
        "retraining": False, "downloads": False, "grants_execution": False,
    }


def write_template(dest: Path, template: dict, authority_root: Path) -> Path:
    dest = Path(dest).expanduser().resolve()
    auth = Path(authority_root).expanduser().resolve()
    if dest == auth or auth in dest.parents:
        raise ReadjudicationRefusal(
            "CANDIDATE_MAY_NOT_WRITE_AUTHORITY",
            "a template is never written inside the reviewer-authority "
            "root; only the external reviewer installs a record there")
    if template.get("reviewer") != TEMPLATE_REVIEWER or \
            template.get("decision") != TEMPLATE_DECISION:
        raise ReadjudicationRefusal(
            "CANDIDATE_MAY_NOT_WRITE_AUTHORITY",
            "the candidate only ever writes an unreviewed template")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(template, indent=1, sort_keys=True) + "\n")
    return dest


def _read_private(conf, path: Path, missing: str) -> bytes:
    fd = conf._open_private_authority_file(path, missing_msg=missing)
    try:
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
    finally:
        os.close(fd)
    return b"".join(chunks)


def _no_dupes(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ReadjudicationRefusal("RECORD_SCHEMA", "duplicate JSON key")
    return dict(pairs)


def require_record(conf, *, authority_root: Path | None = None) -> Path:
    """Called FIRST, before any evidence is opened."""
    root = Path(authority_root) if authority_root else conf.AUTHORITY_ROOT
    path = root / RECORD_NAME
    if not os.path.lexists(path):
        raise ReadjudicationRefusal(
            "READJUDICATION_REVIEW_RECORD_REQUIRED",
            "no external readjudication review record exists at its "
            "fixed pathname; the readjudication stops before opening any "
            "evidence. The candidate can build a template; it cannot "
            "authorize itself")
    return path


def verify_record(conf, *, checkout: Path, preserved: dict,
                  authority_root: Path | None = None,
                  fixture: bool = False) -> dict:
    root = Path(authority_root) if authority_root else conf.AUTHORITY_ROOT
    real = Path(conf.AUTHORITY_ROOT).expanduser().resolve()
    if fixture and (root.expanduser().resolve() == real
                    or real in root.expanduser().resolve().parents):
        raise ReadjudicationRefusal(
            "FIXTURE_AT_AUTHORITY_ROOT",
            "an isolated review fixture is never read from the real "
            "reviewer-authority root")
    path = require_record(conf, authority_root=root)
    try:
        raw = _read_private(conf, path, "READJUDICATION_REVIEW_RECORD_"
                                        "REQUIRED")
    except ReadjudicationRefusal:
        raise
    except SystemExit as exc:
        raise ReadjudicationRefusal("RECORD_CUSTODY", str(exc))
    try:
        rec = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dupes)
    except ValueError as exc:
        raise ReadjudicationRefusal("RECORD_SCHEMA", f"not JSON ({exc})")
    if not isinstance(rec, dict) or set(rec) != set(KEYS):
        raise ReadjudicationRefusal(
            "RECORD_SCHEMA", "keys are not the exact v1 schema")
    for k, t in KEYS.items():
        if type(rec[k]) is not t:
            raise ReadjudicationRefusal(
                "RECORD_SCHEMA", f"{k} is {type(rec[k]).__name__}")
    if rec["schema"] != SCHEMA:
        raise ReadjudicationRefusal("RECORD_SCHEMA", "foreign schema")
    want = ((FIXTURE_REVIEWER, FIXTURE_DECISION) if fixture
            else (REVIEWER, DECISION))
    if (rec["reviewer"], rec["decision"]) != want:
        raise ReadjudicationRefusal(
            "RECORD_NOT_EXTERNALLY_REVIEWED" if not fixture
            else "FIXTURE_NOT_MARKED_AS_FIXTURE",
            "the record's reviewer and decision are not the ones this "
            "mode accepts: a template, a fixture outside fixture mode, "
            "or a reviewer's name inside a fixture are all refused")
    try:
        d = _dt.date.fromisoformat(rec["reviewed_at_date"])
    except ValueError:
        raise ReadjudicationRefusal("RECORD_SCHEMA", "date not ISO")
    if d.isoformat() != rec["reviewed_at_date"]:
        raise ReadjudicationRefusal("RECORD_SCHEMA", "date not canonical")
    if rec["scope"] != SCOPE or rec["retraining"] or rec["downloads"] \
            or rec["grants_execution"]:
        raise ReadjudicationRefusal(
            "SCOPE_OR_GRANT_VIOLATION",
            "a readjudication record is read-only: no retraining, no "
            "downloads, no execution grant")
    for k in ("historical_execution_record_sha256",
              "readjudication_surface_sha256",
              "preserved_root_inventory_sha256",
              "candidate_adjudication_sha256"):
        if not HEX64.fullmatch(rec[k]):
            raise ReadjudicationRefusal("RECORD_SCHEMA", f"{k} not hex64")
    for k in ("historical_pinned_commit", "historical_pinned_tree",
              "reproducer_commit", "reproducer_tree"):
        if not HEX40.fullmatch(rec[k]):
            raise ReadjudicationRefusal("RECORD_SCHEMA", f"{k} not hex40")

    hist_raw = _read_private(conf, root / Path(
        conf.T2_SUCCESSOR_EXECUTION_RECORD_PATH).name,
        "T2_SUCCESSOR_EXECUTION_RECORD_REQUIRED")
    hist = json.loads(hist_raw.decode("utf-8"), object_pairs_hook=_no_dupes)
    if sha_bytes(hist_raw) != rec["historical_execution_record_sha256"] or \
            hist["pinned_commit"] != rec["historical_pinned_commit"] or \
            hist["pinned_tree"] != rec["historical_pinned_tree"] or \
            hist["executor_code_identity"] != rec["historical_code_identity"]:
        raise ReadjudicationRefusal(
            "HISTORICAL_RECORD_MISMATCH",
            "the historical execution record on disk is not the one the "
            "readjudication record reviewed — it was changed or repinned")
    physical = {rel: sha_bytes((Path(checkout) / rel).read_bytes())
                if (Path(checkout) / rel).is_file() else "ABSENT"
                for rel in hist["executor_code_identity"]}
    if physical != hist["executor_code_identity"]:
        raise ReadjudicationRefusal(
            "HISTORICAL_CODE_MISMATCH",
            f"the reproducer does not carry the historical bytes for "
            f"{sorted(k for k in physical if physical[k] != hist['executor_code_identity'][k])}")
    facts = checkout_facts(checkout)
    if not facts["clean"]:
        raise ReadjudicationRefusal("REPRODUCER_CHECKOUT_MISMATCH",
                                    f"dirty: {facts['dirty_paths']}")
    if (facts["commit"], facts["tree"]) != (rec["reproducer_commit"],
                                            rec["reproducer_tree"]):
        raise ReadjudicationRefusal(
            "REPRODUCER_CHECKOUT_MISMATCH",
            "the executing checkout is not the reproducer the record "
            "reviewed")
    surface = readjudication_surface(checkout)
    if surface["files"] != rec["readjudication_surface"] or \
            surface["surface_sha256"] != rec["readjudication_surface_sha256"]:
        raise ReadjudicationRefusal("SURFACE_MISMATCH",
                                    "the readjudication surface differs")
    if preserved["logical_id"] != rec["preserved_root_logical_id"] or \
            preserved["inventory_sha256"] != \
            rec["preserved_root_inventory_sha256"]:
        raise ReadjudicationRefusal(
            "PRESERVED_ROOT_MISMATCH",
            "the root about to be read is not the one the record reviewed")
    return {"record_sha256": sha_bytes(raw),
            "record_kind": ("ISOLATED_FIXTURE_NOT_AN_EXTERNAL_RECORD"
                            if fixture else "EXTERNAL_REVIEW_RECORD"),
            "candidate_adjudication_sha256":
                rec["candidate_adjudication_sha256"],
            "reproducer": facts, "surface_sha256": surface["surface_sha256"],
            "historical_execution_record_sha256":
                rec["historical_execution_record_sha256"],
            "scope": SCOPE}


def assert_candidate_matches(verified: dict, adjudication_sha256: str) -> None:
    if adjudication_sha256 != verified["candidate_adjudication_sha256"]:
        raise ReadjudicationRefusal(
            "CANDIDATE_ADJUDICATION_DIVERGES",
            "the recomputed adjudication differs from the candidate the "
            "record reviewed; the readjudication stops and is explained, "
            "never normalized")
