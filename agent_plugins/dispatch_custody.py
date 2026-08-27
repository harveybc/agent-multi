"""Durable single-use execution custody v2 (DATA-SOTA-358/359/360).

A dispatched one-shot execution must be provably single-use AND its
evidence provably authentic:

* the run is RESERVED atomically in a durable ledger BEFORE the model
  is constructed; the dispatch key binds the IMMUTABLE config-snapshot
  digest (DATA-SOTA-359) along with generation, architecture, data and
  code identities;
* the state machine is ENFORCED — only legal transitions commit, every
  ledger write fsyncs the record file AND its parent directory, the
  transition sequence is persisted monotonically, retirement is
  no-clobber, the ledger root is mode 0700 with 0600 records and
  symlink roots/records refuse (DATA-SOTA-360);
* completion happens ONLY after the evidence file is durably written
  (file + parent-directory fsync) and its SHA-256, schema, run id and
  dispatch id are bound into the completed record; if the completion
  write fails the run is SPENT, never rerunnable;
* the renderer takes a LEDGER KEY, loads only the evidence named by a
  completed record, and verifies digest, schema, run id, dispatch id
  and config/architecture identities before presenting — model-free
  and freely repeatable.

Legal transitions::

    absent   -> reserved
    reserved -> running | failed_before_forward
    running  -> completed | interrupted | spent
             |  failed_before_forward   (ONLY while the durable
                                         forward_started flag is False)
    completed / interrupted / spent -> (terminal; no transition, no retry)
    failed_before_forward -> reserved   (retry via reserve, retired
                                         durably without clobbering)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agent_plugins.branch_pretraining import sha256_obj

DEFAULT_LEDGER_ROOT = (Path.home()
                       / ".local/state/agent-multi/dispatch_ledger")
RETRYABLE_STATES = ("failed_before_forward",)
TERMINAL_STATES = ("completed", "interrupted", "spent",
                   "DISCLOSED_PROTOCOL_DEVIATION")
LEGAL_TRANSITIONS = {
    "reserved": {"running", "failed_before_forward"},
    "running": {"completed", "interrupted", "spent",
                "failed_before_forward"},
}


class ExecutionCustodyError(RuntimeError):
    """Typed refusal: the dispatch identity is already spent, a
    transition is illegal, the output path is unsafe, or the evidence
    cannot be authenticated."""


def dispatch_key(*, dispatch_id: str, generation_digest: str,
                 architecture_digest: str, config_snapshot_digest: str,
                 data_digest: str, code_identity: dict[str, Any]) -> str:
    return sha256_obj({"dispatch_id": dispatch_id,
                       "generation_digest": generation_digest,
                       "architecture_digest": architecture_digest,
                       "config_snapshot_digest": config_snapshot_digest,
                       "data_digest": data_digest,
                       "code_identity": code_identity})


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_write_bytes(path: Path, payload: bytes, *,
                        exclusive: bool) -> None:
    """DATA-SOTA-360: every acknowledged write is durable — file fsync
    AND parent-directory fsync. ``exclusive`` uses O_EXCL (create-only,
    no clobber); otherwise an atomic tmp+rename replace."""
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ExecutionCustodyError(
            f"write target or its parent is a symlink — refused: "
            f"{path.name}")
    if exclusive:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                         0o600)
        except FileExistsError as exc:
            raise ExecutionCustodyError(
                f"exclusive write target already exists (no-clobber): "
                f"{path.name}") from exc
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
    else:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    _fsync_directory(path.parent)


class DispatchLedger:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else DEFAULT_LEDGER_ROOT
        if self.root.is_symlink():
            raise ExecutionCustodyError(
                "ledger root is a symlink — refused")
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _record_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def _marker_path(self, key: str) -> Path:
        return self.root / f"{key}.completion-intent.json"

    def completion_uncertain(self, key: str) -> bool:
        """DATA-SOTA-361: a lingering completion-intent marker means a
        completion acknowledgement FAILED at some durability boundary.
        The marker is authoritative over any completed-looking
        canonical state; only an independent recovery tool may resolve
        it — never the ordinary execution path."""
        return self._marker_path(key).exists()

    def read(self, key: str) -> dict[str, Any] | None:
        path = self._record_path(key)
        if path.is_symlink():
            raise ExecutionCustodyError(
                "ledger record is a symlink — refused")
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _commit(self, key: str, record: dict[str, Any]) -> None:
        record["transition_sequence"] = record.get(
            "transition_sequence", 0) + 1
        durable_write_bytes(self._record_path(key),
                            json.dumps(record, indent=1).encode(),
                            exclusive=False)

    def reserve(self, key: str, *, identity: dict[str, Any],
                output_path: Path) -> None:
        """absent/failed_before_forward -> reserved. Atomic (O_EXCL)
        after durable no-clobber retirement of a retryable prior
        attempt; concurrent reservation has exactly one winner."""
        output_path = Path(output_path)
        if output_path.is_symlink() or output_path.parent.is_symlink():
            raise ExecutionCustodyError(
                "output path or its parent is a symlink — refused")
        if output_path.exists():
            raise ExecutionCustodyError(
                f"output path already exists (no-clobber): "
                f"{output_path.name}")
        if self.completion_uncertain(key):
            raise ExecutionCustodyError(
                "dispatch identity is COMPLETION_UNCERTAIN — a prior "
                "completion acknowledgement failed; permanently spent "
                "for the ordinary path (DATA-SOTA-361)")
        existing = self.read(key)
        attempt = 1
        if existing is not None:
            state = existing.get("state")
            if state == "completed":
                raise ExecutionCustodyError(
                    f"dispatch identity already COMPLETED — a second "
                    f"execution refuses (DATA-SOTA-358)")
            if state not in RETRYABLE_STATES:
                raise ExecutionCustodyError(
                    f"dispatch identity in UNCERTAIN or terminal state "
                    f"{state!r} — treated as spent; refusing "
                    f"(DATA-SOTA-358)")
            retired = self._record_path(
                f"{key}.retired-{existing.get('attempt', 0)}")
            if retired.exists():
                raise ExecutionCustodyError(
                    f"retirement target already exists (no-clobber): "
                    f"{retired.name}")
            os.replace(self._record_path(key), retired)
            _fsync_directory(self.root)
            attempt = int(existing.get("attempt", 0)) + 1
        record = {"schema": "agent_multi.dispatch_ledger.v2",
                  "key": key, "state": "reserved",
                  "attempt": attempt, "transition_sequence": 1,
                  "forward_started": False,
                  "identity": identity,
                  "output_path": f"external:{output_path.name}"}
        durable_write_bytes(self._record_path(key),
                            json.dumps(record, indent=1).encode(),
                            exclusive=True)

    def transition(self, key: str, state: str,
                   extra: dict[str, Any] | None = None) -> None:
        """ENFORCED state machine (DATA-SOTA-360): only legal
        transitions commit; terminal states never move again."""
        record = self.read(key)
        if record is None:
            raise ExecutionCustodyError(
                f"no ledger record for {key[:12]}")
        current = record.get("state")
        if current in TERMINAL_STATES:
            raise ExecutionCustodyError(
                f"illegal transition: {current!r} is terminal "
                f"(DATA-SOTA-360)")
        allowed = LEGAL_TRANSITIONS.get(current, set())
        if state not in allowed:
            raise ExecutionCustodyError(
                f"illegal transition {current!r} -> {state!r} "
                f"(DATA-SOTA-360)")
        if (current == "running" and state == "failed_before_forward"
                and record.get("forward_started")):
            raise ExecutionCustodyError(
                "illegal transition: forward_started is durably True — "
                "a post-forward failure is SPENT, not retryable "
                "(DATA-SOTA-360)")
        record["state"] = state
        if extra:
            record.update(extra)
        self._commit(key, record)

    def mark_forward_started(self, key: str) -> None:
        """Durably flip the forward flag BEFORE the forward executes."""
        record = self.read(key)
        if record is None or record.get("state") != "running":
            raise ExecutionCustodyError(
                "forward flag requires a running reservation")
        record["forward_started"] = True
        self._commit(key, record)

    def complete(self, key: str, evidence_path: Path,
                 *, expected_schema: str, run_id: str,
                 dispatch_id: str) -> None:
        """DATA-SOTA-360/361: completion ONLY after durable evidence,
        under a durable completion-intent sidecar. Sequence:

        1. create + fsync (file AND directory) the no-clobber intent
           marker carrying the expected evidence identity;
        2. commit the completed ledger record durably;
        3. unlink the marker, then fsync the directory again;
        4. only then acknowledge success.

        ANY failure leaves the marker: the dispatch is permanently
        COMPLETION_UNCERTAIN — neither rerunnable nor renderable by the
        ordinary path, whatever the canonical state looks like."""
        evidence_path = Path(evidence_path)
        if not evidence_path.is_file():
            raise ExecutionCustodyError(
                "evidence absent at completion time")
        digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        marker = {"schema": "agent_multi.completion_intent.v1",
                  "key": key,
                  "expected_evidence_path": str(evidence_path),
                  "expected_evidence_sha256": digest,
                  "expected_schema": expected_schema,
                  "run_id": run_id, "dispatch_id": dispatch_id}
        durable_write_bytes(self._marker_path(key),
                            json.dumps(marker, indent=1).encode(),
                            exclusive=True)
        self.transition(key, "completed", {
            "evidence": f"external:{evidence_path.name}",
            "evidence_path_local": str(evidence_path),
            "evidence_sha256": digest,
            "evidence_schema": expected_schema,
            "run_id": run_id,
            "dispatch_id": dispatch_id})
        try:
            os.unlink(self._marker_path(key))
            _fsync_directory(self.root)
        except Exception:
            # DATA-SOTA-361: the completion record is durable, but the
            # marker removal is NOT — restore the marker (best effort)
            # so the dispatch stays completion_uncertain in-process,
            # matching what a crash-recovery would observe.
            try:
                if not self._marker_path(key).exists():
                    with open(self._marker_path(key), "x",
                              encoding="utf-8") as fh:
                        fh.write(json.dumps(marker, indent=1))
            except Exception:
                pass
            raise

    def diagnose_completion(self, key: str) -> dict[str, Any]:
        """Read-only diagnostic for a COMPLETION_UNCERTAIN dispatch:
        reports expected vs actual digests and states, mutates NOTHING.
        Resolution is outside the ordinary path (independent recovery
        tooling under separate authority)."""
        marker_path = self._marker_path(key)
        report: dict[str, Any] = {
            "key": key,
            "completion_uncertain": marker_path.exists(),
            "canonical_state": (self.read(key) or {}).get("state"),
        }
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            report["marker"] = marker
            evidence = Path(marker.get("expected_evidence_path") or "")
            report["evidence_exists"] = evidence.is_file()
            if evidence.is_file():
                actual = hashlib.sha256(
                    evidence.read_bytes()).hexdigest()
                report["actual_evidence_sha256"] = actual
                report["digests_match"] = (
                    actual == marker.get("expected_evidence_sha256"))
        return report

    def verified_render(self, key: str) -> dict[str, Any]:
        """DATA-SOTA-360: model-free presentation from the LEDGER KEY
        only. Verifies completed state, evidence existence, digest,
        schema, run id, dispatch id and bound identities before
        returning the packet. Freely repeatable."""
        if self.completion_uncertain(key):
            raise ExecutionCustodyError(
                "render refused: completion_uncertain — a completion "
                "acknowledgement failed at a durability boundary; the "
                "intent marker is authoritative over any "
                "completed-looking canonical state (DATA-SOTA-361)")
        record = self.read(key)
        if record is None:
            raise ExecutionCustodyError(
                "render refused: no ledger record for this key")
        if record.get("state") != "completed":
            raise ExecutionCustodyError(
                f"render refused: state {record.get('state')!r} is not "
                f"completed")
        evidence_path = Path(record.get("evidence_path_local") or "")
        if not evidence_path.is_file():
            raise ExecutionCustodyError(
                "render refused: evidence MISSING despite a completed "
                "record (DATA-SOTA-360)")
        raw = evidence_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record.get("evidence_sha256"):
            raise ExecutionCustodyError(
                "render refused: evidence digest mismatch — "
                "substituted or corrupted packet (DATA-SOTA-360)")
        packet = json.loads(raw.decode("utf-8"))
        checks = (
            ("schema", record.get("evidence_schema")),
            ("run_id", record.get("run_id")),
            ("dispatch", record.get("dispatch_id")))
        for field, expected in checks:
            if packet.get(field) != expected:
                raise ExecutionCustodyError(
                    f"render refused: packet {field} "
                    f"{packet.get(field)!r} differs from the ledger's "
                    f"{expected!r}")
        identity = record.get("identity") or {}
        for field in ("architecture_digest", "config_snapshot_digest"):
            if field in identity and packet.get(field) is not None \
                    and packet.get(field) != identity[field]:
                raise ExecutionCustodyError(
                    f"render refused: packet {field} differs from the "
                    f"reserved identity")
        return packet

    def record_protocol_deviation(self, key: str,
                                  facts: dict[str, Any]) -> None:
        """Preserve a historical deviation with ONLY the facts actually
        known — never invented metrics."""
        record = {"schema": "agent_multi.dispatch_ledger.v2",
                  "key": key, "state": "DISCLOSED_PROTOCOL_DEVIATION",
                  "transition_sequence": 1,
                  "facts": facts}
        path = self._record_path(key)
        if path.exists():
            raise ExecutionCustodyError(
                "deviation record already present; never overwrite "
                "history")
        durable_write_bytes(path, json.dumps(record, indent=1).encode(),
                            exclusive=True)
