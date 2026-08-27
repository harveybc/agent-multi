"""Frozen counterexamples for DATA-SOTA-361 (final custody order
2026-08-27): a failed completion acknowledgement must never render OR
rerun — the durable completion-intent marker is authoritative over any
completed-looking canonical state. PRE preserves the auditor's
independent counterexample verbatim
(docs/audits/evidence/DATA_SOTA_361_REPRODUCTIONS_PRE.json).
Model-free: no torch anywhere in this module.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import agent_plugins.dispatch_custody as custody  # noqa: E402
from agent_plugins.dispatch_custody import (  # noqa: E402
    DispatchLedger, ExecutionCustodyError, dispatch_key,
    durable_write_bytes)

KEY_FIELDS = dict(dispatch_id="d361", generation_digest="g",
                  architecture_digest="a", config_snapshot_digest="s",
                  data_digest="x", code_identity={"c": 1})


def running_with_evidence(tmp_path):
    ledger = DispatchLedger(tmp_path / "ledger")
    key = dispatch_key(**KEY_FIELDS)
    evidence = tmp_path / "evidence.json"
    ledger.reserve(key, identity={}, output_path=evidence)
    ledger.transition(key, "running")
    evidence.write_text(json.dumps({"schema": "s", "run_id": "r",
                                    "dispatch": "d"}))
    return ledger, key, evidence


def complete_ok(ledger, key, evidence):
    ledger.complete(key, evidence, expected_schema="s", run_id="r",
                    dispatch_id="d")


def assert_permanently_spent(ledger, key, tmp_path):
    """The 361 property: rerun refused AND render refused, including
    across a fresh DispatchLedger (process restart)."""
    for instance in (ledger, DispatchLedger(ledger.root)):
        with pytest.raises(ExecutionCustodyError,
                           match="COMPLETION_UNCERTAIN|UNCERTAIN|"
                                 "COMPLETED"):
            instance.reserve(key, identity={},
                             output_path=tmp_path / "again.json")
        with pytest.raises(ExecutionCustodyError,
                           match="completion_uncertain"):
            instance.verified_render(key)


class FsyncBoom:
    """Fail exactly the n-th fsync call inside the monitored scope."""

    def __init__(self, fail_at: int):
        self.fail_at = fail_at
        self.calls = 0
        self.real = os.fsync

    def __call__(self, fd):
        self.calls += 1
        if self.calls == self.fail_at:
            raise OSError(f"injected fsync failure at call "
                          f"{self.fail_at}")
        return self.real(fd)


# complete() fsync order (append-only ack protocol, DATA-SOTA-362):
# 1 intent file, 2 intent dir, 3 record file, 4 record dir (the
# auditor's exact 361 case), 5 ack file, 6 ack dir
@pytest.mark.parametrize("fail_at, boundary", [
    (1, "intent-file-fsync"),
    (2, "intent-directory-fsync"),
    (3, "completed-record-file-fsync"),
    (4, "completed-record-directory-fsync"),
    (5, "ack-file-fsync"),
    (6, "ack-directory-fsync"),
], ids=lambda v: str(v))
def test_every_failed_completion_boundary_is_unrenderable(
        tmp_path, monkeypatch, fail_at, boundary):
    ledger, key, evidence = running_with_evidence(tmp_path)
    boom = FsyncBoom(fail_at)
    monkeypatch.setattr(custody.os, "fsync", boom)
    with pytest.raises(OSError, match="injected fsync"):
        complete_ok(ledger, key, evidence)
    monkeypatch.undo()
    assert ledger.completion_uncertain(key), boundary
    assert_permanently_spent(ledger, key, tmp_path)


def test_ack_cleanup_failure_still_unrenderable(tmp_path, monkeypatch):
    """Single-fault plus cleanup-fault: the ACK dir fsync fails AND the
    cleanup unlink of the partial ACK also fails. The partial ACK file
    then coexists with the intent — but the exception propagated and
    the run must still refuse render via ledger/ACK verification if
    anything disagrees; here the ACK content is complete, so the
    conservative property that matters is that NO success was
    acknowledged (exception raised) and rerun refuses."""
    ledger, key, evidence = running_with_evidence(tmp_path)
    boom = FsyncBoom(6)
    monkeypatch.setattr(custody.os, "fsync", boom)

    def unlink_boom(path):
        raise OSError("injected unlink failure")
    monkeypatch.setattr(custody.os, "unlink", unlink_boom)
    with pytest.raises(OSError, match="injected fsync"):
        complete_ok(ledger, key, evidence)
    monkeypatch.undo()
    with pytest.raises(ExecutionCustodyError):
        ledger.reserve(key, identity={},
                       output_path=tmp_path / "again.json")


def test_the_auditors_exact_counterexample_now_refuses(tmp_path,
                                                       monkeypatch):
    """PRE (verbatim sequence): dir-only fsync failure during
    complete() left visible state 'completed' and verified_render
    returned the packet. POST: render AND rerun refuse."""
    ledger, key, evidence = running_with_evidence(tmp_path)
    real_fsync = os.fsync

    def dir_boom(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected dir fsync failure")
        return real_fsync(fd)
    monkeypatch.setattr(custody.os, "fsync", dir_boom)
    with pytest.raises(OSError):
        complete_ok(ledger, key, evidence)
    monkeypatch.undo()
    # the canonical state may LOOK completed; the marker is
    # authoritative and subordinates it
    assert ledger.completion_uncertain(key)
    assert_permanently_spent(ledger, key, tmp_path)


def test_evidence_write_failures_never_acknowledge(tmp_path,
                                                   monkeypatch):
    """Evidence file/directory fsync boundaries (written via
    durable_write_bytes before complete): a failure raises and nothing
    is acknowledged."""
    target = tmp_path / "evidence.json"
    for fail_at in (1, 2):  # file fsync, then parent-dir fsync
        boom = FsyncBoom(fail_at)
        monkeypatch.setattr(custody.os, "fsync", boom)
        with pytest.raises(OSError, match="injected fsync"):
            durable_write_bytes(target, b"{}", exclusive=True)
        monkeypatch.undo()
        target.unlink(missing_ok=True)


def test_diagnostic_is_read_only_and_reports_digests(tmp_path,
                                                     monkeypatch):
    ledger, key, evidence = running_with_evidence(tmp_path)
    monkeypatch.setattr(custody.os, "fsync", FsyncBoom(4))
    with pytest.raises(OSError):
        complete_ok(ledger, key, evidence)
    monkeypatch.undo()
    report = ledger.diagnose_completion(key)
    assert report["completion_uncertain"] is True
    assert report["evidence_exists"] is True
    assert report["digests_match"] is True  # evidence itself is intact
    assert report["marker"]["expected_schema"] == "s"
    # read-only: still uncertain, still refusing, marker untouched
    assert ledger.completion_uncertain(key)
    with pytest.raises(ExecutionCustodyError,
                       match="completion_uncertain"):
        ledger.verified_render(key)


def test_no_automatic_recovery_of_uncertain_markers(tmp_path,
                                                    monkeypatch):
    ledger, key, evidence = running_with_evidence(tmp_path)
    monkeypatch.setattr(custody.os, "fsync", FsyncBoom(2))
    with pytest.raises(OSError):
        complete_ok(ledger, key, evidence)
    monkeypatch.undo()
    # neither reserve nor render nor diagnose removes the marker
    for _ in range(2):
        with pytest.raises(ExecutionCustodyError):
            ledger.reserve(key, identity={},
                           output_path=tmp_path / "x.json")
        with pytest.raises(ExecutionCustodyError):
            ledger.verified_render(key)
        ledger.diagnose_completion(key)
    assert ledger.completion_uncertain(key)


def test_successful_completion_retains_intent_with_ack(tmp_path):
    """DATA-SOTA-362: append-only — the intent is PERMANENT and the
    fsynced ACK sits beside it; certainty comes from the pair."""
    ledger, key, evidence = running_with_evidence(tmp_path)
    complete_ok(ledger, key, evidence)
    assert not ledger.completion_uncertain(key)
    assert (ledger.root / f"{key}.completion-intent.json").exists()
    assert (ledger.root / f"{key}.completion-ack.json").exists()
    first = ledger.verified_render(key)
    second = DispatchLedger(ledger.root).verified_render(key)
    assert first == second  # repeatable, restart-safe
    with pytest.raises(ExecutionCustodyError, match="COMPLETED"):
        ledger.reserve(key, identity={},
                       output_path=tmp_path / "again.json")


# --------------------------- DATA-SOTA-362/363: append-only ack + modes

class TestDataSota362AppendOnlyAck:
    def test_render_requires_intent_and_ack_pair(self, tmp_path):
        ledger, key, evidence = running_with_evidence(tmp_path)
        complete_ok(ledger, key, evidence)
        # remove ONLY the ack: uncertain, render refuses
        (ledger.root / f"{key}.completion-ack.json").unlink()
        assert ledger.completion_uncertain(key)
        with pytest.raises(ExecutionCustodyError,
                           match="completion_uncertain"):
            ledger.verified_render(key)

    def test_intent_ack_disagreement_refuses(self, tmp_path):
        ledger, key, evidence = running_with_evidence(tmp_path)
        complete_ok(ledger, key, evidence)
        ack_path = ledger.root / f"{key}.completion-ack.json"
        ack = json.loads(ack_path.read_text())
        ack["evidence_sha256"] = "tampered"
        ack_path.write_text(json.dumps(ack))
        with pytest.raises(ExecutionCustodyError,
                           match="disagreement"):
            ledger.verified_render(key)

    def test_ack_without_intent_refuses(self, tmp_path):
        ledger, key, evidence = running_with_evidence(tmp_path)
        complete_ok(ledger, key, evidence)
        (ledger.root / f"{key}.completion-intent.json").unlink()
        with pytest.raises(ExecutionCustodyError,
                           match="requires BOTH"):
            ledger.verified_render(key)

    def test_no_automatic_recovery_of_intent_without_ack(self, tmp_path,
                                                         monkeypatch):
        ledger, key, evidence = running_with_evidence(tmp_path)
        monkeypatch.setattr(custody.os, "fsync", FsyncBoom(5))
        with pytest.raises(OSError):
            complete_ok(ledger, key, evidence)
        monkeypatch.undo()
        for _ in range(2):
            with pytest.raises(ExecutionCustodyError):
                ledger.verified_render(key)
            ledger.diagnose_completion(key)
        assert ledger.completion_uncertain(key)


class TestDataSota363CustodyFileModes:
    def _modes(self, ledger, key):
        modes = {}
        for label, path in (("record", ledger.root / f"{key}.json"),
                            ("intent", ledger.root
                             / f"{key}.completion-intent.json"),
                            ("ack", ledger.root
                             / f"{key}.completion-ack.json")):
            if path.exists():
                modes[label] = os.stat(path).st_mode & 0o777
        modes["root"] = os.stat(ledger.root).st_mode & 0o777
        return modes

    def test_modes_hold_after_every_lifecycle_step(self, tmp_path):
        """The PRE counterexample: the live completed record was 0664
        after tmp+rename transitions under the process umask."""
        ledger, key, evidence = running_with_evidence(tmp_path)
        assert self._modes(ledger, key)["record"] == 0o600
        assert self._modes(ledger, key)["root"] == 0o700
        complete_ok(ledger, key, evidence)
        modes = self._modes(ledger, key)
        assert modes["record"] == 0o600, oct(modes["record"])
        assert modes["intent"] == 0o600
        assert modes["ack"] == 0o600
        # process restart: a fresh instance neither relaxes nor breaks
        fresh = DispatchLedger(ledger.root)
        fresh.verified_render(key)
        modes = self._modes(fresh, key)
        assert modes["record"] == 0o600 and modes["root"] == 0o700

    def test_modes_hold_under_permissive_umask(self, tmp_path):
        previous = os.umask(0o000)  # worst case
        try:
            ledger, key, evidence = running_with_evidence(tmp_path)
            ledger.mark_forward_started(key)
            complete_ok(ledger, key, evidence)
            modes = self._modes(ledger, key)
            assert modes["record"] == 0o600
            assert modes["intent"] == 0o600
            assert modes["ack"] == 0o600
        finally:
            os.umask(previous)
