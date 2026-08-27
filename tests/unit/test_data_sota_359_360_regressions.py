"""Frozen counterexamples for DATA-SOTA-359/360 (custody order
2026-08-27). PRE reproductions:
docs/audits/evidence/DATA_SOTA_359_360_REPRODUCTIONS_PRE.json.
NO model is constructed and NO forward runs anywhere in this module
(order boundary: CPU tests only).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.dispatch_custody import (  # noqa: E402
    DispatchLedger, ExecutionCustodyError, dispatch_key)
from agent_plugins.grouped_architecture import (  # noqa: E402
    ArchitectureError, snapshot_effective_config)

STRONG_CONFIG = (REPO / "examples/config/"
                 "project3_ethusdt_4h_sac_grouped_strong_v1.json")


# --------------------------------------- 359: immutable config snapshot

class TestDataSota359ImmutableSnapshot:
    def test_snapshot_derives_everything_from_one_read(self, tmp_path):
        copy = tmp_path / "config.json"
        copy.write_text(STRONG_CONFIG.read_text())
        snapshot = snapshot_effective_config(copy)
        original_digest = snapshot["config_sha256"]
        # mutate the path AFTER the snapshot: execution inputs must not
        # change — the snapshot is the identity, not the mutable path
        mutated = json.loads(copy.read_text())
        mutated["max_steps"] = 999999
        mutated["feature_extractor_config"]["fusion"]["params"][
            "output_dim"] = 128
        copy.write_text(json.dumps(mutated))
        assert snapshot["config_sha256"] == original_digest
        assert snapshot["env_config"].get("max_steps") != 999999
        assert snapshot["materialized"]["expected_output_dim"] == 96
        # a fresh read sees the mutation — proving the gap the snapshot
        # closes
        assert snapshot_effective_config(copy)["config_sha256"] != \
            original_digest

    def test_mutation_before_snapshot_binds_the_new_bytes(self, tmp_path):
        copy = tmp_path / "config.json"
        copy.write_text(STRONG_CONFIG.read_text())
        mutated = json.loads(copy.read_text())
        mutated["max_steps"] = 777
        copy.write_text(json.dumps(mutated))
        snapshot = snapshot_effective_config(copy)
        assert snapshot["env_config"]["max_steps"] == 777

    def test_symlink_config_path_refuses(self, tmp_path):
        real = tmp_path / "real.json"
        real.write_text(STRONG_CONFIG.read_text())
        link = tmp_path / "link.json"
        link.symlink_to(real)
        with pytest.raises(ArchitectureError, match="symlink"):
            snapshot_effective_config(link)

    def test_snapshot_digest_is_of_the_exact_bytes(self, tmp_path):
        import hashlib
        copy = tmp_path / "config.json"
        copy.write_text(STRONG_CONFIG.read_text())
        snapshot = snapshot_effective_config(copy)
        assert snapshot["config_sha256"] == hashlib.sha256(
            copy.read_bytes()).hexdigest()

    def test_tool_has_no_post_reservation_config_reads(self):
        """The PRE counterexample: three separate reads of the config
        path (TOCTOU). v3 derives the env config from the snapshot."""
        source = (REPO / "tools/load_pretrained_branches_smoke.py"
                  ).read_text()
        assert "materialize_from_file" not in source
        assert "json.loads(arch_config_path" not in source
        assert 'snapshot["env_config"]' in source
        # the dispatch key binds the snapshot digest
        assert "config_snapshot_digest=snapshot" in source.replace(
            "\n", "").replace(" ", "")


# ------------------------------------ 360: enforced durable state machine

KEY_FIELDS = dict(dispatch_id="d360", generation_digest="g",
                  architecture_digest="a", config_snapshot_digest="s",
                  data_digest="x", code_identity={"c": 1})


def make_ledger(tmp_path):
    return DispatchLedger(tmp_path / "ledger"), dispatch_key(**KEY_FIELDS)


class TestDataSota360StateMachine:
    def test_completed_is_terminal(self, tmp_path):
        """The PRE counterexample: completed -> running was silently
        accepted."""
        ledger, key = make_ledger(tmp_path)
        evidence = tmp_path / "e.json"
        ledger.reserve(key, identity={}, output_path=evidence)
        ledger.transition(key, "running")
        evidence.write_text(json.dumps({"schema": "s", "run_id": "r",
                                        "dispatch": "d"}))
        ledger.complete(key, evidence, expected_schema="s",
                        run_id="r", dispatch_id="d")
        for illegal in ("running", "reserved", "interrupted", "spent"):
            with pytest.raises(ExecutionCustodyError, match="terminal"):
                ledger.transition(key, illegal)

    @pytest.mark.parametrize("state", ["interrupted", "spent"])
    def test_other_terminals_never_move(self, tmp_path, state):
        ledger, key = make_ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e.json")
        ledger.transition(key, "running")
        ledger.transition(key, state)
        with pytest.raises(ExecutionCustodyError, match="terminal"):
            ledger.transition(key, "running")
        with pytest.raises(ExecutionCustodyError, match="UNCERTAIN"):
            ledger.reserve(key, identity={},
                           output_path=tmp_path / "e2.json")

    def test_reserved_cannot_jump_to_completed(self, tmp_path):
        ledger, key = make_ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e.json")
        with pytest.raises(ExecutionCustodyError,
                           match="illegal transition"):
            ledger.transition(key, "completed")

    def test_absent_record_cannot_transition(self, tmp_path):
        ledger, key = make_ledger(tmp_path)
        with pytest.raises(ExecutionCustodyError, match="no ledger"):
            ledger.transition(key, "running")

    def test_forward_started_blocks_failed_before_forward(self, tmp_path):
        ledger, key = make_ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e.json")
        ledger.transition(key, "running")
        ledger.mark_forward_started(key)
        assert ledger.read(key)["forward_started"] is True  # durable
        with pytest.raises(ExecutionCustodyError,
                           match="forward_started"):
            ledger.transition(key, "failed_before_forward")
        ledger.transition(key, "spent")  # the honest terminal

    def test_transition_sequence_is_monotonic(self, tmp_path):
        ledger, key = make_ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e.json")
        seq0 = ledger.read(key)["transition_sequence"]
        ledger.transition(key, "running")
        seq1 = ledger.read(key)["transition_sequence"]
        ledger.mark_forward_started(key)
        seq2 = ledger.read(key)["transition_sequence"]
        assert seq0 < seq1 < seq2

    def test_retirement_is_no_clobber(self, tmp_path):
        ledger, key = make_ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e.json")
        ledger.transition(key, "failed_before_forward")
        (ledger.root / f"{key}.retired-1.json").write_text("{}")
        with pytest.raises(ExecutionCustodyError, match="no-clobber"):
            ledger.reserve(key, identity={},
                           output_path=tmp_path / "e2.json")

    def test_every_ledger_write_fsyncs_file_and_directory(
            self, tmp_path, monkeypatch):
        import stat

        import agent_plugins.dispatch_custody as custody
        fsynced = []
        real_fsync = os.fsync

        def spy(fd):
            fsynced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
            return real_fsync(fd)
        monkeypatch.setattr(custody.os, "fsync", spy)
        ledger, key = make_ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e.json")
        assert True in fsynced and False in fsynced  # dir AND file
        fsynced.clear()
        ledger.transition(key, "running")
        assert True in fsynced and False in fsynced

    def test_directory_fsync_failure_never_acknowledges(self, tmp_path,
                                                        monkeypatch):
        import stat

        import agent_plugins.dispatch_custody as custody
        ledger, key = make_ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e.json")
        real_fsync = os.fsync

        def dir_boom(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("injected directory fsync failure")
            return real_fsync(fd)
        monkeypatch.setattr(custody.os, "fsync", dir_boom)
        with pytest.raises(OSError, match="directory fsync"):
            ledger.transition(key, "running")
        monkeypatch.undo()
        # the acknowledged state never changed... the tmp+rename commit
        # happens before the dir fsync, so the visible state may be
        # running, but the caller saw an EXCEPTION — nothing was
        # acknowledged as durable
        assert ledger.read(key)["state"] in ("reserved", "running")

    def test_symlink_ledger_root_refuses(self, tmp_path):
        real = tmp_path / "real_ledger"
        real.mkdir()
        link = tmp_path / "ledger_link"
        link.symlink_to(real)
        with pytest.raises(ExecutionCustodyError, match="symlink"):
            DispatchLedger(link)

    def test_ledger_root_and_record_permissions(self, tmp_path):
        ledger, key = make_ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e.json")
        assert (os.stat(ledger.root).st_mode & 0o777) == 0o700
        assert (os.stat(ledger.root / f"{key}.json").st_mode
                & 0o777) == 0o600


class TestDataSota360EvidenceAuthenticity:
    def _completed(self, tmp_path, packet_overrides=None):
        ledger, key = make_ledger(tmp_path)
        evidence = tmp_path / "evidence.json"
        ledger.reserve(key, identity={
            "architecture_digest": "archd",
            "config_snapshot_digest": "confd"},
            output_path=evidence)
        ledger.transition(key, "running")
        packet = {"schema": "agent_multi.transfer_loader_smoke.v3",
                  "run_id": "runX", "dispatch": "dispatchX",
                  "architecture_digest": "archd",
                  "config_snapshot_digest": "confd",
                  "forward_output_shape": [3, 96]}
        packet.update(packet_overrides or {})
        evidence.write_text(json.dumps(packet))
        ledger.complete(key, evidence,
                        expected_schema="agent_multi.transfer_loader_"
                                        "smoke.v3",
                        run_id="runX", dispatch_id="dispatchX")
        return ledger, key, evidence

    def test_render_verifies_and_is_repeatable(self, tmp_path):
        ledger, key, _evidence = self._completed(tmp_path)
        first = ledger.verified_render(key)
        second = ledger.verified_render(key)
        assert first == second
        assert first["forward_output_shape"] == [3, 96]
        record = ledger.read(key)
        assert record["evidence_sha256"]

    def test_fabricated_packet_without_ledger_refuses(self, tmp_path):
        """The PRE counterexample: arbitrary JSON rendered rc=0."""
        ledger, key = make_ledger(tmp_path)
        with pytest.raises(ExecutionCustodyError,
                           match="requires BOTH|no ledger"):
            ledger.verified_render(key)

    def test_substituted_evidence_refuses(self, tmp_path):
        ledger, key, evidence = self._completed(tmp_path)
        packet = json.loads(evidence.read_text())
        packet["forward_output_shape"] = [999, 999]
        evidence.write_text(json.dumps(packet))
        with pytest.raises(ExecutionCustodyError,
                           match="digest mismatch"):
            ledger.verified_render(key)

    @pytest.mark.parametrize("field, value, fragment", [
        ("run_id", "OTHER", "run_id"),
        ("schema", "some.other.schema", "schema"),
        ("dispatch", "OTHER_DISPATCH", "dispatch"),
        ("architecture_digest", "tampered", "architecture_digest"),
    ], ids=["run-id", "schema", "dispatch", "arch-digest"])
    def test_wrong_identity_fields_refuse(self, tmp_path, field, value,
                                          fragment):
        ledger, key, _ = self._completed(
            tmp_path, packet_overrides={field: value})
        with pytest.raises(ExecutionCustodyError, match=fragment):
            ledger.verified_render(key)

    def test_missing_evidence_after_completed_refuses(self, tmp_path):
        ledger, key, evidence = self._completed(tmp_path)
        evidence.unlink()
        with pytest.raises(ExecutionCustodyError, match="MISSING"):
            ledger.verified_render(key)

    def test_completion_requires_existing_evidence(self, tmp_path):
        ledger, key = make_ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e.json")
        ledger.transition(key, "running")
        with pytest.raises(ExecutionCustodyError, match="absent"):
            ledger.complete(key, tmp_path / "e.json",
                            expected_schema="s", run_id="r",
                            dispatch_id="d")

    def test_completion_write_failure_leaves_run_spent(self, tmp_path,
                                                       monkeypatch):
        import stat

        import agent_plugins.dispatch_custody as custody
        ledger, key = make_ledger(tmp_path)
        evidence = tmp_path / "evidence.json"
        ledger.reserve(key, identity={}, output_path=evidence)
        ledger.transition(key, "running")
        evidence.write_text(json.dumps({"schema": "s", "run_id": "r",
                                        "dispatch": "d"}))
        real_fsync = os.fsync

        def boom(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("injected completion failure")
            return real_fsync(fd)
        monkeypatch.setattr(custody.os, "fsync", boom)
        with pytest.raises(OSError, match="completion failure"):
            ledger.complete(key, evidence, expected_schema="s",
                            run_id="r", dispatch_id="d")
        monkeypatch.undo()
        # the run is treated as SPENT by the executing tool; even
        # before that, custody refuses a re-reservation (uncertain)
        state = ledger.read(key)["state"]
        if state == "running":
            ledger.transition(key, "spent")
        with pytest.raises(ExecutionCustodyError,
                           match="UNCERTAIN|COMPLETED"):
            ledger.reserve(key, identity={},
                           output_path=tmp_path / "e2.json")
