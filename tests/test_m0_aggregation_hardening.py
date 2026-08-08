"""Hardened M0 aggregation (Musashi in-flight order 2026-08-07): all 16
records verified before voting — contract, hashes, uniform revision,
real compute, no 2025, loadable artifacts, protected entries, no
errors. An unverifiable record blocks aggregation entirely.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tools.aggregate_eth_sac_inner_curriculum import (
    CONTRACT_PATH,
    verify_cross_record_uniformity,
    verify_m0_record,
)

CONTRACT = json.loads(CONTRACT_PATH.read_text())
CONTRACT_SHA = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
M0_ROOT = Path.home() / ".local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1"


def _terminal_zip(tmp_path: Path) -> Path:
    path = tmp_path / "terminal.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", "x")
    return path


def _valid_record(tmp_path: Path, arm="E1_N1_LR01", seed=101) -> dict:
    terminal = _terminal_zip(tmp_path)
    spec = CONTRACT["arms"][arm]
    revs = {"agent-multi": "27a3778b", "gym-fx": "efa4916",
            "doin-plugins": "8c959a6"}
    return {
        "schema": "agent_multi.m0_arm_record.v1",
        "arm": arm,
        "seed": seed,
        "m0_contract_sha256": CONTRACT_SHA,
        "arm_spec": dict(spec),
        "easy_learning_rate": CONTRACT["easy_learning_rate"],
        "execution_id": "e" * 16,
        "anchor_sha256": CONTRACT["anchors"][str(seed)]["sha256"],
        "terminal_sha256": hashlib.sha256(
            terminal.read_bytes()).hexdigest(),
        "code_revisions_before": revs,
        "code_revisions_after": dict(revs),
        "started_utc": "2026-08-07T20:00:00+00:00",
        "finished_utc": "2026-08-07T21:00:00+00:00",
        "boundary_transfer_evidence": {
            "policy_hash_matches_source_after_transfer": True},
        "epoch_history": [
            {"checkpoint_source": "warm_start_normal_baseline",
             "epoch": 0},
            {"epoch": 1, "val_action_raw_std": 0.05,
             "val_action_non_hold_rate": 0.9,
             "val_entry_orders_submitted": 7},
        ][: 1 + spec["normal_epochs"]],
        "terminal_evaluation": {
            "artifact_path": str(terminal),
            "artifact_sha256": hashlib.sha256(
                terminal.read_bytes()).hexdigest(),
            "splits_raw": {"validation": {"trades_total": 122,
                                          "total_return": 0.02}},
        },
        "decision_facts": {
            "activity_survived_normal": True,
            "weights_changed_from_anchor": True,
            "normal_updates_applied":
                spec["normal_epochs"] * 20000 - 1000,
            "terminal_usable": True,
        },
    }


def _verify(record, arm="E1_N1_LR01", seed=101, load_proof=False):
    return verify_m0_record(record, arm, seed, contract=CONTRACT,
                            contract_sha=CONTRACT_SHA,
                            load_proof=load_proof)


class TestVerificationClasses:
    def test_valid_record_passes(self, tmp_path):
        assert _verify(_valid_record(tmp_path)) == []

    def test_contract_hash_mismatch(self, tmp_path):
        record = _valid_record(tmp_path)
        record["m0_contract_sha256"] = "0" * 64
        assert any("contract hash" in p for p in _verify(record))

    def test_tampered_arm_spec(self, tmp_path):
        record = _valid_record(tmp_path)
        record["arm_spec"]["normal_learning_rate"] = 5e-4
        assert any("arm_spec differs" in p for p in _verify(record))

    def test_wrong_anchor(self, tmp_path):
        record = _valid_record(tmp_path)
        record["anchor_sha256"] = "f" * 64
        assert any("anchor sha" in p for p in _verify(record))

    def test_terminal_hash_mismatch_on_disk(self, tmp_path):
        record = _valid_record(tmp_path)
        record["terminal_sha256"] = "a" * 64
        record["terminal_evaluation"]["artifact_sha256"] = "a" * 64
        assert any("hash mismatch on disk" in p for p in _verify(record))

    def test_revision_moved_during_arm(self, tmp_path):
        record = _valid_record(tmp_path)
        record["code_revisions_after"] = {"agent-multi": "deadbeef"}
        assert any("revisions moved" in p for p in _verify(record))

    def test_fake_compute_rejected(self, tmp_path):
        record = _valid_record(tmp_path)
        record["decision_facts"]["normal_updates_applied"] = 500
        assert any("gradient updates" in p for p in _verify(record))

    def test_2025_leak_rejected(self, tmp_path):
        record = _valid_record(tmp_path)
        record["terminal_evaluation"]["splits_raw"]["validation"][
            "window_end"] = "2025-01-03T00:00:00"
        assert any("2025" in p for p in _verify(record))

    def test_test_split_presence_rejected(self, tmp_path):
        record = _valid_record(tmp_path)
        record["terminal_evaluation"]["splits_raw"]["test"] = {}
        assert any("test split" in p for p in _verify(record))

    def test_corrupt_artifact_rejected(self, tmp_path):
        record = _valid_record(tmp_path)
        path = Path(record["terminal_evaluation"]["artifact_path"])
        path.write_bytes(b"not a zip at all")
        record["terminal_sha256"] = hashlib.sha256(
            path.read_bytes()).hexdigest()
        assert any("not a zip" in p for p in _verify(record))

    def test_unloadable_sac_rejected_with_load_proof(self, tmp_path):
        record = _valid_record(tmp_path)
        assert any("NOT loadable" in p
                   for p in _verify(record, load_proof=True))

    def test_survival_without_protected_entry_rejected(self, tmp_path):
        record = _valid_record(tmp_path)
        record["epoch_history"][-1]["val_entry_orders_submitted"] = 0
        assert any("no protected entry" in p for p in _verify(record))

    def test_missing_boundary_proof_rejected(self, tmp_path):
        record = _valid_record(tmp_path)
        record["boundary_transfer_evidence"] = {
            "policy_hash_matches_source_after_transfer": False}
        assert any("boundary transfer" in p for p in _verify(record))


class TestCrossRecordUniformity:
    def test_divergent_revisions_block(self, tmp_path):
        a = _valid_record(tmp_path)
        b = _valid_record(tmp_path)
        b["code_revisions_before"] = {"agent-multi": "other"}
        b["code_revisions_after"] = {"agent-multi": "other"}
        problems = verify_cross_record_uniformity(
            {"seed101/E1_N1_LR01": a, "seed202/E1_N1_LR01": b})
        assert any("different code revisions" in p for p in problems)

    def test_uniform_records_pass(self, tmp_path):
        a = _valid_record(tmp_path)
        b = _valid_record(tmp_path, seed=202)
        assert verify_cross_record_uniformity(
            {"a": a, "b": b}) == []


class TestAgainstLiveRecords:
    def test_landed_real_record_verifies(self):
        """Read-only integration check against a record the RUNNING M0
        fleet already produced. Skipped where the run root is absent."""
        candidates = sorted(M0_ROOT.glob("seed*/*/m0_arm_record.json"))
        if not candidates:
            pytest.skip("no landed M0 records on this host yet")
        path = candidates[0]
        record = json.loads(path.read_text())
        arm = record["arm"]
        seed = record["seed"]
        problems = verify_m0_record(
            record, arm, seed, contract=CONTRACT,
            contract_sha=CONTRACT_SHA, load_proof=False)
        assert problems == [], problems
