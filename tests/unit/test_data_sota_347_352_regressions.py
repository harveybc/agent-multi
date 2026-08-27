"""Frozen counterexamples for DATA-SOTA-347..350 (correction order
@e9af87a3 companion; 351 lives in the lts scheduler suite and 352 in
test_usdcop_trm_collector.py). PRE reproductions:
docs/audits/evidence/DATA_SOTA_347_352_REPRODUCTIONS_PRE.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    ORIGIN_DECISION_SCHEMA, PretrainContractError, parse_iso_utc,
    sha256_file, three_way_split, validate_branch_partition,
    validate_contract, verify_earlier_origin_decision)
from pipeline_plugins._observation_contract import (  # noqa: E402
    feature_columns_sha256)
from tests.unit.test_branch_pretraining import contract_with  # noqa: E402


# ------------------------------------------- 347: verified origin chain

class TestDataSota347VerifiedOriginAuthority:
    def test_impossible_calendar_dates_refuse(self):
        """The PRE counterexample: fit_end 2021-13-45T99:99:99 passed
        the string-prefix comparison."""
        with pytest.raises(PretrainContractError, match="ISO-8601"):
            validate_contract(contract_with(
                fit_end="2021-13-45T99:99:99"))
        with pytest.raises(PretrainContractError, match="ISO-8601"):
            validate_contract(contract_with(score_origin={
                "origin_id": "o2022", "score_start": "not-a-date"}))

    def test_naive_timestamps_are_declared_utc(self):
        naive = parse_iso_utc("2021-12-31T20:00:00", "x")
        aware = parse_iso_utc("2021-12-31T20:00:00+00:00", "x")
        assert naive == aware and naive.tzinfo is not None

    def _decision_contract(self, tmp_path, **artifact_overrides):
        artifact = {"schema": ORIGIN_DECISION_SCHEMA,
                    "origin_id": "o2022",
                    "score_start": "2022-01-01T00:00:00Z",
                    "decided_at": "2022-12-01T00:00:00Z",
                    "decision": "o2022 frozen"}
        artifact.update(artifact_overrides)
        path = tmp_path / "o2022_decision.json"
        path.write_text(json.dumps(artifact))
        contract = contract_with(
            score_origin={"origin_id": "o2023",
                          "score_start": "2023-01-01"},
            origin_plan=[
                {"origin_id": "o2022", "score_start": "2022-01-01",
                 "predecessor_origin_id": None},
                {"origin_id": "o2023", "score_start": "2023-01-01",
                 "predecessor_origin_id": "o2022"},
            ],
            fit_end="2022-12-31T20:00:00",
            materialized_at="2022-12-15T00:00:00Z")
        contract["earlier_origin_decision"] = {
            "origin_id": "o2022",
            "decided_at": "2022-12-01T00:00:00Z",
            "artifact": "o2022_decision.json",
            "artifact_sha256": sha256_file(path)}
        return contract, path

    def test_verified_decision_chain_passes(self, tmp_path):
        contract, _ = self._decision_contract(tmp_path)
        verified = verify_earlier_origin_decision(contract, tmp_path)
        assert verified["origin_id"] == "o2022"

    def test_digest_mismatch_refuses(self, tmp_path):
        contract, path = self._decision_contract(tmp_path)
        path.write_text(path.read_text() + " ")
        with pytest.raises(PretrainContractError,
                           match="digest mismatch"):
            verify_earlier_origin_decision(contract, tmp_path)

    def test_wrong_schema_refuses(self, tmp_path):
        contract, _ = self._decision_contract(
            tmp_path, schema="something.else.v9")
        contract["earlier_origin_decision"]["artifact_sha256"] = \
            sha256_file(tmp_path / "o2022_decision.json")
        with pytest.raises(PretrainContractError, match="schema"):
            verify_earlier_origin_decision(contract, tmp_path)

    def test_decision_must_predate_materialization(self, tmp_path):
        contract, path = self._decision_contract(
            tmp_path, decided_at="2022-12-20T00:00:00Z")
        contract["earlier_origin_decision"]["decided_at"] = \
            "2022-12-20T00:00:00Z"
        contract["earlier_origin_decision"]["artifact_sha256"] = \
            sha256_file(path)
        with pytest.raises(PretrainContractError,
                           match="predate this origin"):
            verify_earlier_origin_decision(contract, tmp_path)

    def test_non_anterior_origin_refuses(self, tmp_path):
        contract, path = self._decision_contract(
            tmp_path, score_start="2024-01-01T00:00:00Z")
        contract["earlier_origin_decision"]["artifact_sha256"] = \
            sha256_file(path)
        with pytest.raises(PretrainContractError,
                           match="does not precede"):
            verify_earlier_origin_decision(contract, tmp_path)

    def test_absent_artifact_refuses(self, tmp_path):
        contract, path = self._decision_contract(tmp_path)
        path.unlink()
        with pytest.raises(PretrainContractError, match="absent"):
            verify_earlier_origin_decision(contract, tmp_path)


# ------------------------------------------- 348: complete partition

class TestDataSota348CompleteOrderedPartition:
    COLUMNS = ["f1", "f2", "f3", "f4", "f5", "f6"]

    def test_missing_feature_refuses(self):
        """The PRE counterexample: 2 of 6 features silently dropped."""
        with pytest.raises(PretrainContractError,
                           match="incomplete partition.*f3"):
            validate_branch_partition(self.COLUMNS, [
                {"name": "a", "features": ["f1", "f2"]},
                {"name": "b", "features": ["f4", "f5"]}])

    def test_duplicate_feature_refuses(self):
        with pytest.raises(PretrainContractError,
                           match="assigned to both"):
            validate_branch_partition(self.COLUMNS, [
                {"name": "a", "features": ["f1", "f2", "f3"]},
                {"name": "b", "features": ["f3", "f4", "f5", "f6"]}])

    def test_empty_family_refuses(self):
        with pytest.raises(PretrainContractError, match="empty family"):
            validate_branch_partition(self.COLUMNS, [
                {"name": "a", "features": self.COLUMNS},
                {"name": "b", "features": []}])

    def test_reordered_family_refuses(self):
        with pytest.raises(PretrainContractError, match="canonical"):
            validate_branch_partition(self.COLUMNS, [
                {"name": "a", "features": ["f2", "f1", "f3"]},
                {"name": "b", "features": ["f4", "f5", "f6"]}])

    def test_valid_partition_binds_both_digest_levels(self):
        report = validate_branch_partition(self.COLUMNS, [
            {"name": "a", "features": ["f1", "f2", "f3"]},
            {"name": "b", "features": ["f4", "f5", "f6"]}])
        assert report["global_ordered_digest"] == \
            feature_columns_sha256(self.COLUMNS)
        assert report["family_ordered_digests"]["a"] == \
            feature_columns_sha256(["f1", "f2", "f3"])
        assert report["coverage"]["feature_count"] == 6

    def test_committed_v3_contract_covers_all_83_exactly_once(self):
        v3 = json.loads((REPO / "examples/config/"
                         "pretrain_contract_eth_h4_o2022_v4.json"
                         ).read_text())
        report = validate_branch_partition(v3["feature_columns"],
                                           v3["branches"])
        assert report["coverage"]["feature_count"] == 83
        assert report["coverage"]["family_count"] == 5
        assert len(report["family_ordered_digests"]) == 5


# ------------------------------------------- 349: honest calibration

class TestDataSota349HonestCalibration:
    def test_partitions_are_chronologically_ordered_and_disjoint(self):
        steps = list(range(100, 300))
        train, calibration, monitor, purged = three_way_split(
            steps, 0.15, 0.15, purge_steps=12)
        rebuilt = sorted(train + calibration + monitor + purged)
        assert rebuilt == steps
        assert max(train) < min(calibration)
        assert max(calibration) < min(monitor)
        # DATA-SOTA-353: purge of max(horizons) at each boundary
        assert min(calibration) - max(train) == 13
        assert min(monitor) - max(calibration) == 13

    def test_runner_calibrates_on_calibration_not_monitor(self):
        """Structural regression on the executing runner source: the
        initial losses feeding balance_objective_weights come from the
        CALIBRATION windows; the monitor never calibrates."""
        source = (REPO / "tools/pretrain_branches.py").read_text()
        calibration_block = source.split(
            "initial = {k: float(v) for k, v in objective_losses(")[1]
        assert "calibration_windows" in calibration_block.split(")")[0]
        assert "initial_calibration_losses" in source
        assert "initial_monitor_losses" not in source


# ------------------------------------------- 350: one input domain

class TestDataSota350SingleInputDomain:
    def test_domain_must_be_declared(self):
        contract = contract_with()
        del contract["objective_domain"]
        with pytest.raises(PretrainContractError,
                           match="objective_domain"):
            validate_contract(contract)

    def test_single_domain_mode_forbids_zscore_policies(self):
        contract = contract_with(
            objective_domain="single_domain_raw_targets")
        with pytest.raises(PretrainContractError,
                           match="single_domain_raw_targets"):
            validate_contract(contract)  # beta declares zscore

    def test_encoder_input_is_policy_independent(self):
        """DATA-SOTA-350: policies transform TARGETS only. The encoder
        consumes values.masked_fill(mask, 0) regardless of policy — the
        loss function receives values and target separately."""
        from agent_plugins.branch_pretraining import (
            masked_reconstruction_loss, reconstruction_target)
        values = torch.randn(2, 16, 3) * 4 + 7
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, 3:7] = True
        captured = []

        class Spy(torch.nn.Module):
            def forward(self, x):
                captured.append(x.clone())
                return x.reshape(x.shape[0], -1)
        head = torch.nn.Identity()
        for policy in ({"policy": "identity_preprocessed", "eps": None},
                       {"policy": "window_zscore_visible",
                        "eps": 1e-5}):
            target = reconstruction_target(values, mask, policy)
            masked_reconstruction_loss(Spy(), head, values, target,
                                       mask)
        assert torch.equal(captured[0], captured[1]), (
            "encoder input differs across policies: domain mixing")
