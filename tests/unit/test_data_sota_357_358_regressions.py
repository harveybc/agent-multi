"""Frozen counterexamples for DATA-SOTA-357/358 (correction order
2026-08-27). PRE reproductions:
docs/audits/evidence/DATA_SOTA_357_358_REPRODUCTIONS_PRE.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.dispatch_custody import (  # noqa: E402
    DispatchLedger, ExecutionCustodyError, dispatch_key)
from agent_plugins.grouped_architecture import (  # noqa: E402
    ArchitectureError, assert_same_materialization, construct_extractor,
    materialize_from_config, materialize_from_file)
from agent_plugins.pretrained_branch_loader import (  # noqa: E402
    TransferLoadError, verify_architecture_matches_contract)

STRONG_CONFIG = (REPO / "examples/config/"
                 "project3_ethusdt_4h_sac_grouped_strong_v1.json")
WEAK_CONFIG = (REPO / "examples/config/"
               "project3_ethusdt_4h_sac_grouped_features_v1.json")
V4_CONTRACT = (REPO / "examples/config/"
               "pretrain_contract_eth_h4_o2022_v4.json")


def observation_space_for(config):
    import numpy as np
    from gymnasium import spaces
    arch = config["feature_extractor_config"]
    blocks = {"features": spaces.Box(-np.inf, np.inf,
                                     shape=(config["window_size"],
                                            len(arch["feature_columns"])),
                                     dtype=np.float32)}
    for key in arch["state_keys"]:
        blocks[key] = spaces.Box(-np.inf, np.inf, shape=(1,),
                                 dtype=np.float32)
    return spaces.Dict(blocks)


# ------------------------------------------- 357: canonical architecture

class TestDataSota357CanonicalArchitecture:
    def test_smoke_tool_has_no_authored_architecture(self):
        """The PRE counterexample: the committed tool hardcoded
        state_branch/fusion dicts and a rejected-keys literal."""
        source = (REPO / "tools/load_pretrained_branches_smoke.py"
                  ).read_text()
        assert '"state_branch": {"plugin"' not in source
        assert '"fusion": {"plugin"' not in source
        assert '"rejected_keys_total": 0' not in source
        assert "materialize_from_file" in source

    def test_weak_config_would_have_refused_the_transfer(self):
        """The masked mismatch (self-reported amplification): the v1
        config declares the WEAK route; binding the effective
        architecture makes the loader refuse it."""
        materialized = materialize_from_file(WEAK_CONFIG)
        contract = json.loads(V4_CONTRACT.read_text())
        with pytest.raises(TransferLoadError, match="do NOT fit"):
            verify_architecture_matches_contract(materialized, contract)

    def test_strong_config_matches_the_sealed_contract(self):
        materialized = materialize_from_file(STRONG_CONFIG)
        contract = json.loads(V4_CONTRACT.read_text())
        verify_architecture_matches_contract(materialized, contract)
        assert materialized["expected_output_dim"] == 96
        assert materialized["ordered_families"][-1] == "account_state"

    @pytest.mark.parametrize("mutator, fragment", [
        (lambda c: c.pop("feature_extractor_config"),
         "must be declared"),
        (lambda c: c["feature_extractor_config"].pop("state_branch"),
         "EXPLICITLY"),
        (lambda c: c["feature_extractor_config"].pop("fusion"),
         "EXPLICITLY"),
        (lambda c: c["feature_extractor_config"].pop("state_keys"),
         "EXPLICITLY"),
        (lambda c: c["feature_extractor_config"].update(
            {"unknown_block": 1}), "unknown"),
        (lambda c: c["feature_extractor_config"]["fusion"]["params"]
            .pop("output_dim"), "output_dim"),
        (lambda c: c.update(feature_columns=["x"]), "IDENTICAL"),
    ], ids=["no-arch", "no-state-branch", "no-fusion", "no-state-keys",
            "extra-key", "no-output-dim", "column-mismatch"])
    def test_materializer_refusals(self, mutator, fragment):
        config = json.loads(STRONG_CONFIG.read_text())
        mutator(config)
        with pytest.raises((ArchitectureError, ValueError),
                           match=fragment):
            materialize_from_config(config)

    @pytest.mark.parametrize("mutator", [
        lambda a: a["fusion"]["params"].update(n_heads=8),
        lambda a: a["fusion"]["params"].update(output_dim=128),
        lambda a: a["state_branch"]["params"].update(output_dim=32),
        lambda a: a["branches"].reverse(),
        lambda a: a["branches"][0].update(plugin="tcn_branch",
                                          params={"channels": [32, 32]}),
    ], ids=["fusion-heads", "fusion-dim", "state-branch", "family-order",
            "same-shape-different-plugin"])
    def test_architecture_digest_is_sensitive(self, mutator):
        base = json.loads(STRONG_CONFIG.read_text())
        mutated = json.loads(STRONG_CONFIG.read_text())
        mutator(mutated["feature_extractor_config"])
        a = materialize_from_config(base)
        b = materialize_from_config(mutated)
        assert a["architecture_digest"] != b["architecture_digest"]

    def test_config_mutation_after_verification_refuses(self, tmp_path):
        copy = tmp_path / "config.json"
        copy.write_text(STRONG_CONFIG.read_text())
        first = materialize_from_file(copy)
        mutated = json.loads(copy.read_text())
        mutated["feature_extractor_config"]["fusion"]["params"][
            "output_dim"] = 128
        copy.write_text(json.dumps(mutated))
        with pytest.raises(ArchitectureError, match="divergence"):
            assert_same_materialization(first,
                                        materialize_from_file(copy))

    def test_sac_route_and_smoke_route_bind_identical_architecture(self):
        """WP1 item 6: one digest from both routes and identical
        initialized non-transferred state/fusion tensors under one
        seed."""
        config = json.loads(STRONG_CONFIG.read_text())
        sac_route = materialize_from_config(config)   # sac_agent path
        smoke_route = materialize_from_file(STRONG_CONFIG)
        assert_same_materialization(sac_route, smoke_route)

        space = observation_space_for(config)
        torch.manual_seed(0)
        a = construct_extractor(sac_route, space)
        torch.manual_seed(0)
        b = construct_extractor(smoke_route, space)
        for module_name in ("state_branch", "fusion"):
            sa = getattr(a, module_name).state_dict()
            sb = getattr(b, module_name).state_dict()
            assert sa.keys() == sb.keys()
            for key in sa:
                assert torch.equal(sa[key], sb[key]), (
                    f"{module_name}.{key} differs between routes")
        assert a.fusion.family_digest == b.fusion.family_digest

    def test_constructed_output_dim_must_match_binding(self):
        config = json.loads(STRONG_CONFIG.read_text())
        materialized = materialize_from_config(config)
        materialized["expected_output_dim"] = 128  # tampered binding
        with pytest.raises(ArchitectureError, match="output dimension"):
            construct_extractor(materialized,
                                observation_space_for(config))


# ------------------------------------------- 358: execution custody

class TestDataSota358SingleUseCustody:
    KEY_FIELDS = dict(dispatch_id="d", generation_digest="g",
                      architecture_digest="a", data_digest="x",
                      code_identity={"c": 1})

    def _ledger(self, tmp_path):
        return DispatchLedger(tmp_path / "ledger"), dispatch_key(
            **self.KEY_FIELDS)

    def test_completed_dispatch_refuses_second_execution(self, tmp_path):
        ledger, key = self._ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e1.json")
        ledger.transition(key, "running")
        ledger.transition(key, "completed")
        with pytest.raises(ExecutionCustodyError, match="COMPLETED"):
            ledger.reserve(key, identity={},
                           output_path=tmp_path / "e2.json")

    @pytest.mark.parametrize("state", ["reserved", "running",
                                       "interrupted"])
    def test_uncertain_states_are_spent(self, tmp_path, state):
        ledger, key = self._ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e1.json")
        if state != "reserved":
            ledger.transition(key, state)
        with pytest.raises(ExecutionCustodyError, match="UNCERTAIN"):
            ledger.reserve(key, identity={},
                           output_path=tmp_path / "e2.json")

    def test_failed_before_forward_permits_retry(self, tmp_path):
        ledger, key = self._ledger(tmp_path)
        ledger.reserve(key, identity={}, output_path=tmp_path / "e1.json")
        ledger.transition(key, "failed_before_forward")
        ledger.reserve(key, identity={}, output_path=tmp_path / "e2.json")
        assert ledger.read(key)["attempt"] == 2  # prior attempt retired

    def test_concurrent_reservation_has_one_winner(self, tmp_path):
        root = tmp_path / "ledger"
        key = dispatch_key(**self.KEY_FIELDS)
        DispatchLedger(root).reserve(key, identity={},
                                     output_path=tmp_path / "e1.json")
        with pytest.raises(ExecutionCustodyError,
                           match="UNCERTAIN|concurrent"):
            DispatchLedger(root).reserve(
                key, identity={}, output_path=tmp_path / "e2.json")

    def test_output_collision_refuses(self, tmp_path):
        ledger, key = self._ledger(tmp_path)
        out = tmp_path / "evidence.json"
        out.write_text("{}")
        with pytest.raises(ExecutionCustodyError, match="no-clobber"):
            ledger.reserve(key, identity={}, output_path=out)

    def test_symlink_destination_refuses(self, tmp_path):
        ledger, key = self._ledger(tmp_path)
        target = tmp_path / "real.json"
        link = tmp_path / "link.json"
        target.write_text("{}")
        link.symlink_to(target)
        with pytest.raises(ExecutionCustodyError, match="symlink"):
            ledger.reserve(key, identity={}, output_path=link)

    def test_renderer_never_constructs_a_model(self, tmp_path,
                                               monkeypatch):
        from tools import load_pretrained_branches_smoke as tool
        evidence = tmp_path / "done.json"
        evidence.write_text(json.dumps(
            {"schema": "agent_multi.transfer_loader_smoke.v2",
             "status": "MECHANICS_ONLY", "run_id": "abc",
             "forward_output_shape": [3, 96]}))

        def boom(*args, **kwargs):
            raise AssertionError("renderer constructed a model")
        monkeypatch.setattr(tool, "construct_extractor", boom)
        monkeypatch.setattr(tool, "load_family_encoders", boom)
        assert tool.render(evidence) == 0  # rerunnable, model-free

    def test_deviation_record_is_written_once_with_known_facts(
            self, tmp_path):
        ledger, key = self._ledger(tmp_path)
        facts = {"incident": "2026-08-27 double invocation",
                 "first_run_metrics": "NOT_PRESERVED (never invented)"}
        ledger.record_protocol_deviation(key, facts)
        record = ledger.read(key)
        assert record["state"] == "DISCLOSED_PROTOCOL_DEVIATION"
        assert record["facts"]["first_run_metrics"].startswith(
            "NOT_PRESERVED")
        with pytest.raises(ExecutionCustodyError, match="never overwrite"):
            ledger.record_protocol_deviation(key, facts)
