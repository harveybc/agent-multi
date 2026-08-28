"""C3 (SAC driver correction order 2026-08-28): adversarial identity
tests for load_into_sac_policy — the treatment initialization applied
INSIDE the accepted trainer. Uses the sealed synthetic generation from
the loader suite; the real-architecture both-arm proof is the C4 CPU
dry run."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.pretrained_branch_loader import (  # noqa: E402
    TransferLoadError, load_into_sac_policy)
from tests.unit.test_pretrained_branch_loader import (  # noqa: E402
    sealed_run, verifiable, working_copy)  # fixtures


def _extractor(contract_file: Path):
    from agent_plugins.component_config import deep_merge_strict
    from app.plugin_loader import load_plugin
    contract = json.loads(contract_file.read_text())
    modules = torch.nn.ModuleList()
    for branch in contract["branches"]:
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      branch["plugin"])
        params = deep_merge_strict(plugin_class.plugin_params,
                                   branch["params"], path="p")
        module, _dim = plugin_class.build(
            len(branch["features"]), contract["window_size"], params)
        modules.append(module)
    return types.SimpleNamespace(temporal_branches=modules)


def _fake_sac(contract_file: Path, *, shared: bool = False,
              freeze_actor_encoders: bool = False):
    """Minimal object with SB3 SAC's policy surface: actor, critic and
    critic_target networks, separate grouped extractors, per-network
    optimizers over their own parameters."""
    actor_ex = _extractor(contract_file)
    critic_ex = actor_ex if shared else _extractor(contract_file)
    target_ex = _extractor(contract_file)
    if freeze_actor_encoders:
        for branch in actor_ex.temporal_branches:
            for param in branch.parameters():
                param.requires_grad = False

    def network(extractor):
        params = [p for b in extractor.temporal_branches
                  for p in b.parameters()]
        return types.SimpleNamespace(
            features_extractor=extractor,
            optimizer=torch.optim.Adam(params, lr=1e-3))

    policy = types.SimpleNamespace(
        actor=network(actor_ex), critic=network(critic_ex),
        critic_target=types.SimpleNamespace(
            features_extractor=target_ex))
    return types.SimpleNamespace(policy=policy)


def _load(verifiable, model, **kwargs):
    return load_into_sac_policy(
        model, verifiable["dir"], verifiable["repo_root"],
        verifiable["csv"], **kwargs)


class TestTransferInitIntoPolicy:
    def test_all_three_extractors_loaded_with_bit_parity(
            self, verifiable):
        model = _fake_sac(verifiable["contract_file"])
        report = _load(verifiable, model)
        assert set(report["extractors"]) == {"actor", "critic",
                                             "critic_target"}
        for name, sub in report["extractors"].items():
            assert all(f["bit_parity"]
                       for f in sub["families"].values()), name
        # actor and critic encoders now agree tensor-for-tensor
        for a_branch, c_branch in zip(
                model.policy.actor.features_extractor
                .temporal_branches,
                model.policy.critic.features_extractor
                .temporal_branches):
            for (ka, va), (kc, vc) in zip(
                    a_branch.state_dict().items(),
                    c_branch.state_dict().items()):
                assert ka == kc and torch.equal(va, vc)
        assert report["trainability"]["actor"]["all_in_optimizer"]
        assert report["trainability"]["critic"]["all_requires_grad"]
        assert report["trainability"]["critic_target"][
            "polyak_tracked"]

    def test_shared_extractor_refuses(self, verifiable):
        model = _fake_sac(verifiable["contract_file"], shared=True)
        with pytest.raises(TransferLoadError, match="share"):
            _load(verifiable, model)

    def test_frozen_encoder_params_refuse(self, verifiable):
        model = _fake_sac(verifiable["contract_file"],
                          freeze_actor_encoders=True)
        with pytest.raises(TransferLoadError,
                           match="remain trainable"):
            _load(verifiable, model)

    def test_seal_binding_mismatch_refuses_before_weights_move(
            self, verifiable):
        model = _fake_sac(verifiable["contract_file"])
        before = [t.clone() for t in model.policy.actor
                  .features_extractor.temporal_branches[0]
                  .state_dict().values()]
        with pytest.raises(TransferLoadError,
                           match="design binding"):
            _load(verifiable, model,
                  expected_seal_manifest_sha256="0" * 64)
        after = list(model.policy.actor.features_extractor
                     .temporal_branches[0].state_dict().values())
        for b, a in zip(before, after):
            assert torch.equal(b, a)  # nothing moved

    def test_non_grouped_extractor_refuses(self, verifiable):
        model = _fake_sac(verifiable["contract_file"])
        model.policy.critic_target.features_extractor = (
            types.SimpleNamespace())
        with pytest.raises(TransferLoadError,
                           match="not the grouped route"):
            _load(verifiable, model)
