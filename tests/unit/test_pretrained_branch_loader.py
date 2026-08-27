"""Adversarial fixtures for the transfer loader (Musashi dispatch
2026-08-27). Every fixture from the dispatch is here: torn/substituted
generation, v3 artifact offered to the v4 loader, reordered identity,
missing/extra/renamed/wrong-dtype/wrong-shape parameters, same-width
family exchange, head injection, optimizer/calibration payloads,
preprocessing/data drift, NaN output, and the repeated clean load with
identical output.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    PretrainContractError, sha256_file, write_generation)
from agent_plugins.pretrained_branch_loader import (  # noqa: E402
    TransferLoadError, check_finite_forward, load_family_encoders,
    refuse_non_encoder_payload, strict_load_encoder, verify_source)
from tests.unit.test_branch_pretraining import (  # noqa: E402
    SOURCE_CONFIG, contract_with, run_runner, synthetic_csv)

REFUSALS = (TransferLoadError, PretrainContractError)


def same_width_contract():
    """alpha and beta are IDENTICAL architectures over disjoint
    features — the hardest family-exchange case."""
    return contract_with(branches=[
        {"name": "alpha", "plugin": "gru_branch",
         "params": {"hidden_size": 8},
         "features": ["f1", "f2", "f3"]},
        {"name": "beta", "plugin": "gru_branch",
         "params": {"hidden_size": 8},
         "features": ["f4", "f5", "f6"]},
    ], normalization_policies={
        "alpha": {"policy": "identity_preprocessed"},
        "beta": {"policy": "identity_preprocessed"}})


@pytest.fixture(scope="module")
def sealed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("loader")
    csv = root / "synthetic.csv"
    synthetic_csv(csv, hours=260)
    source = root / "source_config.json"
    source.write_text(json.dumps(SOURCE_CONFIG))
    contract = same_width_contract()
    contract["observation_pipeline"]["source_config"] = str(source)
    contract_file = root / "contract.json"
    contract_file.write_text(json.dumps(contract))
    out = root / "run"
    run_runner(csv, contract_file, out, "--epochs", "1")
    return {"root": root, "csv": csv, "contract_file": contract_file,
            "out": out, "contract": contract}


@pytest.fixture()
def working_copy(sealed_run, tmp_path):
    """Fresh mutable copy of the sealed run for destructive fixtures.

    The manifest's sealed contract_path is repo-relative-or-external;
    verify_source resolves against repo_root, so tests pass the run's
    ROOT as repo_root with the contract at its sealed relative path."""
    dest = tmp_path / "run"
    shutil.copytree(sealed_run["out"], dest)
    return dest


def _repoint(run_dir: Path, contract_file: Path):
    """Re-seal the copied run so its identity references the given
    contract path relative to the run's parent (our test repo_root)."""
    manifest = json.loads((run_dir / "pretrain_manifest.json"
                           ).read_text())
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    for target in (manifest["identity"], ckpt["identity"]):
        target["contract_path"] = str(
            contract_file.relative_to(run_dir.parent))
        target["contract_sha256"] = sha256_file(contract_file)
    seal = json.loads((run_dir / "generation.json").read_text())
    write_generation(run_dir, ckpt, manifest, seal["generation"])
    return manifest


@pytest.fixture()
def verifiable(sealed_run, working_copy):
    """A working copy whose identity resolves from its own parent dir
    (contract copied beside it), against the synthetic dataset."""
    contract_file = working_copy.parent / "contract.json"
    shutil.copy(sealed_run["contract_file"], contract_file)
    _repoint(working_copy, contract_file)
    return {"dir": working_copy, "repo_root": working_copy.parent,
            "csv": sealed_run["csv"], "contract_file": contract_file}


def _verify(v):
    return verify_source(v["dir"], v["repo_root"], v["csv"])


class TestIdentityChain:
    def test_clean_verify_passes_and_reports_code_identity(self, verifiable):
        # the synthetic run's library/runner shas were sealed from THIS
        # tree, so file-sha identity holds; commit equality reported
        source = _verify(verifiable)
        assert source["code_identity_report"]["library_sha_equal"]
        assert source["partition"]["coverage"]["feature_count"] == 6

    def test_torn_generation_refuses(self, verifiable):
        path = verifiable["dir"] / "pretrain_manifest.json"
        path.write_text(path.read_text() + " ")
        with pytest.raises(REFUSALS, match="TORN GENERATION"):
            _verify(verifiable)

    def test_substituted_checkpoint_refuses(self, verifiable):
        (verifiable["dir"] / "checkpoint.pt").write_bytes(b"substitute")
        with pytest.raises(REFUSALS, match="TORN GENERATION"):
            _verify(verifiable)

    def test_v3_artifact_offered_to_v4_loader_refuses(self, verifiable):
        contract = json.loads(verifiable["contract_file"].read_text())
        contract["schema"] = "agent_multi.pretrain_contract.v3"
        v3_file = verifiable["repo_root"] / "contract_v3.json"
        v3_file.write_text(json.dumps(contract))
        _repoint(verifiable["dir"], v3_file)
        with pytest.raises(TransferLoadError,
                           match="does not validate for the v4"):
            _verify(verifiable)

    def test_reordered_family_identity_refuses(self, verifiable):
        contract = json.loads(verifiable["contract_file"].read_text())
        contract["branches"][0]["features"] = ["f2", "f1", "f3"]
        verifiable["contract_file"].write_text(json.dumps(contract))
        with pytest.raises(TransferLoadError,
                           match="contract identity drift"):
            _verify(verifiable)

    def test_data_digest_drift_refuses(self, verifiable, tmp_path):
        drifted = tmp_path / "drifted.csv"
        synthetic_csv(drifted, hours=260, start="2023-10-01")
        verifiable["csv"] = drifted
        with pytest.raises(TransferLoadError,
                           match="source-data digest drift"):
            _verify(verifiable)

    def test_preprocessing_config_drift_refuses(self, verifiable):
        source_path = Path(json.loads(
            verifiable["contract_file"].read_text()
        )["observation_pipeline"]["source_config"])
        mutated = json.loads(source_path.read_text())
        mutated["feature_scaling_window"] = 64
        source_path.write_text(json.dumps(mutated))
        try:
            with pytest.raises(TransferLoadError,
                               match="preprocessing identity drift"):
                _verify(verifiable)
        finally:
            mutated["feature_scaling_window"] = 32
            source_path.write_text(json.dumps(mutated))


class TestEncoderOnlyStrictLoad:
    @staticmethod
    def _extractor_for(verifiable):
        from app.plugin_loader import load_plugin
        contract = json.loads(verifiable["contract_file"].read_text())
        modules = []
        for branch in contract["branches"]:
            plugin_class, _ = load_plugin("feature_branch.plugins",
                                          branch["plugin"])
            from agent_plugins.component_config import deep_merge_strict
            params = deep_merge_strict(plugin_class.plugin_params,
                                       branch["params"], path="p")
            module, _dim = plugin_class.build(
                len(branch["features"]), contract["window_size"],
                params)
            modules.append(module)
        return (types.SimpleNamespace(temporal_branches=modules),
                contract)

    def test_clean_load_bit_parity_and_repeat_identical(self, verifiable):
        source = _verify(verifiable)
        extractor, contract = self._extractor_for(verifiable)
        result = load_family_encoders(verifiable["dir"],
                                       source["manifest"], contract,
                                       extractor)
        report = result["families"]
        accounting = result["accounting"]
        assert all(f["bit_parity"] for f in report.values())
        # DATA-SOTA-357: accounting is DERIVED with conservation
        assert accounting["offered_tensors"] == \
            accounting["loaded_tensors"] > 0
        assert accounting["rejected_total_derived"] == 0
        assert "DERIVED" in accounting["conservation"]
        assert all(v["bytes"] > 0
                   for v in accounting["loaded_per_family"].values())
        x = torch.randn(2, contract["window_size"], 3)
        with torch.no_grad():
            first = extractor.temporal_branches[0](x)
        # repeated clean load: identical output bit-for-bit
        extractor2, _ = self._extractor_for(verifiable)
        load_family_encoders(verifiable["dir"], source["manifest"],
                             contract, extractor2)
        with torch.no_grad():
            second = extractor2.temporal_branches[0](x)
        assert torch.equal(first, second)

    def test_same_width_family_exchange_refuses(self, verifiable):
        source = _verify(verifiable)
        extractor, contract = self._extractor_for(verifiable)
        a = verifiable["dir"] / "branch_alpha_encoder.pt"
        b = verifiable["dir"] / "branch_beta_encoder.pt"
        tmp = verifiable["dir"] / "swap.pt"
        a.rename(tmp); b.rename(a); tmp.rename(b)
        with pytest.raises(TransferLoadError,
                           match="WRONG family"):
            load_family_encoders(verifiable["dir"], source["manifest"],
                                 contract, extractor)

    def test_mutated_encoder_file_refuses_by_family_digest(
            self, verifiable):
        source = _verify(verifiable)
        extractor, contract = self._extractor_for(verifiable)
        path = verifiable["dir"] / "branch_alpha_encoder.pt"
        state = torch.load(path, weights_only=True)
        key = next(iter(state))
        state[key] = state[key].clone() + 1.0
        torch.save(state, path)
        with pytest.raises(TransferLoadError, match="WRONG family"):
            load_family_encoders(verifiable["dir"], source["manifest"],
                                 contract, extractor)

    @staticmethod
    def _fresh_module_and_state():
        module = torch.nn.Linear(4, 3)
        return module, dict(torch.nn.Linear(4, 3).state_dict())

    def test_missing_key_refuses(self):
        module, state = self._fresh_module_and_state()
        state.pop("bias")
        with pytest.raises(TransferLoadError, match="missing"):
            strict_load_encoder(module, state, "fam")

    def test_extra_or_head_injected_key_refuses(self):
        module, state = self._fresh_module_and_state()
        state["reconstruction_head.weight"] = torch.randn(3, 3)
        with pytest.raises(TransferLoadError, match="extra/injected"):
            strict_load_encoder(module, state, "fam")

    def test_renamed_key_refuses(self):
        module, state = self._fresh_module_and_state()
        state["kernel"] = state.pop("weight")
        with pytest.raises(TransferLoadError, match="missing"):
            strict_load_encoder(module, state, "fam")

    def test_wrong_shape_refuses(self):
        module, state = self._fresh_module_and_state()
        state["weight"] = torch.randn(3, 5)
        with pytest.raises(TransferLoadError, match="shape"):
            strict_load_encoder(module, state, "fam")

    def test_wrong_dtype_refuses(self):
        module, state = self._fresh_module_and_state()
        state["weight"] = state["weight"].double()
        with pytest.raises(TransferLoadError, match="dtype"):
            strict_load_encoder(module, state, "fam")

    def test_optimizer_state_refuses_as_category(self):
        opt = torch.optim.Adam(torch.nn.Linear(4, 3).parameters())
        with pytest.raises(TransferLoadError, match="OPTIMIZER"):
            refuse_non_encoder_payload(opt.state_dict(), "fam")

    @pytest.mark.parametrize("marker", [
        "effective_weights", "generator_state", "replay_buffer",
        "calibration"])
    def test_calibration_and_replay_payloads_refuse(self, marker):
        with pytest.raises(TransferLoadError, match="non-encoder"):
            refuse_non_encoder_payload({marker: 1, "w": 2}, "fam")


class TestForwardGuards:
    def test_nan_output_is_a_typed_failure(self):
        with pytest.raises(TransferLoadError, match="NaN/Inf"):
            check_finite_forward(torch.tensor([1.0, float("nan")]))
        with pytest.raises(TransferLoadError, match="NaN/Inf"):
            check_finite_forward(torch.tensor([float("inf")]))
        out = check_finite_forward(torch.ones(3))
        assert bool(out.isfinite().all())
