"""Regressions for DATA-SOTA-381/382/383 (GPU dispatch runtime
correction order 2026-08-28). CPU tests always run; CUDA tests are
single-GPU, bounded, and skip cleanly when no device is visible."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.dispatch_authorization import (  # noqa: E402
    AuthorizationRefused, bounded_extractor_forward,
    cudnn_micro_preflight, executable_manifest,
    resolve_required_entry_points, verify_device_binding)
from agent_plugins.pretrained_branch_loader import (  # noqa: E402
    TransferLoadError, load_family_encoders)
from tests.unit.test_pretrained_branch_loader import (  # noqa: E402
    _verify, sealed_run, verifiable, working_copy)  # fixtures

CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="single-GPU regression: CUDA absent")


def _extractor(verifiable):
    from tests.unit.test_data_sota_c3_sac_transfer_init import (
        _extractor as build)
    return build(verifiable["contract_file"])


class TestDataSota381CudaParity:
    def test_cpu_parity_unchanged(self, verifiable):
        source = _verify(verifiable)
        extractor = _extractor(verifiable)
        result = load_family_encoders(
            verifiable["dir"], source["manifest"],
            json.loads(verifiable["contract_file"].read_text()),
            extractor)
        assert all(f["bit_parity"]
                   for f in result["families"].values())

    @CUDA
    def test_cuda_target_loads_with_bit_parity(self, verifiable):
        """The 381 counterexample: a CUDA-resident target module used
        to raise 'Expected all tensors to be on the same device'."""
        source = _verify(verifiable)
        extractor = _extractor(verifiable)
        for branch in extractor.temporal_branches:
            branch.cuda()
        result = load_family_encoders(
            verifiable["dir"], source["manifest"],
            json.loads(verifiable["contract_file"].read_text()),
            extractor)
        assert all(f["bit_parity"]
                   for f in result["families"].values())
        for branch in extractor.temporal_branches:
            assert all(str(p.device).startswith("cuda")
                       for p in branch.parameters())

    def test_cross_device_comparison_still_detects_inequality(self):
        """The fix must not weaken the check: the verification copy
        moves device, and a single-element difference still fails."""
        sealed = torch.ones(4)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reloaded = torch.ones(4, device=device)
        assert torch.equal(reloaded, sealed.to(reloaded.device))
        tampered = sealed.clone()
        tampered[2] = torch.nextafter(tampered[2],
                                      torch.tensor(2.0))
        assert not torch.equal(reloaded,
                               tampered.to(reloaded.device))

    def test_dtype_drift_refuses_before_any_device_move(self):
        source_text = (REPO / "agent_plugins/"
                       "pretrained_branch_loader.py").read_text()
        assert "post-load dtype drift" in source_text
        assert "verification_copy" in source_text
        # only the VERIFICATION copy moves; module tensors never do
        assert "sealed.to(reloaded[key].device)" in source_text


class TestDataSota382EnvironmentalPreflight:
    def test_all_required_entry_points_resolve_with_metadata(self):
        result = resolve_required_entry_points(REPO)
        entry_points = result["entry_points"]
        assert len(entry_points) == 12
        for key, meta in entry_points.items():
            assert meta["distribution"], key
            assert meta["version"], key
            assert len(meta["sha256"]) == 64, key
        assert len(result["entry_point_metadata_digest"]) == 64

    def test_missing_registration_refuses(self, monkeypatch):
        import agent_plugins.dispatch_authorization as mod
        import importlib.metadata as md

        def empty(group=None):
            return ()
        monkeypatch.setattr(md, "entry_points", empty)
        with pytest.raises(AuthorizationRefused,
                           match="NOT registered"):
            mod.resolve_required_entry_points(REPO)

    def test_duplicated_registration_refuses(self, monkeypatch):
        import agent_plugins.dispatch_authorization as mod
        import importlib.metadata as md
        real = md.entry_points

        def duplicated(group=None):
            eps = list(real(group=group))
            return eps + eps  # every name appears twice
        monkeypatch.setattr(md, "entry_points", duplicated)
        with pytest.raises(AuthorizationRefused, match="registered 2"):
            mod.resolve_required_entry_points(REPO)

    def test_resolution_outside_pinned_roots_refuses(
            self, monkeypatch, tmp_path):
        import agent_plugins.dispatch_authorization as mod
        rogue = tmp_path / "rogue_module.py"
        rogue.write_text("Plugin = object\n")

        import importlib

        real_import = importlib.import_module

        def hijack(name):
            module = real_import(name)
            if "rl_pipeline" in name:
                class Fake:
                    __file__ = str(rogue)
                return Fake
            return module
        monkeypatch.setattr(importlib, "import_module", hijack)
        with pytest.raises(AuthorizationRefused, match="OUTSIDE"):
            mod.resolve_required_entry_points(REPO)

    def test_executable_manifest_binds_entry_point_identity(self):
        manifest = executable_manifest(REPO)
        assert "entry_point_metadata" in manifest
        assert manifest["entry_point_metadata"] == \
            resolve_required_entry_points(REPO)[
                "entry_point_metadata_digest"]

    def test_bounded_forward_on_cpu(self):
        result = bounded_extractor_forward(REPO, "cpu")
        assert result["output_shape"] == [2, 96]

    @CUDA
    def test_bounded_forward_on_cuda(self):
        result = bounded_extractor_forward(REPO, "cuda")
        assert result["output_shape"] == [2, 96]


class TestDataSota383DeviceBinding:
    def _plan(self, tmp_path, **slot0):
        path = tmp_path / "binding.json"
        path.write_text(json.dumps({"slots": {"gpu_slot_0": slot0}}))
        return path

    def test_absent_plan_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        with pytest.raises(AuthorizationRefused, match="absent"):
            verify_device_binding(
                "gpu_slot_0",
                binding_path=tmp_path / "missing.json")

    def test_multiple_visible_devices_refuse(self, monkeypatch,
                                             tmp_path):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        with pytest.raises(AuthorizationRefused, match="exactly ONE"):
            verify_device_binding(
                "gpu_slot_0", binding_path=self._plan(
                    tmp_path, expected_device_class="RTX"))

    def test_unfilled_plan_refuses(self, monkeypatch, tmp_path):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name",
                            lambda i: "NVIDIA Test GPU")
        with pytest.raises(AuthorizationRefused, match="unfilled"):
            verify_device_binding(
                "gpu_slot_0", binding_path=self._plan(
                    tmp_path,
                    expected_device_class="<TO_BE_FILLED>"))

    def test_wrong_physical_class_refuses(self, monkeypatch,
                                          tmp_path):
        """The 383 counterexample: ordinal 0 was a different physical
        class for PyTorch than the slot assumed."""
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name",
                            lambda i: "NVIDIA GeForce RTX 5090")
        with pytest.raises(AuthorizationRefused,
                           match="wrong.*physical|does not match"):
            verify_device_binding(
                "gpu_slot_0", binding_path=self._plan(
                    tmp_path,
                    expected_device_class="RTX 5070 Ti"))

    def test_matching_class_passes_and_output_is_sanitized(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name",
                            lambda i: "NVIDIA GeForce RTX 5070 Ti")
        result = verify_device_binding(
            "gpu_slot_0", binding_path=self._plan(
                tmp_path, expected_device_class="RTX 5070 Ti"))
        assert result == {"logical_slot": "gpu_slot_0",
                          "device_class_sanitized":
                              "NVIDIA GeForce RTX 5070 Ti",
                          "local_identity_verified": False}

    def test_cudnn_micro_preflight_cpu_domain(self):
        result = cudnn_micro_preflight("cpu")
        assert result["conv2d_forward_backward_ok"] is True

    @CUDA
    def test_cudnn_micro_preflight_real_gpu(self):
        result = cudnn_micro_preflight("cuda")
        assert result["conv2d_forward_backward_ok"] is True

    def test_preflights_run_before_custody_reservation(self):
        source = (REPO / "tools/dispatch_paired_pretrain_comparison"
                  ".py").read_text()
        gpu_section = source.split("no CUDA visibility")[1]
        assert gpu_section.index("verify_device_binding") < \
            gpu_section.index("execute_cell")
        assert gpu_section.index("cudnn_micro_preflight") < \
            gpu_section.index("execute_cell")
