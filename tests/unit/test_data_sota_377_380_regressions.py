"""Adversarial regressions for DATA-SOTA-377/378/379/380 (final SAC
dispatch hardening order 2026-08-28). Model-free; no training, no GPU.
The 377 suite includes the auditor's exact counterexample: /etc/hosts
must never authorize."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.dispatch_authorization import (  # noqa: E402
    AUTHORIZATION_SCHEMA, AuthorizationRefused, executable_manifest,
    executable_manifest_digest, load_authorization,
    verify_authorization, verify_worktree_identity)
from tools.dispatch_paired_pretrain_comparison import (  # noqa: E402
    MANIFEST_DIR, DispatchRefused, attempt_inventory,
    build_cell_config, make_attempt_dir, verify_executable_identity,
    verify_slot_binding)

DESIGN = json.loads(
    (REPO / "docs/audits/evidence/"
     "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json").read_text())
TRIAL_IDS = [t["trial_id"] for t in DESIGN["trial_ledger"]]


def campaign_facts():
    from agent_plugins.branch_pretraining import sha256_file
    manifests = {t: sha256_file(MANIFEST_DIR / f"launch_{t}.json")
                 for t in TRIAL_IDS}
    return {
        "campaign_id": "paired_pretrain_sac_eth_o2022_20260828",
        "trial_ids": TRIAL_IDS,
        "paired_design_sha256": sha256_file(
            REPO / "docs/audits/evidence/"
            "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json"),
        "candidate_seal_manifest_sha256": DESIGN["shared_bindings"][
            "pretrain_generation"]["seal_manifest_sha256"],
        "launch_manifest_sha256": manifests,
        "executable_allowlist_sha256": executable_manifest_digest(
            executable_manifest(REPO)),
    }


def valid_authorization(facts, **overrides):
    auth = {
        "schema": AUTHORIZATION_SCHEMA,
        "campaign_id": facts["campaign_id"],
        "trial_ids": list(facts["trial_ids"]),
        "reviewed_correction_commit": "a" * 40,
        "paired_design_sha256": facts["paired_design_sha256"],
        "candidate_seal_manifest_sha256":
            facts["candidate_seal_manifest_sha256"],
        "launch_manifest_sha256": dict(
            facts["launch_manifest_sha256"]),
        "executable_allowlist_sha256":
            facts["executable_allowlist_sha256"],
        "authorization_scope": "EXECUTE_EIGHT_PAIRED_SAC_CELLS",
        "issued_utc": "2026-08-28T12:00:00Z",
        "auditor": "General Musashi",
        "audit_order_commit_digest": "b" * 40,
    }
    auth.update(overrides)
    return auth


def write_and_verify(tmp_path, facts, auth):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(auth))
    return verify_authorization(path, **facts)


class TestDataSota377TypedAuthorization:
    def test_etc_hosts_never_authorizes(self):
        """The auditor's exact counterexample: an unrelated existing
        file satisfied the old is_file() gate."""
        with pytest.raises(AuthorizationRefused,
                           match="generic file|does not parse"):
            load_authorization(Path("/etc/hosts"))

    def test_absent_file_refuses(self):
        with pytest.raises(AuthorizationRefused, match="not exist"):
            load_authorization(Path("/nonexistent/auth.json"))

    def test_unknown_key_refuses(self, tmp_path):
        facts = campaign_facts()
        auth = valid_authorization(facts)
        auth["extra_grant"] = "also allow other campaigns"
        with pytest.raises(AuthorizationRefused, match="unknown keys"):
            write_and_verify(tmp_path, facts, auth)

    def test_missing_field_refuses(self, tmp_path):
        facts = campaign_facts()
        auth = valid_authorization(facts)
        del auth["auditor"]
        with pytest.raises(AuthorizationRefused, match="missing"):
            write_and_verify(tmp_path, facts, auth)

    def test_wrong_campaign_refuses(self, tmp_path):
        facts = campaign_facts()
        auth = valid_authorization(facts,
                                   campaign_id="some_other_campaign")
        with pytest.raises(AuthorizationRefused, match="campaign_id"):
            write_and_verify(tmp_path, facts, auth)

    def test_stale_design_digest_refuses(self, tmp_path):
        facts = campaign_facts()
        auth = valid_authorization(
            facts, paired_design_sha256="0" * 64)
        with pytest.raises(AuthorizationRefused,
                           match="paired_design_sha256"):
            write_and_verify(tmp_path, facts, auth)

    def test_wrong_scope_and_auditor_refuse(self, tmp_path):
        facts = campaign_facts()
        for override in ({"authorization_scope": "EXECUTE_ANYTHING"},
                         {"auditor": "General Impostor"}):
            auth = valid_authorization(facts, **override)
            with pytest.raises(AuthorizationRefused):
                write_and_verify(tmp_path, facts, auth)

    def test_unfilled_template_refuses(self, tmp_path):
        facts = campaign_facts()
        auth = valid_authorization(
            facts,
            reviewed_correction_commit="<TO_BE_FILLED_BY_MUSASHI>")
        with pytest.raises(AuthorizationRefused, match="40-hex"):
            write_and_verify(tmp_path, facts, auth)

    def test_committed_template_never_authorizes(self):
        template = (REPO / "docs/audits/evidence/"
                    "PAIRED_SAC_DISPATCH_AUTHORIZATION_TEMPLATE_"
                    "2026_08_28.json")
        facts = campaign_facts()
        with pytest.raises(AuthorizationRefused):
            verify_authorization(template, **facts)

    def test_wrong_trial_set_and_manifest_digests_refuse(
            self, tmp_path):
        facts = campaign_facts()
        auth = valid_authorization(facts)
        auth["trial_ids"] = auth["trial_ids"][:-1] + ["rogue_cell"]
        with pytest.raises(AuthorizationRefused, match="trial set"):
            write_and_verify(tmp_path, facts, auth)
        auth = valid_authorization(facts)
        first = next(iter(auth["launch_manifest_sha256"]))
        auth["launch_manifest_sha256"][first] = "f" * 64
        with pytest.raises(AuthorizationRefused,
                           match="launch-manifest"):
            write_and_verify(tmp_path, facts, auth)

    def test_fully_valid_authorization_passes(self, tmp_path):
        facts = campaign_facts()
        auth = write_and_verify(tmp_path, facts,
                                valid_authorization(facts))
        assert auth["authorization_scope"] == \
            "EXECUTE_EIGHT_PAIRED_SAC_CELLS"


class TestDataSota378AttemptIsolation:
    def test_existing_attempt_dir_refuses(self, tmp_path):
        make_attempt_dir(tmp_path, "trial_x", "aaaa")
        with pytest.raises(DispatchRefused, match="already exists"):
            make_attempt_dir(tmp_path, "trial_x", "aaaa")

    def test_symlinked_trial_dir_refuses(self, tmp_path):
        real = tmp_path / "elsewhere"
        real.mkdir()
        (tmp_path / "trial_y").symlink_to(real)
        with pytest.raises(DispatchRefused, match="symlink"):
            make_attempt_dir(tmp_path, "trial_y", "bbbb")

    def test_fresh_nonces_get_sibling_directories(self, tmp_path):
        a = make_attempt_dir(tmp_path, "trial_z", "n1")
        b = make_attempt_dir(tmp_path, "trial_z", "n2")
        assert a != b and a.parent == b.parent
        (a / "model.zip").write_bytes(b"first attempt")
        assert not (b / "model.zip").exists()

    def test_all_artifact_paths_live_inside_the_attempt_dir(
            self, tmp_path):
        cell = dict(DESIGN["trial_ledger"][0]["genesis"])
        cell.update({
            "trial_id": DESIGN["trial_ledger"][0]["trial_id"],
            "pretrain_generation_seal": DESIGN["shared_bindings"][
                "pretrain_generation"]["seal_manifest_sha256"]})
        cfg = build_cell_config(DESIGN, cell, Path("/pretrain"),
                                tmp_path, device="cpu",
                                attempt_nonce="cafe")
        marker = f"{cell['trial_id']}/attempt_cafe/"
        for key in ("save_model", "results_file", "save_config",
                    "nested_split_dir"):
            assert marker in cfg[key], key

    def test_inventory_digests_and_exclusion(self, tmp_path):
        d = make_attempt_dir(tmp_path, "trial_i", "dddd")
        (d / "a.txt").write_text("alpha")
        (d / "sub").mkdir()
        (d / "sub/b.txt").write_text("beta")
        (d / "record.json").write_text("{}")
        inv = attempt_inventory(d, exclude={"record.json"})
        assert set(inv) == {"a.txt", "sub/b.txt"}
        assert all(len(v) == 64 for v in inv.values())


class TestDataSota379ExecutableIdentity:
    def test_manifest_covers_the_whole_executing_surface(self):
        manifest = executable_manifest(REPO)
        for name in ("driver", "nested_pipeline", "sac_agent",
                     "grouped_architecture",
                     "grouped_features_extractor",
                     "pretrained_branch_loader", "dispatch_custody",
                     "dispatch_authorization", "env_gym_fx",
                     "strategy_shared_execution_envelope",
                     "split_contract", "strong_config",
                     "cost_manifest", "design", "candidate_manifest",
                     "envelope_calibration"):
            assert name in manifest, name
        assert all(len(v) == 64 for v in manifest.values())

    def test_identity_drift_refuses(self):
        good = executable_manifest(REPO)
        tampered = dict(good)
        tampered["sac_agent"] = "0" * 64
        with pytest.raises(DispatchRefused, match="drift"):
            verify_executable_identity(
                {"executable_allowlist_sha256": tampered})
        with pytest.raises(DispatchRefused, match="no executable"):
            verify_executable_identity({})
        result = verify_executable_identity(
            {"executable_allowlist_sha256": good})
        assert result["executable_allowlist_digest"] == \
            executable_manifest_digest(good)

    def test_head_and_cleanliness_enforced(self, monkeypatch):
        import agent_plugins.dispatch_authorization as mod

        def fake_run(cmd, **kw):
            class R:
                stdout = ("c" * 40 + "\n") if "rev-parse" in cmd \
                    else " M tools/x.py\n"
            return R()
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        with pytest.raises(AuthorizationRefused, match="reviewed"):
            verify_worktree_identity(REPO, expected_commit="d" * 40)
        with pytest.raises(AuthorizationRefused, match="not clean"):
            verify_worktree_identity(REPO, expected_commit="c" * 40)

        def clean_run(cmd, **kw):
            class R:
                stdout = ("c" * 40 + "\n") if "rev-parse" in cmd \
                    else ""
            return R()
        monkeypatch.setattr(mod.subprocess, "run", clean_run)
        proof = verify_worktree_identity(REPO,
                                         expected_commit="c" * 40)
        assert proof == {"head": "c" * 40, "clean": True}

    def test_no_sidecar_authority(self):
        source = (REPO / "agent_plugins/dispatch_authorization.py"
                  ).read_text()
        assert ".sha256'" not in source and '.sha256"' not in source
        assert "read_bytes" in source  # digests come from bytes


class TestDataSota380SlotBinding:
    @staticmethod
    def _cell(trial_id):
        manifest = json.loads(
            (MANIFEST_DIR / f"launch_{trial_id}.json").read_text())
        return manifest["cell_genesis"]

    def test_correct_binding_passes(self):
        cell = self._cell("control_random_init_s101")
        binding = verify_slot_binding(cell, "gpu_slot_0")
        assert binding["within_slot_position"] == 0
        assert len(binding["manifest_sha256"]) == 64

    def test_wrong_slot_for_seed_refuses(self):
        cell = self._cell("control_random_init_s101")
        with pytest.raises(DispatchRefused, match="seed"):
            verify_slot_binding(cell, "gpu_slot_1")

    def test_unknown_slot_refuses(self):
        cell = self._cell("control_random_init_s101")
        with pytest.raises(DispatchRefused, match="unknown logical"):
            verify_slot_binding(cell, "gpu_slot_9")

    def test_tampered_genesis_refuses(self):
        cell = dict(self._cell("pretrained_finetuned_s202"))
        cell["genesis_sha256"] = "0" * 64
        with pytest.raises(DispatchRefused, match="genesis"):
            verify_slot_binding(cell, "gpu_slot_1")

    def test_driver_requires_slot_and_single_device(self):
        source = (REPO / "tools/dispatch_paired_pretrain_comparison"
                  ".py").read_text()
        assert "execution requires --logical-slot" in source
        assert "device_count() != 1" in source
        assert "get_device_name(0)" in source  # sanitized class only
        assert "uuid" not in source.lower()
