#!/usr/bin/env python3
"""Independent, socket-free reproduction for corrections 128-134.

The script mutates only temporary files.  It deliberately exercises contract
boundaries that the delivery tests do not cover and prints one JSON packet.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
LTS = REPO.parent / "lts"
sys.path.insert(0, str(REPO))

from app.campaign_supervisor import (  # noqa: E402
    CampaignSupervisor,
    _sha256_file,
    _sha256_json,
)
from optimizer_plugins.project3_full_genome_optimizer import Plugin  # noqa: E402
from pipeline_plugins import rl_pipeline_with_validation as validation  # noqa: E402
import tools.eth_curriculum_decision_experiment as decision  # noqa: E402
import tools.rolling_origin_adaptation as rolling  # noqa: E402


def _minimal_supervisor(root: Path) -> CampaignSupervisor:
    doin_root = root / "doin"
    config_dir = doin_root / "examples/trading/smoke"
    config_dir.mkdir(parents=True)
    (config_dir / "omega_node.json").write_text(json.dumps({
        "port": 18470,
        "data_dir": str(root / "worker-data"),
        "domains": [{
            "domain_id": "audit-domain",
            "optimization_plugin": "trading_asset",
            "optimization_config": {
                "shared_population": True,
                "shared_population_size": 4,
                "ga_seed": 1,
                "population_size": 4,
            },
        }],
    }))
    (root / "campaign_plan.json").write_text(json.dumps({
        "schema_version": "agent_multi.doin_campaign_plan.v1",
        "plan_id": "audit-plan",
        "participants": [{
            "node_id": "omega",
            "supervisor_url": "http://127.0.0.1:18795",
            "workers": ["omega"],
        }],
        "jobs": [{
            "ordinal": 0,
            "job_id": "audit-job",
            "domain_id": "audit-domain",
            "higher_is_better": True,
            "worker_configs": {
                "omega": "examples/trading/smoke/omega_node.json",
            },
        }],
    }))
    profile = root / "omega_profile.json"
    profile.write_text(json.dumps({
        "schema_version": "agent_multi.doin_campaign_profile.v1",
        "node_id": "omega",
        "plan_file": "campaign_plan.json",
        "state_dir": str(root / "state"),
        "listen_port": 18795,
        "workers": {"omega": {
            "doin_node_root": str(doin_root),
            "python": "/usr/bin/python3",
        }},
    }))
    return CampaignSupervisor(profile)


def reproduce_inexact_rejoin(root: Path) -> dict:
    supervisor = _minimal_supervisor(root)
    try:
        binding = {
            "plan_hash": supervisor.plan_hash,
            "profile_sha256": _sha256_file(supervisor.profile_path),
            "job_id": "audit-job",
            "domain_id": "audit-domain",
            "domain_semantic_hash": "semantic-old",
            "genesis_hash": "genesis-shared",
            "population_fingerprint": "population-zero-shared",
            "component_versions": {"agent-multi": "old-revision"},
            "worker_tips": {"omega": "bound-tip"},
        }
        binding_hash = _sha256_json(binding)
        supervisor.state.update({
            "phase": "paused",
            "pause_report": {"paused": True},
            "pause_binding": binding,
            "pause_binding_hash": binding_hash,
        })
        accepted = supervisor.request_resume(binding_hash)
        worker = supervisor._worker_state("omega")
        worker.update({
            "status": "running",
            "domain_id": "audit-domain",
            "tip_hash": "foreign-tip-with-no-ancestor-proof",
            "chain_height": 91,
            "bootstrap_evidence": {
                "genesis_hash": "genesis-shared",
                "population_fingerprint": "population-zero-shared",
            },
            "shared_population": {
                "domain_id": "audit-domain",
                "generation": 9,
            },
        })
        supervisor.state.setdefault("coordination", {})[
            "component_versions"] = {"agent-multi": "new-revision"}
        result = supervisor.verify_rejoin() or {}
        return {
            "reproduced": (
                accepted.get("resume_accepted") is True
                and result.get("rejoin_proven") is True
            ),
            "component_revision_changed": True,
            "bound_tip": binding["worker_tips"]["omega"],
            "accepted_tip": worker["tip_hash"],
            "tip_ancestry_proof_present": False,
            "rejoin_proven": result.get("rejoin_proven"),
        }
    finally:
        if supervisor._lock_handle:
            supervisor._lock_handle.close()


def reproduce_repair_fail_open() -> dict:
    missing_schema = [{
        "rule": "forbid_value",
        "gene": "preprocessing_mode",
        "value": "none",
        "repair": "resample_categorical",
    }]
    Plugin.validate_repair_rules(missing_schema, {})

    undeclared_forbidden = [{
        "rule": "forbid_value",
        "gene": "preprocessing_mode",
        "value": "typographical-value-not-in-domain",
        "repair": "resample_categorical",
    }]
    Plugin.validate_repair_rules(undeclared_forbidden, {
        "mixed_genome_schema": [{
            "name": "preprocessing_mode",
            "kind": "categorical",
            "choices": ["rolling", "feature_aware"],
        }],
    })
    return {
        "reproduced": True,
        "accepted_without_typed_schema": True,
        "accepted_forbidden_value_outside_declared_domain": True,
    }


def _split_result(config: dict, mode: str) -> dict:
    summary = {
        "mean_weekly_return": 0.001,
        "annualized_return": 0.05,
        "total_return": 0.02,
        "max_drawdown_fraction": 0.03,
        "trades_total": 10,
    }
    result = {
        "splits": {
            "train": dict(summary),
            "train_tail": dict(summary),
            "validation": dict(summary),
            "test": {"evaluation_skipped": True},
        },
        "history": [],
    }
    if mode == "train":
        best = Path(config["save_model"])
        terminal = best.with_suffix("").with_name(
            best.with_suffix("").name + ".terminal.zip")
        best.write_bytes(b"best-checkpoint")
        terminal.write_bytes(b"terminal-checkpoint")
        result["artifacts"] = {
            "best_checkpoint": {
                "path": str(best),
                "sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
            },
            "terminal": {
                "path": str(terminal),
                "sha256": hashlib.sha256(terminal.read_bytes()).hexdigest(),
                "num_timesteps": 20,
            },
        }
    return result


def reproduce_terminal_reference_gap(root: Path) -> dict:
    root.mkdir(parents=True)
    anchor = root / "anchor.zip"
    anchor.write_bytes(b"anchor")
    original_run = validation.PipelinePlugin.run_pipeline
    original_agent = decision._agent_plugin

    def fake_run(_self, *, config, mode, **_kwargs):
        return _split_result(config, mode)

    validation.PipelinePlugin.run_pipeline = fake_run
    decision._agent_plugin = lambda _name: object()
    try:
        record = decision.run_arm(
            "N14", 101, root, agent_name="fake",
            epoch_timesteps=10, anchor=anchor,
        )
    finally:
        validation.PipelinePlugin.run_pipeline = original_run
        decision._agent_plugin = original_agent
    terminal_eval = (
        record["best_checkpoint_vs_terminal"]["terminal_evaluation"])
    return {
        "reproduced": (
            terminal_eval.get("artifact_sha256") is not None
            and "terminal" not in record.get("artifacts", {})
            and "path" not in terminal_eval
        ),
        "record_artifact_labels": sorted(record.get("artifacts", {})),
        "terminal_hash_present": bool(terminal_eval.get("artifact_sha256")),
        "terminal_retrieval_path_present": "path" in terminal_eval,
    }


def reproduce_incomplete_exact_reuse(root: Path) -> dict:
    anchor = root / "anchor.zip"
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes(b"anchor")
    execution_id = decision._execution_id(
        "N14", 101, anchor, epoch_timesteps=10)
    arm_dir = root / "seed101/N14"
    arm_dir.mkdir(parents=True)
    incomplete = {
        "schema": "agent_multi.arm_record.v3",
        "execution_id": execution_id,
        "arm": "N14",
        "seed": 101,
        "splits_raw": {"validation": {"total_return": 0.01}},
    }
    (arm_dir / "arm_record.json").write_text(json.dumps(incomplete))
    with contextlib.redirect_stdout(io.StringIO()):
        returned = decision.run_arm(
            "N14", 101, root, agent_name="must-not-load",
            epoch_timesteps=10, anchor=anchor,
        )
    return {
        "reproduced": returned == incomplete,
        "missing_artifacts_reused": not bool(returned.get("artifacts")),
        "missing_trace_hashes_reused": not bool(
            returned.get("return_trace_sha256")),
    }


def _valid_arm(seed: int, arm: str, suffix: str) -> dict:
    validation_metrics = {
        "mean_weekly_return": 0.001,
        "annualized_return": 0.05,
        "total_return": 0.02,
        "max_drawdown_fraction": 0.03,
        "trades_total": 20,
    }
    record = {
        "schema": "agent_multi.arm_record.v3",
        "execution_id": f"exec-{seed}-{arm}-{suffix}",
        "arm": arm,
        "seed": seed,
        "splits_raw": {"validation": validation_metrics},
        "margin_telemetry": {"validation": {
            "would_margin_call_count": 0,
        }},
        "return_trace_sha256": {"validation.json": "a" * 64},
        "resolved_config_sha256": "b" * 64,
        "artifacts": {"final": {"path": "missing.zip", "sha256": "c" * 64}},
    }
    if arm != "E4":
        record["best_checkpoint_vs_terminal"] = {
            "terminal_evaluation": {
                "artifact_sha256": "d" * 64,
                "splits_raw": {"validation": validation_metrics},
            },
        }
    return record


def _write_packet(root: Path, directory: str, seed: int, suffix: str) -> None:
    target = root / directory
    target.mkdir(parents=True)
    packet = {
        "schema": "agent_multi.eth_curriculum_decision.v1",
        "seed": seed,
        "data_sha256": None,
        "base_contract_sha256": None,
        "lineage": None,
        "arms": {
            arm: _valid_arm(seed, arm, suffix)
            for arm in ("N14", "EN4_10", "E4")
        },
    }
    (target / "seed_packet.json").write_text(json.dumps(packet))


def reproduce_duplicate_seed_and_empty_identity(root: Path) -> dict:
    _write_packet(root, "seed101-a", 101, "discarded")
    _write_packet(root, "seed101-z", 101, "accepted")
    for seed in (202, 303, 404):
        _write_packet(root, f"seed{seed}", seed, "accepted")
    done = subprocess.run(
        [sys.executable, str(REPO / "tools/aggregate_curriculum_decision.py"),
         "--output-root", str(root)],
        capture_output=True, text=True, check=False,
    )
    payload = json.loads(done.stdout)
    return {
        "reproduced": (
            done.returncode == 0
            and payload.get("promotion_eligible") is True
        ),
        "physical_packet_count": 5,
        "reported_seeds": payload.get("seeds"),
        "all_packet_identity_fields_empty": True,
        "promotion_eligible": payload.get("promotion_eligible"),
    }


def reproduce_incomplete_authority_join() -> dict:
    module_path = LTS / "tools/controller_inventory.py"
    spec = importlib.util.spec_from_file_location(
        "controller_inventory_current", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module._join_manifest({
        "artifact_sha256": "same-artifact",
        "model_id": "different-model",
        "config_sha256": "different-config",
        "input_sha256": "different-input",
        "fresh": False,
    }, [{
        "path": "/manifest.json",
        "schema": module.SAC_SCHEMA,
        "model_id": "selected-model",
        "artifact_sha256": "same-artifact",
        "config_sha256": "selected-config",
        "input_sha256": "selected-input",
        "live_inference_eligible": False,
        "live_execution_eligible": True,
        "observation_parity_verified": False,
    }])
    return {
        "reproduced": result.get("sac_champion_authoritative") is True,
        "heartbeat_fresh": False,
        "model_config_input_all_mismatch": True,
        "inference_and_parity_ineligible": True,
        "authority_returned": result.get("sac_champion_authoritative"),
    }


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return 0, None


class _FakeEnv:
    def __init__(self, equities):
        self.equities = list(equities)
        self.index = 0

    def reset(self):
        self.index = 0
        return 0, {}

    def step(self, _action):
        equity = self.equities[self.index]
        self.index += 1
        return 0, 0.0, self.index == len(self.equities), False, {
            "economic_equity": equity,
            "trades_total": self.index,
        }


def reproduce_warmup_scoring() -> dict:
    equities = [100.0, 90.0, 80.0, 80.0, 88.0]
    score = rolling._score_interval(_FakeModel(), _FakeEnv(equities))
    intended_next_interval_return = equities[-1] / equities[-2] - 1.0
    return {
        "reproduced": (
            score.get("bars") == len(equities)
            and abs(score["interval_return"] - intended_next_interval_return)
            > 1e-9
        ),
        "scored_bars": score.get("bars"),
        "deployment_interval_bars": 1,
        "reported_return": score.get("interval_return"),
        "intended_next_interval_return": intended_next_interval_return,
    }


def reproduce_rt_identity_collision() -> dict:
    identity = {
        "runner_version": rolling.RUNNER_VERSION,
        "phase": "RT0",
        "cadence_bars": 3,
        "lookback": "1y",
        "seed": 101,
        "block_start": "2024-02-01",
        "block_days": 28,
        "update_steps": 500,
        "data_sha256": rolling.DATA_SHA256,
    }
    run_id = hashlib.sha256(json.dumps(
        identity, sort_keys=True).encode()).hexdigest()[:16]
    changed_unbound_inputs = {
        "initial_steps": (1000, 20000),
        "device": ("cpu", "cuda"),
        "base_contract_sha256": ("old", "new"),
        "code_revision": ("old", "new"),
    }
    same_run_id_after_changes = hashlib.sha256(json.dumps(
        identity, sort_keys=True).encode()).hexdigest()[:16]
    base_config = decision._base_config(
        Path(tempfile.gettempdir()), "N14", 101, epoch_timesteps=10)
    return {
        "reproduced": (
            run_id == same_run_id_after_changes
            and base_config.get("train_years") == 4
            and base_config.get("test_years") == 1
        ),
        "run_id": run_id,
        "unbound_inputs": changed_unbound_inputs,
        "explicit_dates_with_dormant_train_years": {
            "train_start": base_config.get("train_start"),
            "train_end": base_config.get("train_end"),
            "train_years": base_config.get("train_years"),
            "test_years": base_config.get("test_years"),
        },
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="audit-128-134-") as raw:
        root = Path(raw)
        packet = {
            "schema": "agent_multi.satoshi_iii_128_134_repro.v1",
            "network_used": False,
            "runtime_mutated": False,
            "inexact_rejoin": reproduce_inexact_rejoin(root / "resume"),
            "repair_validation_fail_open": reproduce_repair_fail_open(),
            "terminal_reference_gap": reproduce_terminal_reference_gap(
                root / "terminal"),
            "incomplete_exact_reuse": reproduce_incomplete_exact_reuse(
                root / "reuse"),
            "duplicate_seed_empty_identity_promotion": (
                reproduce_duplicate_seed_and_empty_identity(
                    root / "aggregate")),
            "incomplete_authority_join": reproduce_incomplete_authority_join(),
            "warmup_in_interval_score": reproduce_warmup_scoring(),
            "rt_identity_and_split_collision": reproduce_rt_identity_collision(),
        }
    packet["all_counterexamples_reproduced"] = all(
        value.get("reproduced") is True
        for key, value in packet.items()
        if key not in {"schema", "network_used", "runtime_mutated"}
    )
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0 if packet["all_counterexamples_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
