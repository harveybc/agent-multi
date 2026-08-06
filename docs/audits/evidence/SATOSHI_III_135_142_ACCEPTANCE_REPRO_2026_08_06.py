#!/usr/bin/env python3
"""Independent acceptance probes for Satoshi III corrections 135-142.

Read-only with respect to campaign and broker state. Temporary files are used
for fixtures. The one simulator probe performs no training and opens no socket.

Run from the repository root with:
    conda run -n trading-stack python docs/audits/evidence/SATOSHI_III_135_142_ACCEPTANCE_REPRO_2026_08_06.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LTS = ROOT.parent / "lts"
sys.path.insert(0, str(ROOT))

from app.campaign_supervisor import CampaignSupervisor  # noqa: E402
from optimizer_plugins.project3_full_genome_optimizer import (  # noqa: E402
    Plugin as Project3FullGenomeOptimizer,
)
from tools import aggregate_curriculum_decision as aggregate  # noqa: E402
from tools import eth_curriculum_decision_experiment as decision  # noqa: E402
from tools import rolling_origin_adaptation as rt  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _supervisor(root: Path) -> CampaignSupervisor:
    doin = root / "doin"
    cfg = doin / "examples/trading/smoke"
    cfg.mkdir(parents=True)
    (cfg / "omega_node.json").write_text(json.dumps({
        "port": 18470,
        "data_dir": str(root / "worker-data"),
        "domains": [{
            "domain_id": "audit-domain",
            "optimization_plugin": "trading_asset",
            "optimization_config": {"shared_population": True},
        }],
    }))
    (root / "campaign_plan.json").write_text(json.dumps({
        "schema_version": "agent_multi.doin_campaign_plan.v1",
        "plan_id": "audit-plan",
        "participants": [{
            "node_id": "omega", "supervisor_url": "http://127.0.0.1:1",
            "workers": ["omega"],
        }],
        "jobs": [{
            "ordinal": 0, "job_id": "audit-job",
            "domain_id": "audit-domain", "higher_is_better": True,
            "worker_configs": {
                "omega": "examples/trading/smoke/omega_node.json"},
        }],
    }))
    profile = root / "profile.json"
    profile.write_text(json.dumps({
        "schema_version": "agent_multi.doin_campaign_profile.v1",
        "node_id": "omega", "plan_file": "campaign_plan.json",
        "state_dir": str(root / "state"), "listen_port": 18795,
        "workers": {"omega": {
            "doin_node_root": str(doin), "python": sys.executable}},
    }))
    return CampaignSupervisor(profile)


def probe_135_tip_ancestry(root: Path) -> dict:
    supervisor = _supervisor(root / "ancestry")
    worker = {
        "tip_hash": "descendant-tip", "chain_height": 8,
        "api_url": "http://worker.invalid",
    }
    bound = {"tip_hash": "bound-tip", "tip_index": 4,
             "chain_height": 5}
    module = sys.modules[CampaignSupervisor.__module__]
    original = module._http_json
    try:
        module._http_json = lambda *_args, **_kwargs: {"hash": "bound-tip"}
        descendant = supervisor._verify_tip_ancestry(
            "omega", worker, bound)
        module._http_json = lambda *_args, **_kwargs: {"hash": "foreign"}
        foreign = supervisor._verify_tip_ancestry("omega", worker, bound)
    finally:
        module._http_json = original
        if supervisor._lock_handle:
            supervisor._lock_handle.close()
    return {
        "descendant_proven": descendant.get("proven") is True,
        "foreign_refuted": foreign.get("contradiction") is True,
        "detail": {"descendant": descendant, "foreign": foreign},
    }


def probe_138_typed_repair() -> dict:
    config = {
        "mixed_genome_schema": [{
            "name": "preprocessing_mode", "kind": "categorical",
            "choices": ["none", "rolling_zscore", "expanding_zscore"],
        }],
        "mixed_genome_repair_rules": [{
            "rule": "forbid_value", "gene": "preprocessing_mode",
            "value": "none", "repair": "resample_categorical",
            "seed": 17,
        }],
    }
    Project3FullGenomeOptimizer.validate_repair_rules(
        config["mixed_genome_repair_rules"], config)
    decoded = {"preprocessing_mode": "none"}
    run_config = {}
    Project3FullGenomeOptimizer._apply_repair_rules(
        run_config, decoded, config)
    try:
        Project3FullGenomeOptimizer.validate_repair_rules(
            config["mixed_genome_repair_rules"], {})
        missing_schema_refused = False
    except ValueError:
        missing_schema_refused = True
    return {
        "repair_applied": decoded["preprocessing_mode"] != "none",
        "provenance_present": bool(run_config.get("_genome_repairs")),
        "missing_schema_refused": missing_schema_refused,
    }


def _record_with_fake_models(root: Path, arm: str = "N14") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    replica = root / "replica"
    replica.mkdir()
    artifacts = {}
    for label in ("best_checkpoint", "terminal"):
        path = root / f"{label}.zip"
        path.write_bytes(b"not-a-loadable-SAC-model")
        copy = replica / path.name
        copy.write_bytes(path.read_bytes())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[label] = {
            "path": str(path), "replica_path": str(copy),
            "sha256": digest, "replica_sha256": digest,
            "load_proven": True,
        }
    metrics = {"mean_weekly_return": 0.01, "total_return": 0.02,
               "max_drawdown_fraction": 0.03}
    return {
        "schema": decision.ARM_RECORD_SCHEMA,
        "execution_id": "e" * 64, "arm": arm, "seed": 101,
        "resolved_config_sha256": "c" * 64,
        "return_trace_sha256": {"trace": "d" * 64},
        "margin_telemetry": {"validation": {"count": 0}},
        "code_revisions_before": {"agent-multi": "rev"},
        "code_revisions_after": {"agent-multi": "rev"},
        "splits_raw": {"validation": metrics},
        "artifacts": artifacts,
        "best_checkpoint_vs_terminal": {"terminal_evaluation": {
            "artifact_sha256": "f" * 64,
            "artifact_path": "/definitely/missing/terminal.zip",
            "splits_raw": {"validation": metrics},
        }},
    }


def probe_136_artifact_validation(root: Path) -> dict:
    record = _record_with_fake_models(root / "fake-models")
    problems = decision.validate_arm_record(record, "N14")
    return {
        "reproduced": not problems,
        "validator_accepted_unloadable_bytes": not problems,
        "validator_accepted_mismatched_terminal_reference": not problems,
        "problems": problems,
    }


def probe_136_aggregator_filesystem_gap(root: Path) -> dict:
    tests = _load(ROOT / "tests/test_decision_experiment_contract.py",
                  "decision_contract_fixtures")
    packet_root = root / "aggregate"
    for seed in (101, 202, 303, 404):
        tests._write_packet(packet_root, seed)
    process = subprocess.run(
        [sys.executable, str(ROOT / "tools/aggregate_curriculum_decision.py"),
         "--output-root", str(packet_root)],
        capture_output=True, text=True, check=False)
    payload = json.loads(process.stdout)
    nonexistent = all(
        not (packet_root / f"seed{seed}" / "N14_term.zip").exists()
        for seed in (101, 202, 303, 404))
    return {
        "reproduced": process.returncode == 0
        and payload.get("promotion_eligible") is True and nonexistent,
        "returncode": process.returncode,
        "promotion_eligible": payload.get("promotion_eligible"),
        "all_terminal_paths_nonexistent": nonexistent,
    }


def probe_139_exact_authority() -> dict:
    inventory = _load(LTS / "tools/controller_inventory.py",
                      "controller_inventory_audit")
    heartbeat = {
        "fresh": True, "model_id": "eth-sac", "artifact_sha256": "a" * 64,
        "config_sha256": "c" * 64, "input_feature_sha256": "f" * 64,
        "preprocessing_sha256": "p" * 64, "manifest_sha256": "m" * 64,
    }
    manifest = dict(heartbeat)
    manifest.update({
        "schema": inventory.SAC_SCHEMA, "path": "/manifest.json",
        "live_inference_eligible": True,
        "live_execution_eligible": True,
        "observation_parity_verified": True,
    })
    exact = inventory._authority(
        {}, heartbeat, [manifest], {"ActiveState": "active"})
    stale = inventory._authority(
        {}, {**heartbeat, "fresh": False}, [manifest],
        {"ActiveState": "active"})
    return {
        "exact_granted": exact.get("sac_champion_authoritative") is True,
        "stale_refused": stale.get("sac_champion_authoritative") is not True,
        "detail": {"exact": exact, "stale": stale},
    }


class _BuyPolicy:
    def predict(self, _observation, deterministic=True):
        return 1, None


def probe_140_warmup_and_account_state(root: Path) -> dict:
    import pandas as pd

    frame = pd.read_csv(rt.DATA_FILE)
    dates = pd.to_datetime(frame["DATE_TIME"])
    origin = int((dates >= "2024-02-01").idxmax())
    cadence = 3
    path = rt._slice_csv(
        frame, origin - rt.WARMUP_BARS, origin + cadence,
        root / "eval.csv")
    env = rt._build_env(
        {**rt.base_config(), "eval_seed": 101}, path,
        starting_cash=10000.0)
    rollout = rt._rollout(_BuyPolicy(), env)
    base = env.unwrapped
    open_position = int(base.bridge.position)
    warmup_end = float(rollout["equities"][rt.WARMUP_BARS - 1])
    score = rt.score_interval(
        rollout["equities"], warmup_bars=rt.WARMUP_BARS,
        starting_equity=10000.0)
    env.close()
    return {
        "reproduced": (
            score["scored_bars"] == cadence + 1
            and warmup_end != 10000.0 and open_position != 0),
        "cadence_bars": cadence,
        "scored_bars": score["scored_bars"],
        "warmup_changed_equity": warmup_end != 10000.0,
        "warmup_end_equity": warmup_end,
        "reported_equity_before": score["equity_before"],
        "open_position_discarded_by_next_origin": open_position,
        "runner_persists_only": ["carried_equity", "model after path/hash"],
    }


def probe_141_commit_pointer_crash(root: Path) -> dict:
    database = root / "rt.sqlite"
    con = rt._olap(database)
    columns = [row[1] for row in con.execute(
        "PRAGMA table_info(rt_intervals_v2)")]
    values = {column: None for column in columns}
    values.update({
        "record_id": "record-0", "schema_version": rt.SCHEMA_VERSION,
        "run_id": "run", "origin_index": 0,
        "model_after_sha256": "a" * 64,
    })
    con.execute(
        f"INSERT INTO rt_intervals_v2 ({','.join(columns)})"
        f" VALUES ({','.join('?' * len(columns))})",
        [values[column] for column in columns])
    con.commit()  # exact crash point: OLAP durable, pointer still old
    pointer = {"origins_committed": [], "after_sha256": None,
               "after_path": None, "carried_equity": None}
    committed = con.execute(
        "SELECT model_after_sha256 FROM rt_intervals_v2"
        " WHERE record_id='record-0'").fetchone()
    mismatch_refused = pointer.get("after_sha256") not in (
        None, committed[0])
    con.close()
    return {
        "reproduced": not mismatch_refused,
        "olap_committed": committed[0] == "a" * 64,
        "pointer_still_empty": pointer["after_sha256"] is None,
        "current_restart_check_refuses_ambiguity": mismatch_refused,
        "consequence": (
            "origin is skipped but model path and carried equity remain old"),
    }


def probe_rt_identity_and_guard() -> dict:
    args = type("Args", (), {
        "phase": "RT0", "cadence_bars": 3, "lookback": "1y",
        "seed": 101, "block_start": "2024-02-01", "block_days": 28,
        "initial_steps": 2000, "update_steps": 500, "device": "cuda",
        "control_mode": "adaptive",
    })()
    identity = rt.run_identity(args, rt.base_config())
    summary = json.loads((
        ROOT / "docs/audits/evidence/repro_runs/rt0_v2_summary.json"
    ).read_text())
    guard = summary["deadline_guard"]
    source = (ROOT / "tools/rolling_origin_adaptation.py").read_text()
    return {
        "identity_has_starting_artifact": any(
            key in identity for key in
            ("starting_artifact_sha256", "anchor_sha256",
             "warm_start_model_sha256")),
        "identity_has_dirty_tree_digest": any(
            key in identity for key in
            ("source_tree_sha256", "worktree_sha256", "dirty_diff_sha256")),
        "runner_constructs_fresh_sac": 'SAC("MlpPolicy"' in source,
        "guard_claims_handover_condition": (
            "unreconciled handovers" in guard["rule"]),
        "guard_measures_handover_condition": any(
            "handover" in key for key in guard if key != "rule"),
    }


def probe_142_split_contract() -> dict:
    config = rt.base_config()
    decision_config = decision._base_config(
        Path("/tmp"), "N14", 101, epoch_timesteps=100)
    fields = set(rt.DORMANT_SPLIT_FIELDS)
    return {
        "rt_fields_removed": not (fields & set(config)),
        "decision_fields_removed": not (fields & set(decision_config)),
        "explicit_dates_present": all(
            key in decision_config for key in
            ("train_start", "train_end", "validation_start",
             "validation_end")),
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = {
            "schema": "agent_multi.musashi_acceptance_135_142.v1",
            "network_used": False,
            "runtime_mutated": False,
            "probes": {
                "135_tip_ancestry": probe_135_tip_ancestry(root),
                "136_artifact_load_and_reference":
                    probe_136_artifact_validation(root),
                "136_aggregator_filesystem":
                    probe_136_aggregator_filesystem_gap(root),
                "138_typed_repair": probe_138_typed_repair(),
                "139_exact_authority": probe_139_exact_authority(),
                "140_warmup_and_state":
                    probe_140_warmup_and_account_state(root),
                "141_commit_pointer_crash":
                    probe_141_commit_pointer_crash(root),
                "141_identity_and_deadline_guard":
                    probe_rt_identity_and_guard(),
                "142_split_contract": probe_142_split_contract(),
            },
        }
    print(json.dumps(payload, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
