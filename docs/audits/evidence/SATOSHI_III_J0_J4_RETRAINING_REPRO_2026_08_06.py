#!/usr/bin/env python3
"""Independent counterexamples for the 2026-08-06 J0/J4 audit.

The script is socket-free and makes no network calls. It creates only temporary
files and prints one JSON evidence packet.
"""
from __future__ import annotations

import contextlib
import csv
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
DATA = (
    REPO.parent
    / "predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv"
)
sys.path.insert(0, str(REPO))

from app.campaign_supervisor import (  # noqa: E402
    CampaignSupervisor,
    _sha256_file,
    _sha256_json,
)
from optimizer_plugins.project3_full_genome_optimizer import Plugin  # noqa: E402
import tools.eth_curriculum_decision_experiment as runner  # noqa: E402
from pipeline_plugins import rl_pipeline_with_validation as validation  # noqa: E402


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


def reproduce_empty_lineage_rejoin(root: Path) -> dict:
    supervisor = _minimal_supervisor(root)
    try:
        binding = {
            "plan_hash": supervisor.plan_hash,
            "profile_sha256": _sha256_file(supervisor.profile_path),
            "job_id": "audit-job",
            "domain_id": None,
            "genesis_hash": None,
            "population_fingerprint": None,
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
            "bootstrap_evidence": {},
            "shared_population": {},
        })
        result = supervisor.verify_rejoin() or {}
        return {
            "reproduced": (
                accepted.get("resume_accepted") is True
                and result.get("rejoin_proven") is True
                and all(binding.get(key) is None for key in (
                    "domain_id", "genesis_hash", "population_fingerprint"))
            ),
            "resume_accepted": accepted.get("resume_accepted"),
            "rejoin_proven": result.get("rejoin_proven"),
            "proof": result.get("rejoin_proof"),
        }
    finally:
        if supervisor._lock_handle:
            supervisor._lock_handle.close()


def reproduce_unbound_repair_rule() -> dict:
    rules = [{
        "rule": "forbid_value",
        "gene": "gene_not_in_schema",
        "value": "none",
        "repair": "resample_categorical",
    }]
    Plugin.validate_repair_rules(rules, {"mixed_genome_schema": []})
    return {
        "reproduced": True,
        "validation_accepted_missing_gene": True,
    }


def _fake_split_result() -> dict:
    summary = {
        "mean_weekly_return": 0.001,
        "annualized_return": 0.05,
        "total_return": 0.02,
        "max_drawdown_fraction": 0.03,
        "trades_total": 10,
    }
    return {
        "splits": {
            "train": dict(summary),
            "train_tail": dict(summary),
            "validation": dict(summary),
            "test": {
                "evaluation_skipped": True,
                "skip_reason": "disabled",
            },
        },
        "history": [],
    }


def reproduce_missing_terminal_artifact(root: Path) -> dict:
    root.mkdir(parents=True)
    anchor = root / "anchor.zip"
    anchor.write_bytes(b"anchor")
    original_run = validation.PipelinePlugin.run_pipeline
    original_agent = runner._agent_plugin
    validation.PipelinePlugin.run_pipeline = lambda *a, **k: _fake_split_result()
    runner._agent_plugin = lambda name: object()
    try:
        record = runner.run_arm(
            "N14", 101, root, agent_name="fake",
            epoch_timesteps=10, anchor=anchor,
        )
    finally:
        validation.PipelinePlugin.run_pipeline = original_run
        runner._agent_plugin = original_agent
    terminal = (
        record.get("best_checkpoint_vs_terminal", {})
        .get("terminal_evaluation")
    )
    return {
        "reproduced": terminal is None,
        "terminal_evaluation": terminal,
        "artifact_labels": sorted((record.get("artifacts") or {}).keys()),
        "note_claims_both_evaluated": "both weight sets evaluated" in (
            record.get("best_checkpoint_vs_terminal", {}).get("note") or ""
        ),
    }


def reproduce_stale_arm_reuse(root: Path) -> dict:
    arm_dir = root / "seed101/N14"
    arm_dir.mkdir(parents=True)
    stale = {
        "arm": "N14",
        "seed": 101,
        "resolved_config_sha256": "stale-contract",
        "anchor_sha256": "stale-anchor",
        "splits_raw": {"validation": {"total_return": 0.01}},
    }
    (arm_dir / "arm_record.json").write_text(json.dumps(stale))
    with contextlib.redirect_stdout(io.StringIO()):
        result = runner.run_arm(
            "N14", 101, root, agent_name="must-not-load",
            epoch_timesteps=999999, anchor=Path("/missing/new-anchor.zip"),
        )
    return {
        "reproduced": result.get("resolved_config_sha256") == "stale-contract",
        "accepted_missing_new_anchor": True,
        "accepted_changed_epoch_timesteps": 999999,
    }


def reproduce_empty_promotion(root: Path) -> dict:
    for seed in (101, 202, 303, 404):
        seed_dir = root / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        arms = {}
        for arm in ("N14", "EN4_10", "E4"):
            arms[arm] = {
                "arm": arm,
                "seed": seed,
                "splits_raw": {"validation": {"garbage": 1}},
            }
        (seed_dir / "seed_packet.json").write_text(json.dumps({
            "seed": seed,
            "data_sha256": f"different-data-{seed}",
            "base_contract_sha256": f"different-contract-{seed}",
            "lineage": {"agent-multi": f"different-code-{seed}"},
            "arms": arms,
        }))
    done = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/aggregate_curriculum_decision.py"),
            "--output-root", str(root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(done.stdout)
    return {
        "reproduced": (
            done.returncode == 0
            and payload.get("promotion_eligible") is True
        ),
        "returncode": done.returncode,
        "promotion_eligible": payload.get("promotion_eligible"),
        "paired_mean_weekly_return": (
            payload.get("paired_differences_EN_minus_N", {})
            .get("mean_weekly_return")
        ),
    }


def reproduce_sac_classifier(root: Path) -> dict:
    root.mkdir(parents=True)
    module_path = LTS / "tools/controller_inventory.py"
    spec = importlib.util.spec_from_file_location("controller_inventory", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    heartbeat = root / "heartbeat.json"
    heartbeat.write_text(json.dumps({
        "observed_at": "2999-01-01T00:00:00+00:00",
        "state": "monitoring",
        "model_id": "selected-sac-v1",
        "artifact_sha256": "abc123",
    }))
    ledger = root / "ledger.sqlite"
    import sqlite3
    with sqlite3.connect(ledger) as con:
        con.execute(
            "CREATE TABLE due_bar_decisions ("
            "bar_close TEXT, model_id TEXT, artifact_sha256 TEXT, outcome TEXT)"
        )
        con.execute(
            "INSERT INTO due_bar_decisions VALUES (?, ?, ?, ?)",
            ("2026-01-01T00:00:00Z", "selected-sac-v1", "abc123", "decided"),
        )
    module.SEATS = {"paper": {
        "unit": "fake.service", "heartbeat": heartbeat, "ledger": ledger,
    }}
    module._unit_state = lambda unit: {
        "ActiveState": "active", "SubState": "running", "MainPID": "1"}
    module._sac_manifests = lambda: [{
        "model_id": "selected-sac-v1",
        "artifact_sha256": "abc123",
        "live_inference_eligible": True,
        "live_execution_eligible": True,
        "observation_parity_verified": True,
    }]
    prior_argv = sys.argv
    sys.argv = [str(module_path)]
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            code = module.main()
    finally:
        sys.argv = prior_argv
    payload = json.loads(stream.getvalue())
    seat = payload["seats"]["paper"]
    return {
        "reproduced": (
            code == 0
            and seat.get("sac_champion_authoritative") is False
            and seat.get("controller_type") == "unclassified"
        ),
        "controller_type": seat.get("controller_type"),
        "sac_champion_authoritative": seat.get("sac_champion_authoritative"),
    }


def data_contract() -> dict:
    base = json.loads(runner.ETH_BASE.read_text())
    feature_count = len(base["feature_columns"])
    window_bars = int(base["window_size"])
    scale_bars = int(base["feature_scaling_window"])
    auxiliary_window_values = 2 * window_bars if base.get(
        "include_price_window") else 0
    agent_state_values = 4 if base.get("include_agent_state") else 0
    by_year: dict[str, int] = {}
    total = 0
    with DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            year = row["DATE_TIME"][:4]
            by_year[year] = by_year.get(year, 0) + 1
            total += 1
    train = sum(count for year, count in by_year.items() if year <= "2023")
    validation_rows = by_year.get("2024", 0)
    test_rows = by_year.get("2025", 0)
    return {
        "path": str(DATA),
        "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "rows_total": total,
        "rows_by_year": dict(sorted(by_year.items())),
        "train_rows_2017_09_28_to_2023_12_31": train,
        "validation_rows_2024": validation_rows,
        "test_rows_2025": test_rows,
        "bar_hours": 4,
        "configured_train_years_overridden_by_explicit_dates": base.get(
            "train_years"),
        "feature_count": feature_count,
        "window_bars": window_bars,
        "window_hours": window_bars * 4,
        "rolling_scale_bars": scale_bars,
        "rolling_scale_hours": scale_bars * 4,
        "flattened_observation_values": (
            feature_count * window_bars
            + auxiliary_window_values
            + agent_state_values
        ),
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="j0-j4-audit-") as raw:
        root = Path(raw)
        packet = {
            "schema": "agent_multi.satoshi_iii_j0_j4_repro.v1",
            "network_used": False,
            "empty_lineage_rejoin": reproduce_empty_lineage_rejoin(
                root / "resume"),
            "repair_schema_gap": reproduce_unbound_repair_rule(),
            "terminal_artifact_gap": reproduce_missing_terminal_artifact(
                root / "terminal"),
            "stale_arm_reuse": reproduce_stale_arm_reuse(root / "stale"),
            "empty_packet_promotion": reproduce_empty_promotion(
                root / "aggregate"),
            "sac_classifier_false_negative": reproduce_sac_classifier(
                root / "inventory"),
            "data_contract": data_contract(),
        }
    packet["all_counterexamples_reproduced"] = all(
        value.get("reproduced") is True
        for key, value in packet.items()
        if key not in {"schema", "network_used", "data_contract"}
    )
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0 if packet["all_counterexamples_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
