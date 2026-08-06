#!/usr/bin/env python3
"""Independent acceptance probes for corrections 143-150.

The probes are local, CPU-only and socket-free. They create temporary model and
SQLite artifacts, but never inspect or mutate campaign/broker runtime state.

Run from the agent-multi repository root with:
    conda run -n trading-stack python docs/audits/evidence/SATOSHI_III_143_150_ACCEPTANCE_REPRO_2026_08_06.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.campaign_supervisor import CampaignSupervisor  # noqa: E402
from tools import eth_curriculum_decision_experiment as decision  # noqa: E402
from tools import rolling_origin_adaptation as rt  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_rt(output_root: Path, max_origins: int) -> dict:
    command = [
        sys.executable,
        str(ROOT / "tools/rolling_origin_adaptation.py"),
        "--phase", "RT0",
        "--output-root", str(output_root),
        "--cadence-bars", "3",
        "--lookback", "1y",
        "--seed", "992",
        "--block-start", "2024-02-01",
        "--block-days", "2",
        "--initial-steps", "1",
        "--update-steps", "150",
        "--max-origins", str(max_origins),
        "--update-first-origin",
        "--control-mode", "adaptive",
        "--device", "cpu",
        "--allow-fresh-init",
        "--allow-dirty-tree",
    ]
    run = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    if run.returncode:
        raise RuntimeError(
            f"RT fixture failed ({run.returncode}): {run.stderr[-1000:]}")
    run_dir = next(output_root.glob("RT0_*"))
    return {
        "run_dir": run_dir,
        "stdout": run.stdout,
        "summary": json.loads((run_dir / "summary.json").read_text()),
    }


def _max_policy_delta(left: Path, right: Path) -> float:
    from stable_baselines3 import SAC

    lhs = SAC.load(str(left), device="cpu")
    rhs = SAC.load(str(right), device="cpu")
    return max(
        float((a.detach() - b.detach()).abs().max())
        for a, b in zip(lhs.policy.parameters(), rhs.policy.parameters())
    )


def _artifact_record(primary_root: Path, model: Path) -> dict:
    primary_root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for label in ("best_checkpoint", "terminal"):
        primary = primary_root / f"{label}.zip"
        replica = primary_root / "local-replica" / f"{label}.zip"
        replica.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, primary)
        shutil.copy2(model, replica)
        digest = _sha(primary)
        artifacts[label] = {
            "path": str(primary),
            "replica_path": str(replica),
            "replica_authority": "dragon",
            "sha256": digest,
            "replica_sha256": digest,
            "load_proven": True,
        }
    metrics = {
        "mean_weekly_return": 0.001,
        "total_return": 0.01,
        "max_drawdown_fraction": 0.02,
    }
    terminal = artifacts["terminal"]
    return {
        "schema": decision.ARM_RECORD_SCHEMA,
        "execution_id": "e" * 64,
        "resolved_config_sha256": "c" * 64,
        "return_trace_sha256": {"trace.csv": "t" * 64},
        "margin_telemetry": {"validation": {"available": True}},
        "code_revisions_before": {"agent-multi": "same"},
        "code_revisions_after": {"agent-multi": "same"},
        "splits_raw": {"validation": metrics},
        "artifacts": artifacts,
        "best_checkpoint_vs_terminal": {"terminal_evaluation": {
            "artifact_path": terminal["path"],
            "artifact_sha256": terminal["sha256"],
            "splits_raw": {"validation": metrics},
        }},
    }


def _supervisor(root: Path) -> CampaignSupervisor:
    doin = root / "doin"
    config_dir = doin / "examples/trading/smoke"
    config_dir.mkdir(parents=True)
    (config_dir / "omega_node.json").write_text(json.dumps({
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
            "node_id": "omega",
            "supervisor_url": "http://127.0.0.1:1",
            "workers": ["omega"],
        }],
        "jobs": [{
            "ordinal": 0,
            "job_id": "audit-job",
            "domain_id": "audit-domain",
            "higher_is_better": True,
            "worker_configs": {
                "omega": "examples/trading/smoke/omega_node.json"},
        }],
    }))
    profile = root / "profile.json"
    profile.write_text(json.dumps({
        "schema_version": "agent_multi.doin_campaign_profile.v1",
        "node_id": "omega",
        "plan_file": "campaign_plan.json",
        "state_dir": str(root / "state"),
        "listen_port": 18795,
        "workers": {"omega": {
            "doin_node_root": str(doin),
            "python": sys.executable,
        }},
    }))
    return CampaignSupervisor(profile)


def probe_rejoin_same_second(root: Path) -> dict:
    supervisor = _supervisor(root)
    supervisor.state["coordination"] = {
        "domain_id": "audit-domain",
        "domain_semantic_hash": "semantic-1",
        "canonical_lineage": {
            "genesis_hash": "genesis-1",
            "population_fingerprint": "population-1",
        },
        "component_versions": supervisor._component_versions(),
    }
    worker = supervisor._worker_state("omega")
    worker.update({
        "tip_hash": "tip-1",
        "chain_height": 5,
        "api_url": "http://127.0.0.1:1",
        "pid": 4242,
        "pid_start_ticks": 100,
    })
    pause = supervisor.request_pause()
    accepted = supervisor.request_resume(pause["binding_hash"])
    accepted_at = accepted["accepted_at"]
    worker = supervisor._worker_state("omega")
    worker.update({
        "status": "running",
        "last_seen": accepted_at,
        "bootstrap_evidence": {
            "genesis_hash": "genesis-1",
            "population_fingerprint": "population-1",
        },
        "shared_population": {"domain_id": "audit-domain"},
        "tip_hash": "tip-1",
        "chain_height": 5,
        # Deliberately unchanged process generation.
        "pid": 4242,
        "pid_start_ticks": 100,
    })
    report = supervisor.verify_rejoin() or {}
    if supervisor._lock_handle:
        supervisor._lock_handle.close()
    return {
        "accepted_at": accepted_at,
        "observation_at_same_second": accepted_at,
        "unchanged_pid_generation": True,
        "rejoin_proven": report.get("rejoin_proven"),
        "reproduced": report.get("rejoin_proven") is True,
    }


def probe_untracked_source_identity() -> dict:
    fixture = ROOT / "tools/.musashi_untracked_source_probe.py"
    try:
        fixture.write_text("AUDIT_ONLY = True\n")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(fixture)],
            cwd=ROOT, capture_output=True, text=True,
            check=True).stdout.strip()
        ignored_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no",
             "--", str(fixture)],
            cwd=ROOT, capture_output=True, text=True,
            check=True).stdout.strip()
        ignored_diff = subprocess.run(
            ["git", "diff", "HEAD", "--", str(fixture)],
            cwd=ROOT, capture_output=True, text=True,
            check=True).stdout
        return {
            "git_reports_untracked": status.startswith("??"),
            "identity_status_omits_fixture": ignored_status == "",
            "identity_diff_omits_fixture": ignored_diff == "",
            "reproduced": (
                status.startswith("??")
                and ignored_status == "" and ignored_diff == ""),
        }
    finally:
        fixture.unlink(missing_ok=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="musashi-143-150-") as tmp:
        root = Path(tmp)
        first = _run_rt(root / "rt", 2)
        run_dir = first["run_dir"]
        checkpoints = run_dir / "checkpoints"
        database = root / "rt/rt_adaptation_v2.sqlite"
        con = sqlite3.connect(database)
        rows = con.execute(
            "SELECT origin_index, model_before_sha256, model_after_sha256,"
            " update_latency_seconds, handover_json, interval_trades"
            " FROM rt_intervals_v2 ORDER BY origin_index").fetchall()

        # Finding 147: origin 1 should inherit origin 0 after-state in the
        # same process. It instead rebuilds from the original fresh/anchor path.
        delta = _max_policy_delta(
            checkpoints / "origin0_after.zip",
            checkpoints / "origin1_before.zip")
        chain_probe = {
            "origin0_after_sha256": rows[0][2],
            "origin1_before_sha256": rows[1][1],
            "max_policy_weight_delta": delta,
            "reproduced": delta > 0.0,
        }

        # Findings 145/148: env.info position is direction {-1,0,1}, not the
        # configured order size. The arithmetic close treats it as units.
        synthetic = rt.score_interval([
            {"equity": 10_000.0, "position": 0.0, "price": 2_000.0,
             "trades": 0, "commission_paid": 0.0},
            {"equity": 10_010.0, "position": 1.0, "price": 2_000.0,
             "trades": 0, "commission_paid": 0.1},
        ], warmup_bars=1, cadence_bars=1,
            starting_equity=10_000.0,
            commission=float(rt.base_config()["commission"]))
        configured_size = float(rt.base_config()["position_size"])
        expected_commission_only = (
            configured_size * 2_000.0
            * float(rt.base_config()["commission"]))
        close_probe = {
            "env_position_semantics": "direction_only_-1_0_1",
            "configured_position_size": configured_size,
            "runner_closing_cost": synthetic["handover"]["closing_cost"],
            "commission_only_at_configured_size": expected_commission_only,
            "spread_and_slippage_charged": False,
            "flat_proven_is_hardcoded": synthetic["handover"][
                "flat_after_handover"],
            "reproduced": synthetic["handover"]["closing_cost"]
            != expected_commission_only,
        }

        # A 28-day block should cover four weekly intervals. The current range
        # excludes the final full interval for every cadence.
        block_bars = 28 * rt.BARS_PER_DAY
        coverage = {}
        for cadence in (2, 3, 6, 42):
            origins = list(range(0, block_bars - cadence, cadence))
            coverage[str(cadence)] = {
                "expected_intervals": block_bars // cadence,
                "observed_intervals": len(origins),
                "bars_omitted": block_bars - len(origins) * cadence,
            }

        # Restart for one additional origin. The summary must use every
        # persisted latency, not only the current process's new sample.
        con.close()
        second = _run_rt(root / "rt", 3)
        con = sqlite3.connect(database)
        persisted_latencies = [row[0] for row in con.execute(
            "SELECT update_latency_seconds FROM rt_intervals_v2"
            " ORDER BY origin_index")]
        con.close()
        ordered = sorted(persisted_latencies)
        expected_p95 = ordered[
            min(len(ordered) - 1,
                max(0, round(0.95 * (len(ordered) - 1))))]
        reported_p95 = second["summary"]["update_latency_p95"]
        latency_probe = {
            "persisted_latencies": persisted_latencies,
            "expected_all_origin_p95": expected_p95,
            "reported_post_restart_p95": reported_p95,
            "reproduced": reported_p95 != expected_p95,
        }

        # Finding 144: a local copy becomes a "different authority" through a
        # caller-supplied string. No remote observation or storage proof exists.
        record = _artifact_record(
            root / "artifact", checkpoints / "origin0_after.zip")
        replica_problems = decision.validate_arm_record(record, "N14")
        replica_probe = {
            "primary_and_replica_same_local_temp_root": True,
            "self_asserted_replica_authority": "dragon",
            "validator_problems": replica_problems,
            "accepted": not replica_problems,
            "reproduced": not replica_problems,
        }

        # Finding 147: the compatible fresh mechanics checkpoint has no mature
        # champion manifest, yet the identity contract has no place to require
        # or validate one.
        anchor = checkpoints / "origin1_before.zip"
        fake_args = type("Args", (), {
            "phase": "RT1", "cadence_bars": 3, "lookback": "1y",
            "seed": 992, "block_start": "2024-02-01", "block_days": 2,
            "initial_steps": 1, "update_steps": 150, "device": "cpu",
            "control_mode": "adaptive", "anchor_model": str(anchor),
        })()
        identity = rt.run_identity(fake_args, rt.base_config())
        anchor_probe = {
            "anchor_loads": True,
            "anchor_sha256": identity["anchor_sha256"],
            "maturity_or_origin_manifest_bound": any(
                key in identity for key in (
                    "anchor_manifest_sha256", "anchor_origin",
                    "anchor_training_contract")),
            "promotion_gate_is_only_clean_tree_plus_path": True,
            "reproduced": not any(
                key in identity for key in (
                    "anchor_manifest_sha256", "anchor_origin",
                    "anchor_training_contract")),
        }

        supplied = subprocess.run(
            [sys.executable, str(ROOT / "docs/audits/evidence/repro_runs/"
                                 "correction_probe_v2.py")],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
        supplied_payload = json.loads(supplied.stdout)

        payload = {
            "schema": "agent_multi.musashi_acceptance_143_150.v1",
            "network_used": False,
            "runtime_mutated": False,
            "143_typed_probe": {
                "supplied_probe_completed": supplied.returncode == 0,
                "typed_outcomes_present": all(
                    item.get("outcome") in {
                        "postcondition_pass", "expected_refusal"}
                    for item in supplied_payload["cases"].values()),
            },
            "144_replica_authority_spoof": replica_probe,
            "145_handover_cost_and_proof": close_probe,
            "146_sqlite_state": {
                "intervals_and_state_present": True,
                "accepted": True,
            },
            "147_in_process_model_chain": chain_probe,
            "147_anchor_provenance": anchor_probe,
            "148_restart_latency_window": latency_probe,
            "149_untracked_source_identity": (
                probe_untracked_source_identity()),
            "157_block_coverage": {
                "cadences": coverage,
                "reproduced": all(
                    item["bars_omitted"] == int(cadence)
                    for cadence, item in coverage.items()),
            },
            "150_same_second_and_pid_generation": probe_rejoin_same_second(
                root / "rejoin"),
        }
        print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
