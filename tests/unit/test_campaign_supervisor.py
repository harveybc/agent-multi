from __future__ import annotations

import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.campaign_supervisor import (
    PLAN_SCHEMA,
    PROFILE_SCHEMA,
    CampaignSupervisor,
    _candidate_duration_samples_from_log,
    _cmdline_references_config,
    _domain_semantic_hash,
)


DOMAIN_ID = "test-shared-domain"


def _node_config(port: int) -> dict:
    return {
        "node_label": "test",
        "port": port,
        "data_dir": "./state",
        "require_deterministic_seed": True,
        "domains": [
            {
                "domain_id": DOMAIN_ID,
                "optimize": True,
                "higher_is_better": True,
                "optimization_config": {
                    "shared_population": True,
                    "shared_population_size": 4,
                    "ga_seed": 1701,
                    "runtime_overlay": "machine.json",
                    "hyperparameter_bounds": {"x": [0.0, 1.0]},
                },
                "param_bounds": {"x": [0.0, 1.0]},
            }
        ],
    }


def _materialize(
    tmp_path: Path,
    *,
    participants: list[dict] | None = None,
    node_id: str = "omega",
) -> tuple[Path, dict, dict]:
    participants = participants or [
        {"node_id": "omega", "supervisor_url": "http://127.0.0.1:18795", "workers": ["omega"]}
    ]
    worker_ids = [worker for item in participants for worker in item["workers"]]
    doin_root = tmp_path / "doin-node"
    doin_root.mkdir()
    configs = {}
    semantic_hash = None
    for offset, worker_id in enumerate(worker_ids):
        value = _node_config(19000 + offset)
        value["node_label"] = worker_id
        value["domains"][0]["optimization_config"]["runtime_overlay"] = f"{worker_id}.json"
        path = doin_root / f"{worker_id}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        configs[worker_id] = path.name
        semantic_hash = _domain_semantic_hash(value)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "test-plan",
        "participants": participants,
        "jobs": [
            {
                "ordinal": 0,
                "job_id": "job-0",
                "domain_id": DOMAIN_ID,
                "higher_is_better": True,
                "domain_semantic_hash": semantic_hash,
                "worker_configs": configs,
            },
            {
                "ordinal": 1,
                "job_id": "job-1",
                "domain_id": DOMAIN_ID,
                "higher_is_better": True,
                "domain_semantic_hash": semantic_hash,
                "worker_configs": configs,
            },
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    local_workers = next(item["workers"] for item in participants if item["node_id"] == node_id)
    profile = {
        "schema_version": PROFILE_SCHEMA,
        "node_id": node_id,
        "plan_file": str(plan_path),
        "state_dir": str(tmp_path / f"state-{node_id}"),
        "workers": {
            worker_id: {
                "doin_node_root": str(doin_root),
                "python": "/usr/bin/python3",
            }
            for worker_id in local_workers
        },
    }
    profile_path = tmp_path / f"{node_id}.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile_path, plan, profile


def _worker_report(worker_id: str, *, tip: str = "abc", artifact: str = "model") -> dict:
    return {
        "worker_id": worker_id,
        "status": "running",
        "converged": True,
        "chain_height": 12,
        "tip_hash": tip,
        "finalized_height": 8,
        "finalized_hash": "finalized-anchor",
        "component_versions": {"agent-multi": "a1", "doin-node": "d1"},
        "stopped_verified": False,
        "artifact": artifact,
    }


def _participant_status(
    supervisor: CampaignSupervisor,
    node_id: str,
    workers: list[str],
    *,
    phase: str = "running",
    tip: str = "abc",
    artifact: str = "model",
) -> dict:
    return {
        "node_id": node_id,
        "plan_hash": supervisor.plan_hash,
        "job_id": "job-0",
        "phase": phase,
        "workers": {worker: _worker_report(worker, tip=tip) for worker in workers},
        "archive": {
            "chain_height": 12,
            "tip_hash": tip,
            "finalized_height": 8,
            "finalized_hash": "finalized-anchor",
            "champion": {"artifact_sha256": artifact},
        },
    }


def test_semantic_hash_ignores_machine_overlay_but_not_domain_changes():
    left = _node_config(1000)
    right = _node_config(2000)
    right["domains"][0]["optimization_config"]["runtime_overlay"] = "other.json"
    assert _domain_semantic_hash(left) == _domain_semantic_hash(right)
    right["domains"][0]["param_bounds"]["x"] = [0.0, 2.0]
    assert _domain_semantic_hash(left) != _domain_semantic_hash(right)


def test_semantic_hash_rejects_shared_population_seed_offsets():
    left = _node_config(1000)
    right = _node_config(2000)
    right["domains"][0]["optimization_config"]["runtime_overlay"] = "other.json"
    right["domains"][0]["optimization_config"]["node_seed_offset"] = 1
    assert _domain_semantic_hash(left) != _domain_semantic_hash(right)


def test_dataset_preflight_validates_runtime_path_and_manifest_hash(tmp_path: Path):
    profile_path, plan, profile = _materialize(tmp_path)
    agent_root = tmp_path / "agent-multi"
    data_root = tmp_path / "financial-data"
    dataset = data_root / "inputs" / "train.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("DATE_TIME,CLOSE\n2022-01-01T00:00:00,1.0\n", encoding="utf-8")
    manifest = agent_root / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({
            "asset": "TEST",
            "timeframe": "1h",
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    canonical = agent_root / "config.json"
    canonical.write_text(
        json.dumps({
            "data": {
                "asset": "TEST",
                "timeframe": "1h",
                "input_data_file": "${DATA_ROOT}/inputs/train.csv",
                "dataset_manifest_file": "${REPO_ROOT}/agent-multi/manifest.json",
            }
        }),
        encoding="utf-8",
    )
    overlay = agent_root / "overlay.json"
    overlay.write_text(
        json.dumps({
            "roots": {
                "DATA_ROOT": str(data_root),
                "REPO_ROOT": str(tmp_path),
            }
        }),
        encoding="utf-8",
    )
    doin_root = Path(profile["workers"]["omega"]["doin_node_root"])
    node_path = doin_root / plan["jobs"][0]["worker_configs"]["omega"]
    node = json.loads(node_path.read_text())
    node["domains"][0]["optimization_config"].update({
        "agent_multi_root": str(agent_root),
        "load_config": "config.json",
        "runtime_overlay": "overlay.json",
    })
    node_path.write_text(json.dumps(node), encoding="utf-8")
    plan["jobs"][0]["domain_semantic_hash"] = _domain_semantic_hash(node)
    Path(profile["plan_file"]).write_text(json.dumps(plan), encoding="utf-8")

    supervisor = CampaignSupervisor(profile_path)
    loaded = supervisor._validate_local_configs(supervisor.plan["jobs"][0])
    assert loaded["omega"]["dataset"]["sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()

    dataset.write_text("DATE_TIME,CLOSE\n2022-01-01T00:00:00,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset sha256"):
        supervisor._validate_local_configs(supervisor.plan["jobs"][0])


def test_process_config_matching_resolves_relative_path_from_process_cwd(tmp_path: Path):
    doin_root = tmp_path / "doin-node"
    target = doin_root / "examples" / "trading" / "campaign" / "omega_node.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    cmdline = [
        "/usr/bin/python3",
        "-m",
        "doin_node.cli",
        "examples/trading/campaign/omega_node.json",
    ]
    assert _cmdline_references_config(
        cmdline, process_cwd=doin_root, config_path=target
    )
    assert not _cmdline_references_config(
        cmdline,
        process_cwd=doin_root,
        config_path=target.with_name("dragon_node.json"),
    )


def test_plan_requires_every_worker_exactly_once(tmp_path: Path):
    profile_path, plan, _ = _materialize(tmp_path)
    plan["jobs"][0]["worker_configs"]["unexpected"] = "bad.json"
    Path(plan["participants"][0]["supervisor_url"] if False else tmp_path / "plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="map every worker exactly once"):
        CampaignSupervisor(profile_path)


def test_network_status_exposes_complete_optimization_queue(tmp_path: Path):
    profile_path, _, _ = _materialize(tmp_path)
    supervisor = CampaignSupervisor(profile_path)
    network = supervisor._network_status()

    assert [job["job_id"] for job in network["plan_jobs"]] == ["job-0", "job-1"]
    assert [job["status"] for job in network["plan_jobs"]] == ["starting", "queued"]


def test_candidate_duration_samples_pair_shared_log_entries(tmp_path: Path):
    log = tmp_path / "worker.log"
    log.write_text(
        "\n".join([
            "2026-07-17 10:00:00 [doin_node.unified] INFO: [SHARED] Evaluating candidate 2/20 gen=3 for domain",
            "2026-07-17 10:03:00 [doin_node.unified] INFO: [SHARED] Candidate 2/20 result: fitness=0.1 gen=3 domain",
            "2026-07-17 10:04:00 [doin_node.unified] INFO: [SHARED] Evaluating candidate 5/20 gen=3 for domain",
            "2026-07-17 10:09:00 [doin_node.unified] INFO: [SHARED] Candidate 5/20 result: fitness=0.2 gen=3 domain",
        ]) + "\n",
        encoding="utf-8",
    )

    assert _candidate_duration_samples_from_log(log) == [180.0, 300.0]


def test_network_eta_covers_current_campaign_and_queued_pool(tmp_path: Path):
    profile_path, _, _ = _materialize(tmp_path)
    supervisor = CampaignSupervisor(profile_path)
    common_worker = {
        "status": "running",
        "candidate": {
            "domain_id": DOMAIN_ID,
            "stage": 2,
            "gen_in_stage": 1,
        },
        "shared_population": {"pop_size": 4, "evaluated": 2},
        "optimization": {
            "domains": [{
                "domain_id": DOMAIN_ID,
                "stage_generations": [5, 5],
            }]
        },
    }
    first = {
        **common_worker,
        "candidate_eta": {
            "candidate_seconds_p25": 80.0,
            "candidate_seconds_median": 100.0,
            "candidate_seconds_p75": 120.0,
        },
    }
    second = {
        **common_worker,
        "candidate_eta": {
            "candidate_seconds_p25": 160.0,
            "candidate_seconds_median": 200.0,
            "candidate_seconds_p75": 240.0,
        },
    }
    participants = {
        "omega": {"online": True, "status": {"workers": {"omega": first}}},
        "dragon": {"online": True, "status": {"workers": {"dragon": second}}},
    }
    jobs = [
        {"ordinal": 0, "job_id": "job-0", "status": "running", "planned_candidates": 40},
        {"ordinal": 1, "job_id": "job-1", "status": "queued", "planned_candidates": 40},
    ]

    eta = supervisor._network_eta(participants, jobs, 0)

    assert eta["current_job_candidates_completed"] == 26
    assert eta["current_job_candidates_remaining"] == 14
    assert eta["pool_candidates_remaining"] == 54
    assert eta["swarm_candidates_per_hour"] == pytest.approx(54.0)
    assert eta["current_job_eta_seconds"] == pytest.approx(14 / 0.015)
    assert eta["pool_eta_seconds"] == pytest.approx(54 / 0.015)


def test_supervisor_restart_preserves_running_worker_component_contract(tmp_path: Path):
    profile_path, _, _ = _materialize(tmp_path)
    supervisor = CampaignSupervisor(profile_path)
    job = supervisor.plan["jobs"][0]
    configs = supervisor._validate_local_configs(job)
    original = supervisor._coordination_contract(job, configs)
    original["component_versions"] = {"agent-multi": "old-worker-commit"}
    original["contract_hash"] = "old-contract"
    supervisor.state.update({
        "phase": "running",
        "coordination": original,
        "workers": {
            "omega": {
                "status": "running",
                "last_chain_status": {
                    "component_versions": {"agent-multi": "old-worker-commit"},
                },
            }
        },
    })

    prepared = supervisor._prepare_coordination(job, configs)

    assert prepared["component_versions"] == {"agent-multi": "old-worker-commit"}
    assert prepared["contract_hash"] == "old-contract"


def test_supervisor_reanchors_accidentally_recomputed_contract_to_running_worker(tmp_path: Path):
    profile_path, _, _ = _materialize(tmp_path)
    supervisor = CampaignSupervisor(profile_path)
    job = supervisor.plan["jobs"][0]
    configs = supervisor._validate_local_configs(job)
    wrong = supervisor._coordination_contract(job, configs)
    wrong["component_versions"] = {"agent-multi": "new-supervisor-commit"}
    wrong["contract_hash"] = "wrong-contract"
    supervisor.state.update({
        "phase": "running",
        "coordination": wrong,
        "workers": {
            "omega": {
                "status": "running",
                "last_chain_status": {
                    "component_versions": {"agent-multi": "running-worker-commit"},
                },
            }
        },
    })

    prepared = supervisor._prepare_coordination(job, configs)

    assert prepared["component_versions"] == {"agent-multi": "running-worker-commit"}
    assert prepared["contract_hash"] != "wrong-contract"


def test_startup_barrier_launches_workers_in_global_order(tmp_path: Path):
    participants = [
        {"node_id": "omega", "supervisor_url": "http://omega:8795", "workers": ["omega"]},
        {"node_id": "dragon", "supervisor_url": "http://dragon:8795", "workers": ["dragon"]},
        {"node_id": "gamma", "supervisor_url": "http://gamma:8795", "workers": ["gamma-0", "gamma-1"]},
    ]
    profile_path, _, _ = _materialize(tmp_path, participants=participants)
    supervisor = CampaignSupervisor(profile_path)
    job = supervisor.plan["jobs"][0]
    configs = supervisor._validate_local_configs(job)
    contract = supervisor._prepare_coordination(job, configs)
    network = {"participants": {}}
    for participant in participants:
        network["participants"][participant["node_id"]] = {
            "online": True,
            "status": {
                "plan_hash": supervisor.plan_hash,
                "job_index": 0,
                "job_id": "job-0",
                "coordination": {
                    "preflight_ready": True,
                    "contract_hash": contract["contract_hash"],
                },
                "workers": {
                    worker_id: {"join_ready": False}
                    for worker_id in participant["workers"]
                },
            },
        }

    ready, reason = supervisor._worker_launch_ready(network, job, "omega")
    assert ready, reason
    ready, reason = supervisor._worker_launch_ready(network, job, "dragon")
    assert not ready
    assert "omega" in reason

    omega = network["participants"]["omega"]["status"]["workers"]["omega"]
    omega.update({
        "status": "running",
        "bootstrap_evidence": {
            "genesis_hash": "genesis",
            "population_block_hash": "population",
            "population_fingerprint": "fingerprint",
        },
    })
    ready, reason = supervisor._worker_launch_ready(network, job, "dragon")
    assert ready, reason
    ready, reason = supervisor._worker_launch_ready(network, job, "gamma-0")
    assert not ready
    assert "dragon" in reason

    dragon = network["participants"]["dragon"]["status"]["workers"]["dragon"]
    dragon.update({
        "status": "running",
        "bootstrap_evidence": dict(omega["bootstrap_evidence"]),
    })
    ready, reason = supervisor._worker_launch_ready(network, job, "gamma-0")
    assert ready, reason


def test_single_process_lock_rejects_second_supervisor(tmp_path: Path):
    profile_path, _, _ = _materialize(tmp_path)
    first = CampaignSupervisor(profile_path)
    second = CampaignSupervisor(profile_path)
    first._acquire_lock()
    with pytest.raises(RuntimeError, match="another supervisor"):
        second._acquire_lock()
    first._lock_handle.close()


def test_completion_barrier_requires_all_workers_same_tip_and_versions(tmp_path: Path):
    participants = [
        {"node_id": "omega", "supervisor_url": "http://omega:8795", "workers": ["omega"]},
        {"node_id": "dragon", "supervisor_url": "http://dragon:8795", "workers": ["dragon"]},
        {"node_id": "gamma", "supervisor_url": "http://gamma:8795", "workers": ["gamma-0", "gamma-1"]},
    ]
    profile_path, _, _ = _materialize(tmp_path, participants=participants)
    supervisor = CampaignSupervisor(profile_path)
    network = {
        "participants": {
            item["node_id"]: {
                "online": True,
                "status": _participant_status(supervisor, item["node_id"], item["workers"]),
            }
            for item in participants
        }
    }
    ready, _, _ = supervisor._completion_evidence(network, supervisor.plan["jobs"][0])
    assert ready
    network["participants"]["dragon"]["status"]["workers"]["dragon"]["tip_hash"] = "fork"
    ready, reason, _ = supervisor._completion_evidence(network, supervisor.plan["jobs"][0])
    assert ready
    assert "terminal forks" in reason
    network["participants"]["dragon"]["status"]["workers"]["dragon"]["finalized_hash"] = "other"
    ready, reason, _ = supervisor._completion_evidence(network, supervisor.plan["jobs"][0])
    assert not ready
    assert "finalized anchor" in reason
    network["participants"]["dragon"]["online"] = False
    ready, reason, _ = supervisor._completion_evidence(network, supervisor.plan["jobs"][0])
    assert not ready
    assert "offline" in reason


def test_runtime_health_detects_bootstrap_lineage_divergence(tmp_path: Path):
    participants = [
        {"node_id": "omega", "supervisor_url": "http://omega:8795", "workers": ["omega"]},
        {"node_id": "dragon", "supervisor_url": "http://dragon:8795", "workers": ["dragon"]},
    ]
    profile_path, _, _ = _materialize(tmp_path, participants=participants)
    supervisor = CampaignSupervisor(profile_path)
    network = {"participants": {}}
    for participant in participants:
        status = _participant_status(
            supervisor, participant["node_id"], participant["workers"]
        )
        for worker in status["workers"].values():
            worker.update({
                "bootstrap_evidence": {
                    "genesis_hash": "genesis",
                    "population_block_hash": "population",
                    "population_fingerprint": "fingerprint",
                },
                "shared_population": {
                    "generation": 0,
                    "pop_size": 4,
                },
            })
        network["participants"][participant["node_id"]] = {
            "online": True,
            "status": status,
        }

    healthy, issues = supervisor._runtime_swarm_health(
        network, supervisor.plan["jobs"][0]
    )
    assert healthy, issues
    network["participants"]["dragon"]["status"]["workers"]["dragon"][
        "bootstrap_evidence"
    ]["population_block_hash"] = "fork"
    healthy, issues = supervisor._runtime_swarm_health(
        network, supervisor.plan["jobs"][0]
    )
    assert not healthy
    assert "lineage" in "; ".join(issues)


def test_claimless_repair_is_suspended_while_swarm_is_unhealthy(tmp_path: Path):
    profile_path, _, _ = _materialize(tmp_path)
    supervisor = CampaignSupervisor(profile_path)
    worker = supervisor._worker_state("omega")
    worker.update({
        "status": "running",
        "owns_candidate": False,
        "shared_population": {"free": 4},
        "claimless_since": 1.0,
        "claimless_repair_count": 2,
    })

    supervisor._repair_claimless_local_workers(
        supervisor.plan["jobs"][0],
        {"omega": {"path": "unused.json"}},
        swarm_healthy=False,
    )

    assert worker["claimless_since"] is None
    assert worker["claimless_repair_count"] == 2


def test_stop_barrier_rejects_unverified_process_or_different_artifact(tmp_path: Path):
    participants = [
        {"node_id": "omega", "supervisor_url": "http://omega:8795", "workers": ["omega"]},
        {"node_id": "dragon", "supervisor_url": "http://dragon:8795", "workers": ["dragon"]},
    ]
    profile_path, _, _ = _materialize(tmp_path, participants=participants)
    supervisor = CampaignSupervisor(profile_path)
    network = {"participants": {}}
    for item in participants:
        status = _participant_status(
            supervisor, item["node_id"], item["workers"], phase="stopped"
        )
        for worker in status["workers"].values():
            worker["stopped_verified"] = True
        network["participants"][item["node_id"]] = {"online": True, "status": status}
    ready, _ = supervisor._stop_barrier_ready(network, supervisor.plan["jobs"][0])
    assert ready
    network["participants"]["dragon"]["status"]["archive"]["champion"]["artifact_sha256"] = "other"
    ready, reason = supervisor._stop_barrier_ready(network, supervisor.plan["jobs"][0])
    assert not ready
    assert "same champion" in reason
    network["participants"]["dragon"]["status"]["archive"]["champion"]["artifact_sha256"] = "model"
    network["participants"]["dragon"]["status"]["workers"]["dragon"]["stopped_verified"] = False
    ready, reason = supervisor._stop_barrier_ready(network, supervisor.plan["jobs"][0])
    assert not ready
    assert "stop is not verified" in reason


def test_stop_barrier_accepts_durable_ack_from_node_that_already_advanced(tmp_path: Path):
    participants = [
        {"node_id": "omega", "supervisor_url": "http://omega:8795", "workers": ["omega"]},
        {"node_id": "dragon", "supervisor_url": "http://dragon:8795", "workers": ["dragon"]},
    ]
    profile_path, _, _ = _materialize(tmp_path, participants=participants)
    supervisor = CampaignSupervisor(profile_path)
    omega = _participant_status(supervisor, "omega", ["omega"], phase="stopped")
    omega["workers"]["omega"]["stopped_verified"] = True
    dragon = _participant_status(supervisor, "dragon", ["dragon"], phase="running")
    dragon.update({
        "job_id": "job-1",
        "completed_campaigns": [{
            "job_id": "job-0",
            "chain_height": 12,
            "tip_hash": "abc",
            "artifact_sha256": "model",
        }],
    })
    network = {"participants": {
        "omega": {"online": True, "status": omega},
        "dragon": {"online": True, "status": dragon},
    }}
    ready, reason = supervisor._stop_barrier_ready(network, supervisor.plan["jobs"][0])
    assert ready, reason
    dragon["completed_campaigns"][0]["artifact_sha256"] = "other"
    ready, reason = supervisor._stop_barrier_ready(network, supervisor.plan["jobs"][0])
    assert not ready
    assert "same champion" in reason


class _ChainHandler(BaseHTTPRequestHandler):
    blocks: list[dict] = []

    def log_message(self, fmt, *args):
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/chain/status":
            value = {
                "chain_height": len(self.blocks),
                "tip_hash": self.blocks[-1]["hash"],
                "component_versions": {"agent-multi": "abc"},
            }
        elif self.path.startswith("/chain/block/"):
            value = self.blocks[int(self.path.rsplit("/", 1)[1])]
        else:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_champion_archive_extracts_and_verifies_embedded_model(tmp_path: Path):
    profile_path, _, _ = _materialize(tmp_path)
    supervisor = CampaignSupervisor(profile_path)
    model = b"stable-baselines-model"
    digest = hashlib.sha256(model).hexdigest()
    _ChainHandler.blocks = [
        {
            "hash": "genesis",
            "transactions": [],
        },
        {
            "hash": "tip",
            "transactions": [
                {
                    "id": "tx1",
                    "tx_type": "optimae_accepted",
                    "domain_id": DOMAIN_ID,
                    "peer_id": "dragon-peer",
                    "payload": {
                        "verified_performance": 0.25,
                        "parameters": {
                            "x": 0.4,
                            "_model_b64": base64.b64encode(model).decode(),
                        },
                        "champion_metrics": {
                            "risk_adjusted_total_return": 0.2,
                            "model_artifact_sha256": digest,
                            "model_artifact_bytes": len(model),
                            "model_artifact_format": "stable_baselines3_zip",
                        },
                    },
                }
            ],
        },
    ]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChainHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        archive = supervisor._archive_champion(
            supervisor.plan["jobs"][0], f"http://127.0.0.1:{server.server_port}"
        )
    finally:
        server.shutdown()
        server.server_close()
    champion = archive["champion"]
    assert champion["fitness"] == 0.25
    assert champion["artifact_sha256"] == digest
    assert Path(champion["artifact_path"]).read_bytes() == model
    manifest = json.loads(Path(champion["manifest_path"]).read_text())
    assert manifest["parameters"] == {"x": 0.4}


def test_state_recovery_keeps_same_job_and_plan(tmp_path: Path):
    profile_path, _, _ = _materialize(tmp_path)
    first = CampaignSupervisor(profile_path)
    first.state["phase"] = "stopped"
    first.state["archive"] = {"tip_hash": "abc"}
    first._save_state()
    recovered = CampaignSupervisor(profile_path)
    assert recovered.state["phase"] == "stopped"
    assert recovered.state["job_id"] == "job-0"
    assert recovered.state["archive"] == {"tip_hash": "abc"}
