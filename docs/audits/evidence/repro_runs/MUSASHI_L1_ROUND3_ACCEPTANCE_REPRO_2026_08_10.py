#!/usr/bin/env python3
"""Independent, read-only reproduction of the L1 round-3 packet."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools import aggregate_l1_factorial as aggregator  # noqa: E402
from tools import collect_l1_factorial as collector  # noqa: E402
from tools import l1_factorial_screen as runner  # noqa: E402

COLLECTION = Path.home() / (
    ".local/share/agent-multi/l1_smoke_collection_7aae0431"
)
SMOKE_ID = "7aae043107a87554"
DECISION_ID = "2de49ea9225e2baf"
EXECUTABLE_COMMIT = "f5e18696daa119edb5cee3103ea7ed1ab7f07094"
GYM_FX_COMMIT = "efa491600bdc9fee10efdfbe251474d63284a28b"
EXPECTED_GPU = {
    101: ("omega", "GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326"),
    202: ("dragon", "GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9"),
    303: ("gamma", "GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519"),
    404: ("gamma", "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"),
}


def ssh(host: str, command: str) -> str:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         host, command],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return proc.stdout.strip()


def host_read(host: str, path: str) -> str:
    if host == "omega":
        return Path(path).expanduser().read_text().strip()
    return ssh(host, f"cat {path}")


def host_exists(host: str, path: str) -> bool:
    if host == "omega":
        return Path(path).expanduser().exists()
    return ssh(host, f"test -e {path} && echo yes || echo no") == "yes"


def host_seed_dirs(host: str, path: str) -> list[str]:
    if host == "omega":
        root = Path(path).expanduser()
        return sorted(p.name for p in root.glob("seed*") if p.is_dir())
    output = ssh(
        host,
        f"test ! -d {path} || find {path} -maxdepth 1 -type d "
        "-name 'seed*' -printf '%f\\n' | sort",
    )
    return output.splitlines() if output else []


def main() -> int:
    manifest = json.loads((COLLECTION / "collection_manifest.json").read_text())
    sealed = COLLECTION / "sealed" / SMOKE_ID
    current_digest = collector.tree_digest(sealed)
    aggregation_path = (
        COLLECTION / "aggregations" / SMOKE_ID /
        "l1_factorial_aggregation.json"
    )
    aggregation = json.loads(aggregation_path.read_text())

    contract = runner.load_contract()
    system_manifest = runner.load_system_manifest()
    computed_decision_id = runner.experiment_identity(
        contract,
        system_manifest,
        False,
        sources={
            "agent-multi": {
                "commit": EXECUTABLE_COMMIT,
                "dirty_untracked_digest": None,
            },
            "gym-fx": {
                "commit": GYM_FX_COMMIT,
                "dirty_untracked_digest": None,
            },
        },
    )

    records = [
        json.loads(path.read_text())
        for path in sorted(sealed.rglob("l1_cell_record.json"))
    ]
    gpu_rows = {}
    epoch_rows = {}
    financing_rows = {}
    for record in records:
        key = f"{record['seed']}:{record['cell']}"
        gpu_rows[key] = record.get("gpu_binding")
        epoch_rows[key] = {
            "requested": record.get("phase1_requested_epochs"),
            "realized": record.get("phase1_realized_epochs"),
            "baseline": record.get("phase1_baseline_evaluations"),
        }
        financing_rows[key] = (
            (record.get("cost_contract") or {}).get("financing_treatment")
        )

    deployed_env = {}
    decision_roots = {}
    for seed, (host, uuid) in EXPECTED_GPU.items():
        env_path = f"/home/harveybc/.config/agent-multi/l1-factorial@{seed}.env"
        root = (
            "/home/harveybc/.local/share/agent-multi/"
            f"l1_matched_factorial_20260809_v1/{DECISION_ID}"
        )
        deployed_env[str(seed)] = {
            "host": host,
            "expected_uuid": uuid,
            "content": host_read(host, env_path),
        }
        decision_roots[str(seed)] = host_exists(host, root)

    decision_seed_dirs = {
        host: host_seed_dirs(
            host,
            "/home/harveybc/.local/share/agent-multi/"
            f"l1_matched_factorial_20260809_v1/{DECISION_ID}",
        )
        for host in ("omega", "dragon", "gamma")
    }
    expected_seed_dirs = {
        "omega": ["seed101"],
        "dragon": ["seed202"],
        "gamma": ["seed303", "seed404"],
    }
    decision_state_uncontaminated = (
        not any(decision_roots.values())
        or decision_seed_dirs == expected_seed_dirs
    )

    aggregate_source = inspect.getsource(aggregator.aggregate_from_collection)
    collector_source = inspect.getsource(collector.main)
    refusals = list(aggregation.get("refusals") or [])
    all_gpu_bound = all(
        row
        and row.get("assigned_gpu_uuid") == EXPECTED_GPU[int(key.split(":")[0])][1]
        and row.get("cuda_visible_devices") == EXPECTED_GPU[int(key.split(":")[0])][1]
        and row.get("torch_cuda_available") is True
        and row.get("torch_cuda_device_count") == 1
        for key, row in gpu_rows.items()
    )
    all_env_bound = all(
        row["content"] == (
            "L1_EXTRA_ARGS=\nCUDA_VISIBLE_DEVICES=" + row["expected_uuid"]
        )
        for row in deployed_env.values()
    )

    checks = {
        "196_seal_immutable_after_aggregation": (
            manifest["collection_tree_digest"] == current_digest
            == manifest["replica"]["replica_tree_digest"]
            and not any(sealed.rglob("l1_factorial_aggregation.json"))
            and aggregation_path.is_file()
        ),
        "197_single_collection_authority": (
            "load_collection_envelope" in aggregate_source
            and "aggregate_from_collection" in collector_source
            and aggregation.get("collection_envelope", {}).get(
                "sealed_input_digest"
            ) == current_digest
        ),
        "198_exact_gpu_binding_in_records_and_env": (
            len(records) == 16 and all_gpu_bound and all_env_bound
        ),
        "199_financing_explicit_in_every_record": all(
            row
            and row.get("charged") is False
            and bool(row.get("mechanism"))
            and bool(row.get("reason"))
            for row in financing_rows.values()
        ),
        "200_training_epochs_exclude_baseline": all(
            row == {"requested": 1, "realized": 1, "baseline": 1}
            for row in epoch_rows.values()
        ),
        "decision_identity_recomputed": computed_decision_id == DECISION_ID,
        "decision_state_uncontaminated": decision_state_uncontaminated,
        "smoke_never_decision_eligible": (
            aggregation.get("outcome") == "INCONCLUSIVE"
            and len(refusals) == 16
            and all(record.get("evidence_class") == "mechanics_smoke"
                    and record.get("decision_eligible") is False
                    for record in records)
        ),
        "replica_loaded_all_terminals": (
            len(manifest["replica"]["verification"]) == 16
            and all(row["loads"] for row in manifest["replica"]["verification"])
        ),
    }
    output = {
        "schema": "agent_multi.audit.l1_round3_acceptance_repro.v1",
        "runtime_mutated": False,
        "accepted": all(checks.values()),
        "checks": checks,
        "facts": {
            "collection_digest": current_digest,
            "computed_decision_id": computed_decision_id,
            "decision_roots_present": decision_roots,
            "decision_seed_dirs": decision_seed_dirs,
            "deployed_env": deployed_env,
            "records": len(records),
            "aggregation_outcome": aggregation.get("outcome"),
            "aggregation_refusals": len(refusals),
            "aggregation_artifact": str(aggregation_path),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
