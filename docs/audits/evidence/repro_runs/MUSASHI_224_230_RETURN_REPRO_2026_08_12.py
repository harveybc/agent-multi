#!/usr/bin/env python3
"""Independent reproducer for the 224-230 return audit.

The source checks are intentionally small and structural.  The runtime checks
read the active decision identity and the historical replica without mutating
either workload.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
IDENTITY = "1434685bfdf52911"
CONTRACT = REPO / (
    "examples/config/phase_3_eth_sac_dynamics/"
    "p1_difficulty_lr_factorial_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return ast.get_source_segment(source, child) or ""
    raise RuntimeError(f"missing {class_name}.{method_name} in {path}")


def _command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO,
        check=check,
        text=True,
        capture_output=True,
        timeout=45,
    )


def _remote(host: str, command: str) -> str:
    return _command(
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        host, command,
    ).stdout.strip()


def _remote_int(host: str, command: str) -> int:
    value = _remote(host, command)
    return int(value.splitlines()[-1]) if value else 0


def _host_command(host: str, command: str) -> str:
    if host == "omega":
        return _command("bash", "-lc", command).stdout.strip()
    return _remote(host, command)


def reproduce() -> dict:
    pipeline_path = REPO / "pipeline_plugins/rl_pipeline_with_validation.py"
    runner_path = REPO / "tools/p1_difficulty_lr_factorial.py"
    status_path = REPO / "tools/multifront_status.py"
    guard_path = REPO / "tools/p1lr_idle_guard.py"

    make_env = _method_source(pipeline_path, "PipelinePlugin", "_make_split_env")
    rollout = _method_source(pipeline_path, "PipelinePlugin", "_rollout")
    run_pipeline = _method_source(pipeline_path, "PipelinePlugin", "run_pipeline")
    run_cell = runner_path.read_text(encoding="utf-8")
    status_source = status_path.read_text(encoding="utf-8")
    guard_source = guard_path.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        status_output = Path(tmp.name)
    try:
        _command(
            str(Path.home() / "anaconda3/envs/trading-stack/bin/python"),
            str(status_path),
            "--p1lr-identity", IDENTITY,
            "--no-l1", "--no-emit-alerts",
            "--output", str(status_output),
        )
        status = json.loads(status_output.read_text(encoding="utf-8"))
    finally:
        status_output.unlink(missing_ok=True)

    active = status["fronts"]["f1_optimization"]["active_p1lr_factorial"]
    source_root = "/home/harveybc/Documents/GitHub/_pre_trading_stack_20260713"
    dest_root = "/home/harveybc/replicas/_pre_trading_stack_20260713_gamma_replica"
    source_bytes = _remote_int("gamma", f"du -sb {source_root} | cut -f1")
    dest_bytes = _remote_int("dragon", f"du -sb {dest_root} | cut -f1")
    source_files = _remote_int("gamma", f"find {source_root} -type f | wc -l")
    dest_files = _remote_int("dragon", f"find {dest_root} -type f | wc -l")
    transfer_processes = _remote_int(
        "gamma",
        "pgrep -fc '^rsync .*_pre_trading_stack_20260713' || true",
    )
    decision_runtime = {}
    for host, seeds in (("omega", (101,)), ("dragon", (202,)),
                        ("gamma", (303, 404))):
        processes = {}
        for seed in seeds:
            processes[str(seed)] = int(_host_command(
                host,
                "pgrep -fc '^/home/harveybc/anaconda3/envs/trading-stack/"
                f"bin/python tools/p1_difficulty_lr_factorial.py --seed {seed} "
                "--mode decision' || true",
            ))
        gpu_rows = _host_command(
            host,
            "nvidia-smi --query-gpu=index,name,temperature.gpu,"
            "utilization.gpu,memory.used --format=csv,noheader,nounits",
        ).splitlines()
        decision_runtime[host] = {
            "decision_processes_by_seed": processes,
            "gpu_rows_index_name_temp_c_util_pct_memory_mib": gpu_rows,
        }

    unit_files = sorted(
        path.name for path in (REPO / "examples/systemd").glob("*p1lr*")
    )
    result = {
        "schema": "agent_multi.musashi_224_230_return_repro.v1",
        "subject": {
            "contract": str(CONTRACT.relative_to(REPO)),
            "contract_sha256": _sha256(CONTRACT),
            "decision_identity": IDENTITY,
        },
        "nested_context_execution": {
            "wrapper_declared": "ContextPrefixWrapper" in (
                REPO / "pipeline_plugins/_nested_splits.py"
            ).read_text(encoding="utf-8"),
            "wrapper_applied_by_internal_split_factory": "ContextPrefixWrapper" in make_env,
            "rollout_filters_is_context_prefix": "is_context_prefix" in rollout,
            "internal_roles_with_context_rows": {
                role: contract["nested_split_contract"]["role_facts"][role]["context_rows"]
                for role in ("train_monitor", "inner_validation")
            },
            "finding_reproduced": (
                "ContextPrefixWrapper" not in make_env
                and "is_context_prefix" not in rollout
            ),
        },
        "inactive_decision_publication": {
            "pipeline_returns_typed_inactive": (
                'final["best_model_path"] = None' in run_pipeline
                and 'final["terminal_model_path"]' in run_pipeline
            ),
            "decision_runner_unconditionally_requires_best": (
                'if mode == "decision"' in run_cell
                and "decision cell finished without a restorable best" in run_cell
            ),
            "finding_reproduced": (
                'final["best_model_path"] = None' in run_pipeline
                and "decision cell finished without a restorable best" in run_cell
            ),
        },
        "decision_observability": {
            "screen_output_root": contract["output_root"],
            "decision_output_root": contract["decision_run"]["output_root"],
            "status_reads_top_level_output_root": (
                'contract.get("output_root")' in status_source
            ),
            "idle_guard_reads_top_level_output_root": (
                'contract["output_root"]' in guard_source
            ),
            "idle_guard_unit_template_is_screen": (
                'UNIT_TEMPLATE = "p1lr-screen@{seed}.service"' in guard_source
            ),
            "shipped_p1lr_unit_files": unit_files,
            "reported_mode": active.get("mode"),
            "reported_state": active.get("state"),
            "reported_workers_running": active.get("workers_running_fresh", {}).get("value"),
            "reported_records": active.get("records_landed", {}).get("value"),
            "finding_reproduced": (
                active.get("mode") == "screen"
                and active.get("workers_running_fresh", {}).get("value") == 0
            ),
            "direct_runtime_facts": decision_runtime,
        },
        "historical_replica": {
            "transfer_process_count": transfer_processes,
            "source_bytes": source_bytes,
            "destination_bytes_observed": dest_bytes,
            "byte_fraction_observed": (
                dest_bytes / source_bytes if source_bytes else None
            ),
            "source_regular_files": source_files,
            "destination_regular_files_observed": dest_files,
            "terminal_digest_proof_available": False,
            "finding_230_state": "running_incomplete",
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reproduce()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
