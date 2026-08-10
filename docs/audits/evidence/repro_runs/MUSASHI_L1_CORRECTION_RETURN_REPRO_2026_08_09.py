#!/usr/bin/env python3
"""Socket-free adversarial reproduction for the L1 correction return.

This script does not train, contact a broker, or contact another fleet node.
It exercises the delivered launcher/collector contracts and inspects the exact
materialized experiment configuration.
"""
from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid
from tests.test_collect_l1_factorial import (
    build_source,
    local_fetch,
    make_fleet_contract,
)
from tests.test_aggregate_l1_factorial import EXP
from tools import collect_l1_factorial as collector
from tools import l1_factorial_screen as runner


def main() -> int:
    facts: dict[str, object] = {
        "schema": "agent_multi.audit.l1_correction_return_repro.v1",
        "network_used": False,
        "counterexamples": {},
    }

    # A SEED_FAILED outcome maps to exit 2, while the delivered systemd unit
    # declares exit 2 successful. Restart=on-failure therefore cannot retry it.
    service = (REPO / "examples/systemd/l1-factorial@.service").read_text()
    failed_exit = 0 if "SEED_FAILED" in ("SEED_COMPLETE", "ALREADY_COMPLETE") else 2
    facts["counterexamples"]["failed_seed_is_not_restarted"] = {
        "seed_failed_exit_code": failed_exit,
        "success_exit_status": "0 2" if "SuccessExitStatus=0 2" in service else None,
        "restart_policy": "on-failure" if "Restart=on-failure" in service else None,
        "reproduced": (
            failed_exit == 2
            and "SuccessExitStatus=0 2" in service
            and "Restart=on-failure" in service
        ),
    }

    with tempfile.TemporaryDirectory(prefix="l1-correction-audit-") as td:
        root = Path(td)
        contract = make_fleet_contract(root)
        build_source(root, contract)
        result = collector.collect(
            contract=contract,
            experiment_id=EXP,
            collection_root=root / "collection",
            fetch_fn=local_fetch,
            replica_host=None,
        )
        facts["counterexamples"]["collection_seals_without_replica"] = {
            "outcome": result.get("outcome"),
            "replica_fact": result.get("replica"),
            "reproduced": (
                result.get("outcome") == "COLLECTION_SEALED"
                and result.get("replica") is None
            ),
        }

        sealed = Path(result["sealed_root"])
        # A real collector does not have the remote worker's filesystem.
        # Remove the local source fixture after the copy to model that fact.
        shutil.rmtree(Path(contract["output_root"]))
        record_path = next(sealed.rglob("l1_cell_record.json"))
        record = json.loads(record_path.read_text())
        original_attempt = Path(record["attempt_dir"])
        copied_attempts = sorted(record_path.parent.glob("attempt-*"))
        facts["counterexamples"]["sealed_records_keep_remote_absolute_paths"] = {
            "record_path": str(record_path),
            "record_attempt_dir": str(original_attempt),
            "record_attempt_exists": original_attempt.exists(),
            "sealed_attempt_dirs": [str(p) for p in copied_attempts],
            "terminal_path_exists_from_collector_host": Path(
                record["terminal_model_path"]
            ).exists(),
            "reproduced": bool(copied_attempts) and not original_attempt.exists(),
        }

    contract = runner.load_contract()
    manifest = runner.load_system_manifest()
    materialized = sysid.materialize_system_config(
        contract, manifest, "L1_N_M10", 101, Path("/tmp/l1-audit-output")
    )
    declared = manifest.get("plugins", {})
    actual_agent_default = inspect.signature(runner.run_cell).parameters[
        "agent_name"
    ].default
    runner_source = inspect.getsource(runner.run_cell)
    materializer_source = inspect.getsource(sysid.materialize_system_config)
    facts["counterexamples"]["manifest_plugins_do_not_bind_execution"] = {
        "declared_agent_plugin": declared.get("agent_plugin"),
        "executed_agent_plugin_default": actual_agent_default,
        "declared_pipeline_plugin": declared.get("pipeline_plugin"),
        "executed_pipeline_class": "rl_pipeline_with_solvency_curriculum",
        "materializer_validates_plugins": "system_manifest[\"plugins\"]" in materializer_source,
        "runner_hardcodes_solvency_pipeline": (
            "rl_pipeline_with_solvency_curriculum" in runner_source
        ),
        "reproduced": (
            declared.get("agent_plugin") != actual_agent_default
            or declared.get("pipeline_plugin")
            != "rl_pipeline_with_solvency_curriculum"
        ) and "system_manifest[\"plugins\"]" not in materializer_source,
    }

    facts["counterexamples"]["normal_contract_is_not_fully_realistic_or_protected"] = {
        "commission_fraction_per_side": materialized.get("commission"),
        "slippage_rate_per_side": materialized.get("slippage"),
        "full_spread_rate": materialized.get("full_spread_rate"),
        "min_equity": materialized.get("min_equity"),
        "require_protected_entries": materialized.get("require_protected_entries"),
        "manifest_cost_keys": sorted(
            (manifest.get("costs", {}).get("config_bindings") or {}).keys()
        ),
        "reproduced": (
            materialized.get("slippage") == 0
            and materialized.get("full_spread_rate") is None
            and materialized.get("require_protected_entries") is not True
        ),
    }

    source_fact = manifest.get("source_identity_at_manifest", {}).get(
        "agent-multi", {}
    )
    facts["counterexamples"]["frozen_manifest_was_generated_dirty"] = {
        "commit": source_fact.get("commit"),
        "dirty": source_fact.get("dirty"),
        "dirty_untracked_digest": source_fact.get("dirty_untracked_digest"),
        "reproduced": source_fact.get("dirty") is True,
    }

    print(json.dumps(facts, indent=2, sort_keys=True))
    return 1 if not all(
        item.get("reproduced") is True
        for item in facts["counterexamples"].values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
