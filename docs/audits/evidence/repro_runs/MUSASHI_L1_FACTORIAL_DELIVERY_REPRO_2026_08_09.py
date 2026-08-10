#!/usr/bin/env python3
"""Independent, socket-free counterexamples for the L1 factorial delivery."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tools import aggregate_l1_factorial as aggregate  # noqa: E402


def _load_fixture_module():
    path = ROOT / "tests/test_aggregate_l1_factorial.py"
    spec = importlib.util.spec_from_file_location("l1_audit_fixtures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rollout(fixtures, record: dict) -> dict:
    if str(record["phase1_mode"]).startswith("easy"):
        return fixtures.healthy_rollout(record)
    return fixtures.inactive_rollout(record)


def _aggregate(fixtures, *, mutate=None, remove_results=False) -> dict:
    contract = fixtures.make_contract()
    with tempfile.TemporaryDirectory(prefix="musashi-l1-repro-") as tmp:
        root = fixtures.build_tree(Path(tmp), contract, mutate=mutate)
        if remove_results:
            (root / fixtures.EXP / "seed101" / "L1_N_M10" /
             "results.json").unlink()
        return aggregate.aggregate(
            root,
            fixtures.EXP,
            contract=contract,
            probe_fn=fixtures.healthy_probe,
            rollout_fn=lambda record: _rollout(fixtures, record),
        )


def main() -> int:
    fixtures = _load_fixture_module()

    missing_metrics = _aggregate(fixtures, remove_results=True)

    def tamper_code(record: dict, seed: int, cell: str) -> None:
        if seed == 101 and cell == "L1_N_M10":
            record["code_revisions"] = {
                "agent-multi": "adversarial-revision",
                "gym-fx": "adversarial-revision",
            }

    bad_code = _aggregate(fixtures, mutate=tamper_code)

    contract = fixtures.make_contract()
    accepted = fixtures.make_record(contract, 101, "L1_N_M10")
    mandatory_but_absent = [
        key for key in (
            "system_manifest_sha256",
            "resolved_config_sha256",
            "observation_manifest_sha256",
            "terminal_model_sha256",
            "terminal_policy_tensor_sha256",
            "phase1_requested_epochs",
            "phase2_requested_epochs",
        ) if key not in accepted
    ]
    accepted_reasons = aggregate.validate_record_bindings(
        accepted,
        contract=contract,
        seed=101,
        cell="L1_N_M10",
        experiment_id=fixtures.EXP,
        evidence=fixtures.make_evidence(),
    )

    dispatch = (ROOT / "tools/dispatch_l1_factorial_fleet.sh").read_text()
    runner = (ROOT / "tools/l1_factorial_screen.py").read_text()
    result = {
        "schema": "agent_multi.audit.l1_factorial_delivery_repro.v1",
        "network_used": False,
        "counterexamples": {
            "missing_results_metrics": {
                "expected": "INCONCLUSIVE",
                "actual": missing_metrics["outcome"],
                "refusals": missing_metrics["refusals"],
                "raw_metric_fact": missing_metrics["raw_metrics_per_seed"]
                    ["seed101/L1_N_M10"],
                "reproduced": missing_metrics["outcome"] != "INCONCLUSIVE",
            },
            "tampered_subject_code_revision": {
                "expected": "INCONCLUSIVE",
                "actual": bad_code["outcome"],
                "refusals": bad_code["refusals"],
                "reproduced": bad_code["outcome"] != "INCONCLUSIVE",
            },
            "mandatory_identity_fields_absent": {
                "missing_fields": mandatory_but_absent,
                "binding_refusals": accepted_reasons,
                "reproduced": bool(mandatory_but_absent) and not accepted_reasons,
            },
            "duplicate_dispatch_guards_absent": {
                "flock_present": "flock" in dispatch,
                "exclusive_claim_present": "exclusive_claim" in dispatch,
                "completed_record_reuse_present": (
                    "l1_cell_record.json" in runner and
                    "exists()" in runner
                ),
                "atomic_record_replace_present": "replace(" in runner,
                "reproduced": all((
                    "flock" not in dispatch,
                    "exclusive_claim" not in dispatch,
                    not ("l1_cell_record.json" in runner and
                         "exists()" in runner),
                    "replace(" not in runner,
                )),
            },
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(
        item.get("reproduced", True)
        for item in result["counterexamples"].values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
