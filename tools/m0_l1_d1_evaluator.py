#!/usr/bin/env python3
"""D1_EVALUATOR_ONLY — the measurement-gate arm of the mechanism
ladder (finding 220, order WP3 §3.3).

CPU-only, evaluation-only: takes D0_M0_EXACT's terminal artifact and
its recorded normal-realistic evaluation facts and applies BOTH
activity definitions to the SAME facts:

  * the M0 screen's executable definition (terminal_usable: positive
    validation trades, weights changed from the anchor, positive
    normal gradient updates, loadable terminal);
  * the L1 paired activity classifier — the EXACT
    ``activity_facts`` predicate the L1 aggregation used (reused,
    never copied).

No training, no weight mutation, no environment rollout: the artifact
is loaded read-only on CPU to verify the tensor chain and the update
counter; every behavioral fact comes from D0's recorded evaluation.
If D0 is active and the two labels disagree, the defect is the
activity DEFINITION, not learning (order §3.4).
"""
from __future__ import annotations

import os

# CPU-ONLY by construction: when executed as the D1 tool the CUDA
# binding is cleared BEFORE any torch import so this process can never
# claim a fleet GPU. (Library imports — e.g. the socket-free tests —
# must not mutate the host process environment; every artifact load
# below is explicitly device="cpu" regardless.)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools.aggregate_l1_factorial import activity_facts  # noqa: E402
from tools.l1_factorial_screen import atomic_write_json  # noqa: E402

RECORD_SCHEMA = "agent_multi.m0_l1_d1_evaluation.v1"
D0_RECORD_SCHEMA = "agent_multi.m0_l1_ladder_arm_record.v1"


def _positive(value) -> bool | None:
    if value is None:
        return None
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return None


def m0_activity_label(rollout_summary: dict, *, terminal_loads: bool,
                      weights_changed_from_anchor: bool,
                      updates_positive: bool) -> dict:
    """The M0 screen's executable activity definition, applied to the
    same facts the L1 classifier sees. Absent facts stay None — an
    unavailable fact is never treated as zero."""
    trades = rollout_summary.get("trades_total")
    facts = {
        "terminal_loads": bool(terminal_loads),
        "weights_changed_from_anchor": bool(
            weights_changed_from_anchor),
        "normal_updates_positive": bool(updates_positive),
        "validation_trades": trades,
        "validation_trades_positive": _positive(trades),
        "raw_action_std_positive": _positive(
            rollout_summary.get("action_raw_std")),
        "non_hold_rate_positive": _positive(
            rollout_summary.get("action_non_hold_rate")),
    }
    required = (facts["terminal_loads"],
                facts["weights_changed_from_anchor"],
                facts["normal_updates_positive"],
                facts["validation_trades_positive"])
    if any(value is None for value in required):
        facts["label"] = "invalid"
        facts["invalid_reasons"] = ["a required M0 fact is unavailable"]
    else:
        facts["label"] = "active" if all(required) else "inactive"
    return facts


def l1_activity_label(terminal_probe: dict,
                      rollout_summary: dict) -> dict:
    """The L1 paired activity classifier, verbatim (reused from the L1
    aggregation). Invalid is never inactive."""
    facts = activity_facts(terminal_probe=terminal_probe,
                           rollout_summary=rollout_summary)
    if facts.get("valid") is not True:
        facts["label"] = "invalid"
    else:
        facts["label"] = "active" if facts.get("active") else "inactive"
    return facts


def probe_terminal_artifact(path: Path, *, expected_sha256: str,
                            anchor_tensor_sha256: str,
                            recorded_tensor_sha256: str) -> dict:
    """Read-only CPU probe: verify bytes, load, hash the policy
    tensors, read the update counter. Nothing is trained or written."""
    actual = sysid.sha_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"terminal artifact hash {actual[:16]}… does not equal the "
            f"D0 record binding {expected_sha256[:16]}…")
    probe: dict = {"artifact_sha256": actual}
    try:
        from stable_baselines3 import SAC

        from agent_plugins.sac_agent import _policy_tensor_hash

        model = SAC.load(str(path), device="cpu")
        try:
            tensor = _policy_tensor_hash(model.policy)
            n_updates = int(getattr(model, "_n_updates", 0) or 0)
        finally:
            del model
        probe["loads"] = True
        probe["terminal_policy_tensor_sha256"] = tensor
        probe["n_updates"] = n_updates
        probe["phase2_updates_occurred"] = n_updates > 0
        chain_ok = (tensor == recorded_tensor_sha256
                    and tensor != anchor_tensor_sha256)
        probe["tensor_chain_consistent"] = chain_ok
        probe["tensor_chain_detail"] = (
            "observed terminal tensor equals the recorded terminal "
            "tensor and differs from the anchor tensor" if chain_ok else
            f"observed {tensor[:16]}…, recorded "
            f"{recorded_tensor_sha256[:16]}…, anchor "
            f"{anchor_tensor_sha256[:16]}…")
    except Exception as exc:                       # noqa: BLE001
        probe["loads"] = False
        probe["error"] = f"{type(exc).__name__}: {exc}"
        probe["tensor_chain_consistent"] = False
        probe["phase2_updates_occurred"] = None
    return probe


def evaluate_d0(record_path: Path) -> dict:
    record = json.loads(Path(record_path).read_text())
    if record.get("schema") != D0_RECORD_SCHEMA:
        raise ValueError(
            f"not a ladder arm record: {record.get('schema')!r}")
    if record.get("arm") != "D0_M0_EXACT":
        raise ValueError(
            "D1 evaluates the POSITIVE CONTROL only; got arm "
            f"{record.get('arm')!r}")
    rollout = ((record.get("terminal_evaluation_as_run") or {})
               .get("splits_raw") or {}).get("validation")
    if not isinstance(rollout, dict):
        raise ValueError("D0 record carries no as-run validation "
                         "evaluation — D1 has no facts to classify")
    terminal_path = Path(record["terminal_model_path"])
    probe = probe_terminal_artifact(
        terminal_path,
        expected_sha256=record["terminal_model_sha256"],
        anchor_tensor_sha256=record["anchor_policy_tensor_sha256"],
        recorded_tensor_sha256=record[
            "terminal_policy_tensor_sha256"])

    under_l1 = l1_activity_label(probe, rollout)
    under_m0 = m0_activity_label(
        rollout,
        terminal_loads=probe.get("loads") is True,
        weights_changed_from_anchor=(
            probe.get("terminal_policy_tensor_sha256")
            != record["anchor_policy_tensor_sha256"]),
        updates_positive=bool(probe.get("phase2_updates_occurred")),
    )
    labels_agree = under_m0["label"] == under_l1["label"]
    if under_m0["label"] == "active" and not labels_agree:
        interpretation = ("D0 is active under the M0 definition and "
                          "the L1 classifier changes the label: the "
                          "defect is the ACTIVITY DEFINITION, not "
                          "learning (order §3.4)")
    elif under_m0["label"] == "active":
        interpretation = ("both definitions call D0 active: the "
                          "measurement gate is not the collapsing "
                          "mechanism; look to D2/D3/D4")
    else:
        interpretation = ("D0 is not active under the M0 definition: "
                          "the ladder is INVALID — diagnose source "
                          "anchor, data/code revision and the M0 claim "
                          "before any new compute (order §3.4)")

    import torch
    return {
        "schema": RECORD_SCHEMA,
        "arm": "D1_EVALUATOR_ONLY",
        "evidence_class": "mechanism_diagnostic",
        "decision_eligible": False,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "no_training": True,
        "no_weight_mutation": True,
        "d0_record_path": str(Path(record_path).resolve()),
        "diagnostic_identity": record.get("diagnostic_identity"),
        "d0_arm_identity": record.get("arm_identity"),
        "terminal_model_sha256": record.get("terminal_model_sha256"),
        "terminal_probe": probe,
        "facts_source": ("D0 terminal artifact (read-only CPU probe) + "
                         "D0 recorded as-run normal-realistic "
                         "validation evaluation"),
        "label_under_m0_definition": under_m0["label"],
        "label_under_l1_definition": under_l1["label"],
        "labels_agree": labels_agree,
        "m0_definition_facts": under_m0,
        "l1_definition_facts": under_l1,
        "interpretation": interpretation,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0-record", type=Path, required=True,
                        help="path to D0_M0_EXACT's "
                             "ladder_arm_record.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default: "
                             "d1_evaluator_record.json beside the D0 "
                             "record)")
    args = parser.parse_args()
    result = evaluate_d0(args.d0_record)
    out = args.out or args.d0_record.parent / "d1_evaluator_record.json"
    atomic_write_json(out, result)
    print(json.dumps({
        "arm": "D1_EVALUATOR_ONLY",
        "label_under_m0_definition": result[
            "label_under_m0_definition"],
        "label_under_l1_definition": result[
            "label_under_l1_definition"],
        "labels_agree": result["labels_agree"],
        "record": str(out),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
