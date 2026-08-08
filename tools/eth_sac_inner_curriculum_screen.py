#!/usr/bin/env python3
"""M0 mechanism screen — one seed, four frozen arms (WP2 of the SAC
inner-curriculum order 2026-08-07 §8).

Tests ONLY whether easy pretraining changes survival versus
equal-compute normal-only, and whether a reduced normal fine-tuning LR
prevents first-epoch activity collapse. Everything else is D1-identical
by construction: the base config, anchors, validation contract, replica
discipline and record validation are REUSED from the validated D1
runner (`eth_curriculum_decision_experiment`) — never copied.

Typed validation happens before any model construction; the anchor is
the EXACT D1 per-seed artifact verified by sha256; decision facts are
computed from direct evidence and `terminal_usable` is never inferred
from best-checkpoint success (the anchor-fallback lesson,
rl_pipeline_with_validation.py:1040-1051).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import eth_curriculum_decision_experiment as d1  # noqa: E402

SCHEMA = "agent_multi.m0_arm_record.v1"
CONTRACT_PATH = REPO / "examples/config/phase_3_eth_sac_dynamics/m0_contract.json"
M0_ARM_ORDER = ("N2_LR1", "E1_N1_LR1", "E1_N1_LR03", "E1_N1_LR01")


SCHEMA_V2 = "agent_multi.inner_curriculum_screen_contract.v2"


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    contract = json.loads(path.read_text())
    contract["_contract_sha256"] = hashlib.sha256(
        path.read_bytes()).hexdigest()
    if contract.get("schema") == SCHEMA_V2:
        validate_contract_v2(contract)
    else:
        validate_contract(contract)
    return contract


def validate_contract_v2(contract: dict) -> None:
    """Generic per-asset screen contract (Musashi doc-37 correction 5).

    v2 speaks LR MULTIPLIERS over the anchor's own baseline training
    rate (correction 1): an absolute rate copied across assets silently
    changes meaning when anchors were trained at different rates
    (ETH 1e-4 vs USDCAD 3.559e-4). The loader resolves absolute rates
    so the run path stays identical. Same typed refusals BEFORE model
    construction: bool-as-number, NaN, nonpositive, negative epochs and
    unequal compute all refuse.
    """
    epoch = int(contract["epoch_timesteps"])
    total_epochs = int(contract["total_epochs_per_arm"])
    if epoch <= 0 or total_epochs <= 0:
        raise ValueError("epoch budgets must be positive")

    def _positive_number(value, label):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
        return value

    baseline = _positive_number(
        contract["baseline_learning_rate"], "baseline_learning_rate")
    easy_mult = _positive_number(
        contract["easy_lr_multiplier"], "easy_lr_multiplier")
    contract["easy_learning_rate"] = baseline * easy_mult
    arms = contract["arms"]
    if not arms:
        raise ValueError("contract declares no arms")
    for name, spec in arms.items():
        easy = spec["easy_epochs"]
        normal = spec["normal_epochs"]
        for value, label in ((easy, "easy_epochs"), (normal, "normal_epochs")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}.{label} must be an integer")
            if value < 0:
                raise ValueError(f"{name}.{label} must not be negative")
        mult = _positive_number(
            spec["normal_lr_multiplier"], f"{name}.normal_lr_multiplier")
        spec["normal_learning_rate"] = baseline * mult
        if (easy + normal) != total_epochs:
            raise ValueError(
                f"{name}: unequal compute — {easy}+{normal} epochs"
                f" != {total_epochs}")
    contract["total_updates_per_arm"] = total_epochs * epoch
    if contract.get("launch_eligible") is not True:
        contract["_launch_blocked_reason"] = (
            "contract instance is not launch_eligible (winner multiplier"
            " pending audit selection)")


def validate_contract(contract: dict) -> None:
    """Reject invalid factors BEFORE any model construction (§7)."""
    total = int(contract["total_updates_per_arm"])
    epoch = int(contract["epoch_timesteps"])
    if total <= 0 or epoch <= 0:
        raise ValueError("update budgets must be positive")
    easy_lr = contract["easy_learning_rate"]
    if isinstance(easy_lr, bool) or not isinstance(easy_lr, (int, float)) \
            or not math.isfinite(float(easy_lr)) or float(easy_lr) <= 0:
        raise ValueError("easy_learning_rate must be a positive finite number")
    arms = contract["arms"]
    if set(arms) != set(M0_ARM_ORDER):
        raise ValueError(f"unknown or missing arms: {sorted(arms)}")
    for name, spec in arms.items():
        easy = spec["easy_epochs"]
        normal = spec["normal_epochs"]
        rate = spec["normal_learning_rate"]
        for value, label in ((easy, "easy_epochs"), (normal, "normal_epochs")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}.{label} must be an integer")
            if value < 0:
                raise ValueError(f"{name}.{label} must not be negative")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError(f"{name}.normal_learning_rate must be a number")
        rate = float(rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(
                f"{name}.normal_learning_rate must be finite and positive")
        if (easy + normal) * epoch != total:
            raise ValueError(
                f"{name}: unequal compute — ({easy}+{normal})x{epoch}"
                f" != {total}")


def resolve_anchor(contract: dict, seed: int) -> Path:
    entry = contract["anchors"][str(seed)]
    path = Path(entry["path"]).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"anchor missing for seed {seed}: {path}")
    actual = d1._sha(path)
    if actual != entry["sha256"]:
        raise ValueError(
            f"anchor hash mismatch for seed {seed}: contract"
            f" {entry['sha256'][:16]}..., disk {actual[:16]}...")
    return path


def _activity_facts(summary: dict | None) -> dict:
    """Direct activity facts from a normal-validation summary; absent
    facts stay None, never zero."""
    if not isinstance(summary, dict):
        return {"classification": "source_unavailable"}
    return {
        "val_trades": summary.get("val_trades"),
        "raw_std": summary.get("val_action_raw_std"),
        "non_hold_rate": summary.get("val_action_non_hold_rate"),
        "no_trade_diagnosis": summary.get("val_no_trade_diagnosis"),
    }


def run_m0_arm(arm: str, seed: int, out_root: Path, *,
               agent_name: str, contract: dict) -> dict:
    from pipeline_plugins.rl_pipeline_with_validation import (
        PipelinePlugin as ValidationPipeline)
    from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
        PipelinePlugin as CurriculumPipeline)

    spec = contract["arms"][arm]
    epoch_timesteps = int(contract["epoch_timesteps"])
    anchor = resolve_anchor(contract, seed)
    out_dir = out_root / f"seed{seed}" / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    config = d1._base_config(out_dir, arm, seed,
                             epoch_timesteps=epoch_timesteps)
    config["learning_rate"] = float(spec["normal_learning_rate"])
    config["easy_learning_rate"] = float(contract["easy_learning_rate"])
    config["warm_start_model"] = str(anchor)
    config["warm_start_model_sha256"] = d1._sha(anchor)
    config["l1_patience"] = 10_000            # no early stopping (§8.1)
    config["easy_patience"] = 10_000
    # D1 set this field to its arm epoch count; it only feeds the
    # progress annotation (no execution_cost_curriculum contract exists
    # in the D1/M0 configs — order §3.3) and its validator requires >=2.
    config["execution_cost_curriculum_epochs"] = max(
        2, int(spec["normal_epochs"]))
    agent = d1._agent_plugin(agent_name)
    code_before = {repo: d1._git_rev(repo)
                   for repo in ("agent-multi", "gym-fx", "doin-plugins")}
    started = datetime.now(timezone.utc)

    if spec["easy_epochs"] == 0:
        config["max_epochs"] = int(spec["normal_epochs"])
        config["solvency_mode"] = "normal_realistic"
        pipeline = ValidationPipeline(config)
        result = pipeline.run_pipeline(
            config=config, env_plugin=None, agent_plugin=agent,
            mode="train")
    else:
        config["max_epochs"] = int(spec["normal_epochs"])
        config["easy_max_epochs"] = int(spec["easy_epochs"])
        pipeline = CurriculumPipeline(config)
        result = pipeline.run_pipeline(
            config=config, env_plugin=None, agent_plugin=agent,
            mode="train")

    finished = datetime.now(timezone.utc)
    code_after = {repo: d1._git_rev(repo)
                  for repo in ("agent-multi", "gym-fx", "doin-plugins")}
    if code_before != code_after:
        raise RuntimeError(
            f"code revisions moved during arm {arm}: {code_before} ->"
            f" {code_after}")

    history = result.get("history") or []
    last = history[-1] if history else {}
    anchor_sha = config["warm_start_model_sha256"]
    terminal_path = result.get("terminal_model_path")
    terminal_sha = (d1._sha(Path(terminal_path))
                    if terminal_path and Path(terminal_path).is_file()
                    else None)
    best_path = result.get("best_model_path")
    best_sha = (d1._sha(Path(best_path))
                if best_path and Path(best_path).is_file() else None)
    updates = last.get("gradient_updates_total")

    # Terminal weights evaluated RAW under normal validation — the D1
    # discipline: terminal_usable is never inferred from the selected
    # best checkpoint (the anchor-fallback lesson).
    terminal_eval: dict = {}
    terminal_val_trades = None
    if terminal_path and Path(terminal_path).is_file():
        from pipeline_plugins.rl_pipeline_with_validation import (
            PipelinePlugin as ValidationPipeline)
        eval_config = dict(config)
        eval_config["load_model"] = str(terminal_path)
        eval_config["solvency_mode"] = "normal_realistic"
        eval_config["return_trace_dir"] = str(
            out_dir / "return_traces_terminal")
        terminal_result = ValidationPipeline(eval_config).run_pipeline(
            config=eval_config, env_plugin=None, agent_plugin=agent,
            mode="inference")
        terminal_splits = d1._splits_raw(terminal_result)
        terminal_eval = {
            "artifact_sha256": terminal_sha,
            "artifact_path": str(terminal_path),
            "splits_raw": terminal_splits,
        }
        terminal_val_trades = (
            (terminal_splits.get("validation") or {}).get("trades_total"))
    decision_facts = {
        "activity_survived_normal": (
            None if terminal_val_trades is None
            else bool(terminal_val_trades and terminal_val_trades > 0)),
        "weights_changed_from_anchor": (
            None if terminal_sha is None else terminal_sha != anchor_sha),
        "post_easy_activity": (
            ((result.get("curriculum") or {}).get("post_easy") or {})
            .get("probe") if spec["easy_epochs"] else
            {"classification": "not_applicable"}),
        "terminal_activity": _activity_facts(last),
        "normal_updates_applied": updates,
        "anchor_selected_as_best": (
            None if best_sha is None else best_sha == anchor_sha),
        "terminal_usable": bool(
            terminal_val_trades and terminal_val_trades > 0
            and terminal_sha is not None and terminal_sha != anchor_sha
            and updates and updates > 0),
    }

    record = {
        "schema": SCHEMA,
        "arm": arm,
        "seed": seed,
        "m0_contract_sha256": contract["_contract_sha256"],
        "arm_spec": spec,
        "easy_learning_rate": float(contract["easy_learning_rate"]),
        "execution_id": d1._execution_id(
            f"M0_{arm}", seed, Path(anchor),
            epoch_timesteps=epoch_timesteps),
        "anchor_sha256": anchor_sha,
        "terminal_sha256": terminal_sha,
        "best_sha256": best_sha,
        "code_revisions_before": code_before,
        "code_revisions_after": code_after,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "epoch_history": history,
        "terminal_evaluation": terminal_eval,
        "boundary_transfer_evidence": result.get(
            "warm_start_transfer_evidence"),
        "decision_facts": decision_facts,
        "pipeline_result_keys": sorted(result.keys()),
    }
    (out_dir / "m0_arm_record.json").write_text(
        json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True,
                        choices=(101, 202, 303, 404))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--agent", default="sac_agent")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH,
                        help="frozen screen contract (v1 M0 or generic"
                             " v2 per-asset instance)")
    parser.add_argument("--arms", nargs="*", default=None)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    if contract.get("_launch_blocked_reason"):
        print(json.dumps({"error": contract["_launch_blocked_reason"]}))
        return 2
    arm_order = list(contract["arms"])
    requested = args.arms if args.arms is not None else arm_order
    unknown = set(requested) - set(arm_order)
    if unknown:
        print(json.dumps({"error": f"unknown arms: {sorted(unknown)}"}))
        return 2
    args.arms = requested
    results = {}
    for arm in arm_order:
        if arm not in args.arms:
            continue
        record = run_m0_arm(arm, args.seed, args.output_root,
                            agent_name=args.agent, contract=contract)
        results[arm] = record["decision_facts"]
        print(json.dumps({"arm": arm, "seed": args.seed,
                          "decision_facts": record["decision_facts"]},
                         default=str))
    packet = {
        "schema": "agent_multi.m0_seed_packet.v1",
        "seed": args.seed,
        "m0_contract_sha256": contract["_contract_sha256"],
        "arms": sorted(results),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_root / f"seed{args.seed}" /
     "m0_seed_packet.json").write_text(
        json.dumps(packet, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
