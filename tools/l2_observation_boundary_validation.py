#!/usr/bin/env python3
"""AUD-P1LR-20260815-235 bounded validation: does the CORRECTED observation
contract keep the SAC actor alive through a real phase-1 -> phase-2 boundary
on real ETH data?

One seed, one GPU, one candidate, 1 easy epoch + 3 normal epochs.

Environment knobs (defaults are the GPU dispatch values):
    OBS_VALIDATION_DEVICE      cuda
    OBS_VALIDATION_SEED        101
    OBS_VALIDATION_EPOCH_STEPS 20000
    OBS_VALIDATION_PHASE2      3
    OBS_VALIDATION_OUT         ~/.local/share/agent-multi/l2_observation_contract_validation_20260815
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import l2_curriculum_arms as l2                      # noqa: E402
from tools import m0_l1_mechanism_ladder as ladder              # noqa: E402
from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (  # noqa: E402
    PipelinePlugin)

DEVICE = os.environ.get("OBS_VALIDATION_DEVICE", "cuda")
SEED = int(os.environ.get("OBS_VALIDATION_SEED", "101"))
EPOCH_STEPS = int(os.environ.get("OBS_VALIDATION_EPOCH_STEPS", "20000"))
PHASE2_EPOCHS = int(os.environ.get("OBS_VALIDATION_PHASE2", "3"))
OUT = Path(os.environ.get(
    "OBS_VALIDATION_OUT",
    "~/.local/share/agent-multi/l2_observation_contract_validation_20260815"
)).expanduser()


def main() -> int:
    contract = l2.load_contract()
    bindings = l2.load_bindings()
    stage = l2.arm_schedule(contract, "L2_EN", "smoke")[0]
    genome = {"parameters": {"batch_size": 256, "gamma": 0.99, "tau": 0.005,
                             "train_freq": 1, "gradient_steps": 1}}
    OUT.mkdir(parents=True, exist_ok=True)

    config = l2.materialize_candidate_config(
        contract, bindings, arm="L2_EN", stage=stage, genome=genome,
        out_dir=OUT, mode="smoke")
    identity = config.pop("_identity")

    # The 2724-dim frozen anchor cannot warm-start a 2660-dim observation
    # (sac_agent supports observation EXPANSION only), and reusing it
    # would carry the dead first layer forward anyway. Cold start.
    config["warm_start_model"] = None
    config["warm_start_model_sha256"] = None

    # This is a mechanism diagnostic, not an L2 campaign candidate. Exercise
    # the disputed L1 easy -> normal handoff explicitly; the production L2
    # contract remains frozen and is not mutated by this run.
    config["phase1_mode"] = "easy_chronological_continuation"
    config["train_seed"] = SEED
    config["eval_seed"] = SEED
    config["ga_seed"] = SEED

    config["device"] = DEVICE
    config["epoch_timesteps"] = EPOCH_STEPS
    config["easy_max_epochs"] = 1
    config["max_epochs"] = PHASE2_EPOCHS
    config["total_max_passes"] = 1 + PHASE2_EPOCHS
    config["phase1_max_fraction"] = 0.5
    config["normal_phase_min_passes"] = 1
    config["execution_cost_curriculum_epochs"] = max(2, PHASE2_EPOCHS)
    config["quiet_mode"] = False
    config["save_model"] = str(OUT / "model.zip")
    config["return_trace_dir"] = str(OUT / "return_traces")
    # Measure, do not kill: this run exists to OBSERVE the boundary.
    config["refuse_dead_actor"] = False

    l2.materialize_l2_nested_roles(contract, bindings, OUT)
    agent = ladder._agent_plugin(identity["plugins"]["agent_plugin"])

    started = time.time()
    result = PipelinePlugin(config).run_pipeline(
        config=config, env_plugin=None, agent_plugin=agent, mode="train")
    elapsed = time.time() - started

    liveness = result.get("actor_liveness_history") or []
    payload = {
        "schema": "agent_multi.observation_boundary_validation.v1",
        "finding": "AUD-P1LR-20260815-235",
        "experiment_contract_sha256":
            identity["experiment_contract_sha256"],
        "evidence_scope": "bounded_mechanism_diagnostic_not_l2_decision",
        "diagnostic_intervention": {
            "phase1_mode": "easy_chronological_continuation",
            "phase2_mode": "normal_realistic",
            "old_anchor_removed": True,
            "campaign_contract_mutated": False,
        },
        "observation_contract": result.get("observation_contract"),
        "elapsed_seconds": round(elapsed, 1),
        "epoch_timesteps": EPOCH_STEPS,
        "phase1_epochs": 1,
        "phase2_epochs": PHASE2_EPOCHS,
        "cold_start": True,
        "seed": SEED,
        "device": DEVICE,
        "stop_reason": result.get("stop_reason"),
        "phase1_handoff_liveness": (
            ((result.get("curriculum") or {}).get("post_easy") or {})
            .get("handoff_viability_evidence")),
        "phase2_liveness_by_epoch": [
            {"epoch": f.get("epoch"),
             "classification": f.get("classification"),
             "live_unit_count": f.get("live_unit_count"),
             "first_layer_units": f.get("first_layer_units"),
             "live_unit_fraction": f.get("live_unit_fraction"),
             "varying_unit_count": f.get("varying_unit_count"),
             "preactivation_mean": f.get("preactivation_mean"),
             "action_raw_std": f.get("action_raw_std"),
             "constant_policy": f.get("constant_policy")}
            for f in liveness],
    }
    report = OUT / "observation_boundary_validation.json"
    report.write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== OBSERVATION BOUNDARY VALIDATION ===")
    print(f"observation contract: "
          f"{(result.get('observation_contract') or {}).get('source')}")
    print(f"applied: "
          f"{json.dumps((result.get('observation_contract') or {}).get('applied'))}")
    for row in payload["phase2_liveness_by_epoch"]:
        print(f"  phase2 epoch {row['epoch']}: {row['classification']} "
              f"live={row['live_unit_count']}/{row['first_layer_units']} "
              f"varying={row['varying_unit_count']} "
              f"action_raw_std={row['action_raw_std']}")
    print(f"stop_reason={payload['stop_reason']} "
          f"elapsed={payload['elapsed_seconds']}s")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
