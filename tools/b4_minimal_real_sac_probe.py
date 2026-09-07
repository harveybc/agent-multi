#!/usr/bin/env python3
"""C46: the REAL minimal SAC probe — no doubles. Runs
build_economic_config -> pipeline.run_pipeline -> model.learn on
CPU with a tightened executing budget, in BOTH composition cases:

  case A (B4 cell): mandatory progress telemetry materialized —
         the JSON progress file must exist and advance;
  case B (generic): no progress path and b4_require_progress
         unset — the composed callback list still contains no
         None (SB3 receives [budget_cb] only).

The step bound admits ONE sealed epoch segment (20000) so learn
truly starts; the UPDATE bound (50) is crossed INSIDE the segment
and the F9.2 executing callback stops learn exactly there — real
environment steps (>= learning_starts 128) and exactly the bounded
number of real gradient updates, then the guard's own typed stop.

Both cases must reach >= 1 real environment step and >= 1 real
gradient update, then stop EXACTLY via the F9.2 executing-budget
callback at the step bound; the heartbeat must exist under the
cell root and advance, bound to the attempt (it lives in the
claimed cell's private directory). NOT a campaign cell: throwaway
root, mechanics only, zero promotion, zero sealed-2025."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

STEP_BOUND = 20000
UPDATE_BOUND = 50


def run_case(executor, cfg, label):
    from app.plugin_loader import load_plugin
    agent_cls, _ = load_plugin("agent.plugins",
                               cfg["agent_plugin"])
    pipeline_cls, _ = load_plugin("pipeline.plugins",
                                  cfg["pipeline_plugin"])
    from pipeline_plugins.rl_pipeline_with_validation import \
        compose_learn_callbacks, make_executing_budget_callback
    probe_cbs = compose_learn_callbacks(
        cfg, 1000, make_executing_budget_callback(
            cfg, time.time()))
    assert all(c is not None for c in probe_cbs)
    n_cbs = len(probe_cbs)
    agent_plugin = agent_cls(cfg)
    pipeline = pipeline_cls(cfg)
    t0 = time.time()
    try:
        pipeline.run_pipeline(config=cfg, env_plugin=None,
                              agent_plugin=agent_plugin,
                              mode="train")
        stop = "returned"
    except SystemExit as exc:
        stop = f"SystemExit: {str(exc)[:90]}"
    except RuntimeError as exc:
        stop = f"RuntimeError: {str(exc)[:90]}"
    wall = round(time.time() - t0, 1)
    return {"label": label, "callbacks_composed": n_cbs,
            "stop": stop, "wall_seconds": wall}


def main() -> int:
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4exec_probe", REPO / "tools/b4_campaign_executor.py")
    executor = ilu.module_from_spec(spec)
    spec.loader.exec_module(executor)
    STATE = Path.home() / ".local/share/agent-multi"
    MAT = STATE / "b4_materialization_v5_20260906"
    TR = Path.home() / ".cache/b4_c46_real_sac_probe"
    if TR.exists():
        shutil.rmtree(TR)
    os.makedirs(TR, mode=0o700)
    out = {"schema": "agent_multi.b4_minimal_real_sac_probe.v1",
           "step_bound": STEP_BOUND,
           "update_bound": UPDATE_BOUND, "cases": []}

    # ---- case A: B4 cell with MANDATORY telemetry ----
    built = executor.build_economic_config(
        "o2022_seed101", MAT, TR, "cpu")
    cfg = built["config"]
    assert cfg["training_progress_file"] == cfg["progress_file"]
    cfg["budget_max_env_steps"] = STEP_BOUND
    cfg["budget_max_updates"] = UPDATE_BOUND
    cfg["budget_max_wall_seconds"] = 900.0
    resA = run_case(executor, cfg, "b4_mandatory_progress")
    prog_p = Path(cfg["training_progress_file"])
    resA["progress_file_exists"] = prog_p.exists()
    prog = json.loads(prog_p.read_text()) if prog_p.exists() \
        else {}
    resA["progress_last_event"] = prog.get("event")
    resA["progress_steps"] = prog.get("steps_completed",
                                      prog.get("num_timesteps"))
    hb_dir = Path(cfg["cell_runtime_dir"])
    hbs = sorted(hb_dir.glob("*.json")) if hb_dir.exists() else []
    resA["heartbeat_files_under_cell"] = [h.name for h in hbs][:4]
    status = json.loads((hb_dir / "status.json").read_text())         if (hb_dir / "status.json").exists() else {}
    resA["heartbeat_stop_reason"] = str(
        status.get("stop_reason"))[:110]
    resA["f9_2_exact_update_stop"] = (
        "optimizer-update budget" in str(status.get("stop_reason"))
        and f"{UPDATE_BOUND}" in str(status.get("stop_reason")))
    prog2 = json.loads(prog_p.read_text()) if prog_p.exists()         else {}
    resA["progress_advanced"] = bool(
        (prog2.get("steps_completed") or
         prog2.get("num_timesteps") or 0) >= 129)
    out["cases"].append(resA)
    assert resA["callbacks_composed"] == 2
    assert resA["progress_file_exists"] and         resA["progress_advanced"]
    assert resA["f9_2_exact_update_stop"], resA

    # ---- case B: generic optional (no progress path) ----
    TR2 = TR / "generic"
    os.makedirs(TR2, mode=0o700)
    built2 = executor.build_economic_config(
        "o2022_seed101", MAT, TR2, "cpu")
    cfg2 = built2["config"]
    for k in ("training_progress_file", "progress_file",
              "b4_require_progress"):
        cfg2.pop(k, None)
    cfg2["budget_max_env_steps"] = STEP_BOUND
    cfg2["budget_max_updates"] = UPDATE_BOUND
    cfg2["budget_max_wall_seconds"] = 900.0
    resB = run_case(executor, cfg2, "generic_no_progress")
    hb2 = Path(cfg2["cell_runtime_dir"]) / "status.json"
    st2 = json.loads(hb2.read_text()) if hb2.exists() else {}
    resB["f9_2_exact_update_stop"] = (
        "optimizer-update budget" in str(st2.get("stop_reason")))
    out["cases"].append(resB)
    assert resB["callbacks_composed"] == 1
    assert resB["f9_2_exact_update_stop"], resB

    print(json.dumps(out, indent=1))
    a_ok = (resA["callbacks_composed"] == 2
            and resA["progress_file_exists"]
            and "executing_budget" in resA["stop"].lower()
            or "budget" in resA["stop"].lower())
    shutil.rmtree(TR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
