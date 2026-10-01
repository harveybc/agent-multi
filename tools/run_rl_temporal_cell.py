#!/usr/bin/env python
"""Run one materialized RL cell end to end: verify the data pin, train through the
existing pipeline (app.main._run -> rl_pipeline_with_validation), then evaluate the
best policy on the chronological validation episode, write the policy bundle, the
no-trade baseline on the same episode and the rl_temporal_result.v1 record.

    python tools/run_rl_temporal_cell.py --cell <cell.json> --data-root <predictor checkout> \
        --out <dir> [--pilot] [--device cpu|cuda] [--host-alias worker_a]

--pilot: 2 epochs x 2,000 steps, status PILOT_NOT_A_RESULT. Nothing here claims
financial utility; a non-pilot run is still DEVELOPMENT evidence when the manifest
is FROZEN_DEVELOPMENT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows_by_period(csv_path: Path, cfg: dict) -> dict:
    import pandas as pd

    frame = pd.read_csv(csv_path, usecols=[cfg.get("date_column", "DATE_TIME")])
    ts = pd.to_datetime(frame.iloc[:, 0])
    def rows(a, b):
        mask = (ts >= pd.Timestamp(a)) & (ts < pd.Timestamp(b))
        idx = mask.to_numpy().nonzero()[0]
        return [int(idx[0]), int(idx[-1]) + 1] if len(idx) else None
    return {"train_rows": rows(cfg["train_start"], cfg["train_end"]),
            "validation_rows": rows(cfg["validation_start"], cfg["validation_end"]),
            "test_rows": rows(cfg["test_start"], cfg["test_end"]), "total_rows": int(len(ts)),
            "first_timestamp": str(ts.iloc[0]), "last_timestamp": str(ts.iloc[-1])}


def _transitions(summary: dict, out: Path):
    """Transitions actually collected: the pipeline's own count, else the heartbeat's."""
    for key in ("num_timesteps", "total_timesteps_collected"):
        if summary.get(key):
            return int(summary[key])
    hist = summary.get("history") or []
    for h in reversed(hist):
        for key in ("num_timesteps", "timesteps", "steps_after"):
            if isinstance(h, dict) and h.get(key):
                return int(h[key])
    hb = out / "heartbeat.json"
    if hb.exists():
        return int(json.loads(hb.read_text()).get("num_timesteps") or 0) or None
    return None


def _no_trade(factory):
    from rl_temporal.reconciliation import reconcile_episode
    env = factory()
    try:
        return reconcile_episode(env, actions=lambda step, info: 0)
    finally:
        env.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--data-root", required=True, help="local checkout root the GIT_PINNED resource path resolves against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host-alias", default="UNDECLARED")
    ap.add_argument("--pilot-epochs", type=int, default=2)
    ap.add_argument("--pilot-epoch-timesteps", type=int, default=2000)
    args = ap.parse_args(argv)

    import stable_baselines3
    import torch

    from rl_temporal.checkpoint import save_policy_bundle
    from rl_temporal.monitoring import chronological_validation_factory
    from rl_temporal.reconciliation import reconcile_episode
    from rl_temporal.warehouse_binding import validate_result_record

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cell = json.loads(Path(args.cell).read_text())
    if not cell["pilot_gate"]["real_data_fit_allowed"]:
        raise SystemExit(f"REFUSED: pilot gate closed: {cell['pilot_gate']}")
    data = Path(args.data_root) / cell["input_data_file"]
    declared = cell["task"].get("resource", {}).get("sha256") if isinstance(cell["task"].get("resource"), dict) else None
    declared = declared or cell.get("input_data_sha256_declared")
    actual = _sha(data)
    if declared and actual != declared:
        raise SystemExit(f"REFUSED: data sha {actual[:12]} != manifest {declared[:12]}")
    cfg = dict(cell)
    cfg.update(input_data_file=str(data), device=args.device,
               save_model=str(out / "best_policy.zip"), return_trace_dir=str(out / "traces"),
               heartbeat_file=str(out / "heartbeat.json"), training_progress_file=str(out / "progress.json"),
               results_file=str(out / "results.json"), save_config=str(out / "config_out.json"), quiet_mode=True)
    if args.pilot:
        cfg.update(max_epochs=int(args.pilot_epochs), epoch_timesteps=int(args.pilot_epoch_timesteps),
                   l1_patience=10**6, l1_activity_patience=10**6, learning_starts=min(int(cfg.get("learning_starts", 2000)), 500))
    status = "PILOT_NOT_A_RESULT" if args.pilot else ("RESULT" if cell["selected_feature_manifest_status"] == "FROZEN" else "DEVELOPMENT_NOT_CONFIRMATORY")
    rows = _rows_by_period(data, cfg)
    (out / "EPISODES.json").write_text(json.dumps({**rows, "data_sha256": actual, "split": {k: cfg[k] for k in
        ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end")}}, indent=2) + "\n")

    # ---- train through the existing pipeline
    from app.main import _run
    t0 = time.monotonic()
    summary = _run(dict(cfg))
    wall = time.monotonic() - t0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    (out / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    best_path = summary.get("best_model_path")
    policy_kind = "best_validation_checkpoint"
    if not best_path or not Path(best_path).exists():
        best_path = summary.get("terminal_model_path")
        policy_kind = "terminal_inactive_no_eligible_checkpoint"
    if not best_path or not Path(best_path).exists():
        raise SystemExit(f"no policy artifact (best or terminal): {summary.get('stop_reason')}")

    # ---- evaluate the best policy on the chronological validation episode (held out from training)
    from gymnasium.wrappers import FlattenObservation
    from agent_plugins.sac_agent import Plugin as SacPlugin
    from agent_plugins.dqn_agent import Plugin as DqnPlugin
    plugin = SacPlugin() if cell["algorithm"] == "SAC" else DqnPlugin()
    factory = chronological_validation_factory(cfg, train_rows=rows["train_rows"], validation_rows=rows["validation_rows"])
    venv = FlattenObservation(factory())
    try:
        model = plugin.load(best_path, venv)
        def policy(step, info):
            action, _ = model.predict(policy.last_obs, deterministic=True)
            return action
        episode = reconcile_episode(venv, actions=policy, policy_sees_obs=True)
        bundle = save_policy_bundle(model, out / "bundle", env_config={k: v for k, v in cfg.items() if not isinstance(v, dict)},
                                    arm=cell["arm"], extra={"status": status, "evidence_class": cell["pilot_gate"].get("evidence_class")})
    finally:
        venv.close()
    baseline = _no_trade(factory)
    # per-row trace (M05 secondary cut on rows [13699,15888] is taken from this file)
    import csv
    frame_rows = rows["validation_rows"][0] - factory.description["context_rows"]
    with (out / "trace.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["view_row", "step", "bar_index", "action", "raw_action", "position", "position_units", "equity", "pnl", "trades", "is_context_prefix"])
        for tr in episode["trace"]:
            w.writerow([frame_rows + tr["bar_index"] - 1, tr["step"], tr["bar_index"], tr["action"], tr["raw_action"], tr["position"],
                        tr["position_units"], tr["equity"], tr["pnl"], tr["trades"], tr["is_context_prefix"]])
    episode = {k: v for k, v in episode.items() if k != "trace"}
    baseline = {k: v for k, v in baseline.items() if k != "trace"}
    rep = cell["representation"]
    record = {
        "arm": cell["arm"], "algorithm": cell["algorithm"], "representation": rep["kind"],
        "task": {"dataset_id": cell["task"].get("dataset_id"), "selected_feature_manifest_sha256": cell["selected_feature_manifest_sha256"],
                 "manifest_status": cell["selected_feature_manifest_status"], "availability_class": cell["pilot_gate"].get("availability_class"),
                 "selected_variant": cell["selected_variant"]},
        "data": {"model_ready_view_sha256": actual, "train_rows": rows["train_rows"], "validation_rows": rows["validation_rows"], "test_rows": None,
                 "validation_context_rows": factory.description["context_rows"], "episode_definition": factory.description},
        "representation_identity": {"modular_config_sha256": rep.get("modular_config_sha256"), "layout_digest": bundle["representation"].get("layout_digest"),
                                    "regimes": rep.get("regimes") or {"all": "R0"}, "donor_identity": rep.get("donor"),
                                    "parameter_count": cell["accounting"].get("measured", {}).get("unique_parameters_all_networks", cell["accounting"].get("extractor_parameters"))},
        "seed": int(cell["train_seed"]), "paired_seed_group": cell["paired_seed_group"],
        "actions": cell["action_mapping"], "reward": {"plugin": cell["reward_plugin"], "frozen_before_fit": True},
        "evaluation_population": {"split": "validation", "episodes": 1, "rows": rows["validation_rows"], "selection_metric": cell["selection_metric"],
                                  "pipeline_selection": {"best_composite": summary.get("best_composite"), "stop_reason": summary.get("stop_reason"),
                                                         "epochs": len(summary.get("history") or []), "policy_evaluated": policy_kind,
                                                         "activity_eligible_checkpoint": policy_kind == "best_validation_checkpoint"}},
        "costs": episode["costs"],
        "metrics": {k: episode[k] for k in ("net_return", "max_drawdown_fraction", "sharpe", "turnover_units", "trades_closed", "exposure_fraction")},
        "baselines": {"no_trade": {k: baseline[k] for k in ("net_return", "max_drawdown_fraction", "sharpe", "turnover_units", "trades_closed", "exposure_fraction")},
                      "heuristic": "UNAVAILABLE"},
        "resources": {"host_alias": args.host_alias, "device": args.device, "wall_s": wall, "peak_rss_bytes": int(peak_rss),
                      "learn_s": (json.loads((out / "heartbeat.json").read_text()).get("elapsed_s") if (out / "heartbeat.json").exists() else None),
                      "gradient_updates": summary.get("compute_contract", {}).get("measured", {}).get("optimizer_calls") if isinstance(summary.get("compute_contract"), dict) else None,
                      "transitions": _transitions(summary, out),
                      "epochs": len(summary.get("history") or []), "pilot": bool(args.pilot),
                      "pretraining_cost": {"state": "NONE"} if rep.get("regimes_summary", "R0") == "R0" else {"state": "CHARGED_FROM_DONOR_RECEIPT"}},
        "versions": {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__, "engine_pin": rep.get("engine_pin") or "NOT_APPLICABLE"},
        "status": status,
        "caveats": ["checkpoint selected on the same validation episode it is reported on (selection-on-validation); not a held-out number",
                    "DEVELOPMENT availability class; one seed is not a comparison", "metrics exclude the forced-hold context prefix (bars_prefix_excluded)"],
    }
    doc = validate_result_record(record)
    (out / "RESULT.json").write_text(json.dumps({**record, "validation": doc, "episode": episode, "no_trade_episode": baseline}, indent=2, default=str) + "\n")
    per_step = wall / max(1, int(record["resources"]["transitions"] or 1))
    hb = json.loads((out / "heartbeat.json").read_text()) if (out / "heartbeat.json").exists() else {}
    print(json.dumps({"arm": cell["arm"], "seed": cell["train_seed"], "status": status, "wall_s": round(wall, 1),
                      "learn_s": hb.get("elapsed_s"), "policy_evaluated": policy_kind,
                      "transitions": record["resources"]["transitions"], "s_per_transition": round(per_step, 5),
                      "peak_rss_mb": round(peak_rss / 1e6), "net_return": episode["net_return"], "no_trade": baseline["net_return"],
                      "trades": episode["trades_closed"], "stop_reason": summary.get("stop_reason"), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
