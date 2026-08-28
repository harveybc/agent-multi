#!/usr/bin/env python3
"""Bounded CUDA ACTIVE reconciliation check (steps-1-2 correction
order 2026-08-28, point 6). MECHANICS_ONLY.

The prior CUDA evidence covered only the zero-trade case. This check
runs the REAL treatment model on ONE CUDA device — a genuine forward
per step proves device execution — while the EXECUTED action follows a
deterministic SCRIPT (as the order permits: "frozen/scripted policy")
that forces BOTH closure populations on a synthetic series:

1. a backtrader LIFECYCLE close (reversal closes the long through the
   bt order path), then
2. a DIRECT envelope settlement (the short's stop is pierced intrabar
   on a later bar).

The run must end with every authoritative conservation identity exact
and both sources present. Refuses without exactly one CUDA device.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import numpy as np
    import torch
    if torch.cuda.device_count() != 1:
        raise SystemExit("REFUSED: exactly one CUDA device required")

    import pandas as pd

    from agent_plugins.pretrained_branch_loader import (
        load_into_sac_policy)
    from agent_plugins.sac_agent import Plugin as SacPlugin
    from pipeline_plugins import _return_trace as trace_mod
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)
    from tools.dispatch_paired_pretrain_comparison import (
        build_cell_config, verify_cell)

    design = json.loads(
        (REPO / "docs/audits/evidence/"
         "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json"
         ).read_text())
    pretrain_dir = Path(args.pretrain_dir)
    cell = verify_cell(design, pretrain_dir, 101,
                       "pretrained_finetuned")
    out_root = Path(tempfile.mkdtemp(prefix="recon_cuda_active_"))
    cfg = build_cell_config(design, cell, pretrain_dir, out_root,
                            device="cuda")
    cfg.pop("_snapshot_config_sha256")

    # synthetic deterministic series under the REAL strong window
    # (32): flat 100s; from bar 10 every HIGH pierces the SHORT's
    # fixed-fraction stop (105) intrabar while staying below the
    # long's take-profit (110) and above its stop (95) — the long
    # closes ONLY by the scripted reversal (bt lifecycle) and the
    # reversal short settles ON its entry bar (direct settlement)
    n = 400
    closes = [100.0] * n
    highs = [c * 1.0005 for c in closes]
    lows = [c * 0.9995 for c in closes]
    for i in range(10, n):
        highs[i] = 106.0
    feature_columns = list(cfg["feature_columns"])
    frame = {"DATE_TIME": pd.date_range("2024-01-01", periods=n,
                                        freq="4h"),
             "OPEN": closes, "HIGH": highs, "LOW": lows,
             "CLOSE": closes, "VOLUME": 1000.0}
    for column in feature_columns:
        if column not in frame:
            frame[column] = np.linspace(0.0, 1.0, n)
    csv = out_root / "cuda_active.csv"
    pd.DataFrame(frame).to_csv(csv, index=False)

    env_cfg = {
        **cfg, "input_data_file": str(csv), "max_steps": 100,
        "min_equity": 0.0,
        "execution_envelope": {"envelope_mode": "fixed_fraction",
                               "sl_fraction": 0.05,
                               "tp_fraction": 0.10,
                               "leverage_cap": 1.0},
        "continuous_action_threshold": 0.0,
    }
    env = _load_env_plugin("gym_fx_env", env_cfg).make_env(env_cfg)

    smoke_cfg = {**cfg, "learning_starts": 10_000}  # no updates
    model = SacPlugin(smoke_cfg).build(env, smoke_cfg)
    transfer = load_into_sac_policy(
        model, pretrain_dir, REPO, Path(cfg["input_data_file"]),
        expected_seal_manifest_sha256=cfg[
            "pretrained_branch_expected_seal"])

    # deterministic script: long entry -> reversal (bt lifecycle
    # close + short reentry) -> the short's stop pierces -> direct
    # settlement; the model still runs a REAL forward on CUDA each
    # step (device execution proof); its action is NOT executed
    script = [0.0] * 100
    # fraction 0.6: a full-fraction entry is rejected by the broker
    # (cost + commission exceeds cash) — measured, not assumed
    script[2] = 0.6    # enter long (fills on flat bars, no touch)
    for step in range(8, 14):
        # PERSISTING flip (finding 329: entry-fill-bar reversals defer
        # one bar and require the signal to persist): lifecycle close
        # of the long, then a short whose ENTRY bar's high (106)
        # pierces its 105 stop -> DIRECT settlement on the entry bar
        script[step] = -0.6
    rows, done = [], False
    obs, _info = env.reset(seed=7)
    forwards = 0
    for a in script:
        if done:
            break
        _model_action, _ = model.predict(obs, deterministic=True)
        forwards += 1
        obs, _r, term, trunc, info = env.step([float(a)])
        done = bool(term or trunc)
        rows.append({"closed_trades_cumulative": info.get("trades"),
                     "position": info.get("position")})
    torch.cuda.synchronize()
    summary = env.summary()

    recon = trace_mod.reconcile_trace_trades(
        rows, summary["trades_total"],
        terminal_open_positions=1 if rows[-1]["position"] else 0)
    sources = summary["closed_trades_by_source"]
    conserved = (
        summary["trades_won"] + summary["trades_lost"]
        + summary["trades_breakeven"] == summary["trades_total"]
        and sum(sources.values()) == summary["trades_total"]
        and sum(summary["close_reason_counts"].values())
        == summary["trades_total"])
    both_populations = (sources.get("bt_trade_closed", 0) >= 1
                        and sources.get("envelope_direct_settlement",
                                        0) >= 1)
    report = {
        "schema": "agent_multi."
                  "trade_reconciliation_cuda_active_check.v1",
        "classification": "MECHANICS_ONLY",
        "policy": "SCRIPTED deterministic actions (order point 6); "
                  "the treatment model executed a REAL CUDA forward "
                  "per step but its action was not applied",
        "device_class_sanitized": torch.cuda.get_device_name(0),
        "treatment_seal": transfer["seal_manifest_sha256"],
        "cuda_forwards_executed": forwards,
        "trades_total": summary["trades_total"],
        "trades_won": summary["trades_won"],
        "trades_lost": summary["trades_lost"],
        "trades_breakeven": summary["trades_breakeven"],
        "closed_trades_by_source": sources,
        "close_reason_counts": summary["close_reason_counts"],
        "terminal_settlement": recon["terminal_settlement_trades"],
        "final_cumulative": recon["final_cumulative"],
        "conservation_identities_exact": bool(conserved),
        "both_closure_populations_present": bool(both_populations),
        "verdict": ("CUDA_ACTIVE_RECONCILIATION_EXACT"
                    if conserved and both_populations
                    and recon["final_cumulative"]
                    == summary["trades_total"]
                    else "CUDA_ACTIVE_RECONCILIATION_BROKEN"),
    }
    payload = json.dumps(report, indent=1)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload)
    return 0 if report["verdict"] == \
        "CUDA_ACTIVE_RECONCILIATION_EXACT" else 3


if __name__ == "__main__":
    raise SystemExit(main())
