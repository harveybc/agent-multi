#!/usr/bin/env python3
"""Bounded CUDA end-to-end reconciliation check (runtime order
2026-08-28 §2). MECHANICS_ONLY: runs ONE bounded cell through the
accepted nested trainer on CUDA — outside the campaign driver, no
custody identity, no venue socket, dry-run budget — solely to prove
the corrected closed-trade stream reconciles exactly on device, as
ordered AFTER the CPU result agrees. Refuses unless exactly one CUDA
device is visible."""
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
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--arm", default="control_random_init")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f"REFUSED: {torch.cuda.device_count()} CUDA devices "
            "visible — exactly one for the bounded check")

    from app.plugin_loader import load_plugin
    from tools.dispatch_paired_pretrain_comparison import (
        DRY_RUN_BUDGET, build_cell_config, verify_cell)

    design = json.loads(
        (REPO / "docs/audits/evidence/"
         "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json"
         ).read_text())
    cell = verify_cell(design, Path(args.pretrain_dir), args.seed,
                       args.arm)
    out_root = Path(tempfile.mkdtemp(prefix="recon_cuda_"))
    cfg = build_cell_config(design, cell, Path(args.pretrain_dir),
                            out_root, device="cuda",
                            dry_run_budget=DRY_RUN_BUDGET,
                            attempt_nonce="cuda_recon_check")
    cfg.pop("_snapshot_config_sha256")
    Path(cfg["save_model"]).parent.mkdir(parents=True, exist_ok=True)

    agent_cls, _ = load_plugin("agent.plugins", "sac_agent")
    pipeline_cls, _ = load_plugin("pipeline.plugins",
                                  "rl_pipeline_with_validation")
    final = pipeline_cls(cfg).run_pipeline(
        config=cfg, env_plugin=None, agent_plugin=agent_cls(cfg),
        mode="train")

    splits = {}
    coherent = True
    for name, summary in (final.get("splits") or {}).items():
        if not isinstance(summary, dict) or "trades_total" not in summary:
            continue
        recon = summary.get("trace_trades_reconciliation") or {}
        entry = {
            "trades_total": summary.get("trades_total"),
            "analyzer_trades_total":
                summary.get("analyzer_trades_total"),
            "closed_trades_by_source":
                summary.get("closed_trades_by_source"),
            "open_position_at_end":
                summary.get("open_position_at_end"),
            "terminal_settlement":
                recon.get("terminal_settlement_trades"),
            "final_cumulative": recon.get("final_cumulative"),
        }
        splits[name] = entry
        if entry["final_cumulative"] is not None and \
                entry["final_cumulative"] != entry["trades_total"]:
            coherent = False

    report = {
        "schema": "agent_multi.trade_reconciliation_cuda_check.v1",
        "classification": "MECHANICS_ONLY",
        "device_class_sanitized": torch.cuda.get_device_name(0),
        "arm": args.arm, "seed": args.seed,
        "budget_disclosed": dict(DRY_RUN_BUDGET),
        "stop_reason": final.get("stop_reason"),
        "splits": splits,
        "verdict": ("CUDA_RECONCILIATION_EXACT" if coherent and splits
                    else "CUDA_RECONCILIATION_BROKEN"),
    }
    payload = json.dumps(report, indent=1)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload)
    return 0 if report["verdict"] == "CUDA_RECONCILIATION_EXACT" \
        else 3


if __name__ == "__main__":
    raise SystemExit(main())
