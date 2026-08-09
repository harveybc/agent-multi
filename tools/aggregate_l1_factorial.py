#!/usr/bin/env python3
"""Generic L1 matched-factorial aggregator (repair spec §7.1/§7.2).

Consumes the 16 ``l1_cell_record.json`` decision records of one
experiment identity and emits ONE typed outcome:

    EASY_CONTRIBUTES | LR_ONLY | INTERACTION | EASY_HARMFUL | INCONCLUSIVE

Activity facts (§7.1) are DIRECT evidence, never trusted telemetry: the
terminal artifact is loaded, its update counter compared against the
phase-1 artifact, and a fresh deterministic verification rollout is run
on the cell's own materialized inner_validation split under
normal_realistic dynamics. Missing or contradictory facts make a cell
INVALID — never "inactive" — and any invalid cell forces INCONCLUSIVE
(§7.2 rule 1). Outcomes concern activity survival only; profit never
gates this mechanism screen.

Spec note: §7.2 names the reduced-LR level "M0.1". This contract family
carries the reduced level as a generic ``phase2_lr_multiplier``; the
rules bind to the LOWEST multiplier level (0.3 in contract v3). The
deviation is declared in the output, not silently absorbed.

The decision core (``validate_record_bindings``, ``activity_facts``,
``decide_outcome``) is pure so mutation tests can prove that malformed
cells, duplicate physical records, contract drift, tensor mismatch,
asset mismatch and absent metrics all yield INCONCLUSIVE/refusal and
never a promotion outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import l1_factorial_screen as runner  # noqa: E402
from tools import eth_curriculum_decision_experiment as d1  # noqa: E402

AGGREGATION_SCHEMA = "agent_multi.l1_factorial_aggregation.v1"
RECORD_SCHEMA = "agent_multi.l1_factorial_cell_record.v1"
OUTCOMES = ("EASY_CONTRIBUTES", "LR_ONLY", "INTERACTION",
            "EASY_HARMFUL", "INCONCLUSIVE")
PROMOTION_OUTCOMES = ("EASY_CONTRIBUTES", "LR_ONLY", "INTERACTION",
                      "EASY_HARMFUL")

MODE_TO_FACTOR = {
    "easy_chronological_continuation": "E",
    "normal_realistic": "N",
}

RAW_METRIC_UNITS = {
    "trades_total": "count (closed trades, verification rollout)",
    "mean_weekly_return": "fraction per week (results.json)",
    "total_return": "fraction of initial cash (final_equity/initial-1)",
    "max_drawdown_pct": "percent of peak equity (results.json)",
    "sharpe_ratio": "dimensionless (results.json)",
}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# discovery — refuses duplicate physical records and mixed identities
# ---------------------------------------------------------------------------

def discover_records(output_root: Path, experiment_id: str,
                     contract: dict) -> Tuple[Dict[Tuple[int, str], Path],
                                              List[str]]:
    """Map (seed, cell) -> record path. Any anomaly is a refusal reason."""
    refusals: List[str] = []
    found: Dict[Tuple[int, str], Path] = {}
    seeds = sorted(int(s) for s in contract["anchors"])
    cells = sorted(contract["cells"])
    exp_dir = output_root / experiment_id
    if not exp_dir.is_dir():
        return {}, [f"experiment directory missing: {exp_dir}"]
    for rec_path in sorted(output_root.rglob("l1_cell_record.json")):
        try:
            rec = json.loads(rec_path.read_text())
        except Exception as exc:
            refusals.append(f"unreadable record {rec_path}: {exc}")
            continue
        if rec.get("experiment_id") != experiment_id:
            continue  # other experiments may share the root; not ours
        try:
            key = (int(rec.get("seed", -1)), str(rec.get("cell")))
        except (TypeError, ValueError):
            refusals.append(f"malformed record {rec_path}: unusable "
                            "seed/cell fields")
            continue
        expected_dir = exp_dir / f"seed{key[0]}" / key[1]
        if rec_path.parent != expected_dir:
            refusals.append(
                f"duplicate/misplaced physical record for seed={key[0]} "
                f"cell={key[1]} at {rec_path} (expected {expected_dir})")
            continue
        if key in found:
            refusals.append(
                f"duplicate physical record for seed={key[0]} cell={key[1]}")
            continue
        found[key] = rec_path
    for seed in seeds:
        for cell in cells:
            if (seed, cell) not in found:
                refusals.append(f"missing record seed={seed} cell={cell}")
    return found, refusals


# ---------------------------------------------------------------------------
# pure validation of one record's bindings (§7.1 "missing facts = invalid")
# ---------------------------------------------------------------------------

def validate_record_bindings(record: dict, *, contract: dict, seed: int,
                             cell: str, experiment_id: str,
                             evidence: dict | None) -> List[str]:
    reasons: List[str] = []
    spec = contract["cells"].get(cell)
    if spec is None:
        return [f"cell {cell} absent from contract"]
    if record.get("schema") != RECORD_SCHEMA:
        reasons.append(f"record schema {record.get('schema')!r} != "
                       f"{RECORD_SCHEMA!r}")
    if record.get("evidence_class") != "decision_run":
        reasons.append("evidence_class is not decision_run "
                       f"({record.get('evidence_class')!r}) — smoke and "
                       "unknown classes never aggregate")
    if record.get("decision_eligible") is not True:
        reasons.append("record is not decision_eligible")
    if record.get("experiment_id") != experiment_id:
        reasons.append("experiment identity mismatch (single chain "
                       "identity required)")
    if record.get("contract_sha256") != contract.get("_contract_sha256"):
        reasons.append("contract drift: record contract_sha256 differs "
                       "from the loaded contract")
    if int(record.get("seed", -1)) != seed or record.get("cell") != cell:
        reasons.append("record seed/cell disagree with its directory")
    if record.get("phase1_mode") != spec["phase1_mode"]:
        reasons.append("phase1_mode differs from contract cell spec")
    if record.get("phase2_lr_multiplier") != spec["phase2_lr_multiplier"]:
        reasons.append("phase2_lr_multiplier differs from contract cell spec")
    anchor = contract["anchors"].get(str(seed), {})
    if record.get("anchor_sha256") != anchor.get("sha256"):
        reasons.append("anchor artifact sha does not match contract anchor")

    curriculum = record.get("curriculum") or {}
    post_easy = curriculum.get("post_easy") or {}
    boundary = record.get("boundary_transfer_evidence") or {}
    if not post_easy:
        reasons.append("phase-1 (post_easy) block missing")
    if not boundary:
        reasons.append("boundary transfer evidence missing")
    if post_easy and boundary:
        if boundary.get("policy_hash_matches_source_after_transfer") \
                is not True:
            reasons.append("boundary: transferred policy does not hash-match "
                           "its source")
        src = boundary.get("source_policy_tensor_hash")
        after = boundary.get("target_policy_tensor_hash_after_transfer")
        before = boundary.get("target_policy_tensor_hash_before_transfer")
        if not src or not after or after != src:
            reasons.append("boundary: target-after tensor hash does not "
                           "equal source tensor hash")
        if before and after and before == after:
            reasons.append("boundary: transfer left the target tensor "
                           "unchanged (sham boundary)")
        if boundary.get("source_artifact_sha256") != \
                post_easy.get("artifact_sha256"):
            reasons.append("boundary source artifact sha != phase-1 "
                           "selected artifact sha")
        if post_easy.get("phase1_terminal_policy_tensor_sha256") != src:
            reasons.append("phase-1 terminal tensor sha != boundary source "
                           "tensor hash")
        updates = post_easy.get("phase1_gradient_updates")
        if not isinstance(updates, int) or updates <= 0:
            reasons.append("phase-1 required gradient updates fact missing "
                           "or zero")

    for field in ("terminal_model_path", "started_utc", "finished_utc"):
        if not record.get(field):
            reasons.append(f"record field {field} missing")

    nested_path = REPO / contract["nested_split_contract"]
    try:
        nested = json.loads(nested_path.read_text())
    except Exception as exc:
        reasons.append(f"nested split contract unreadable: {exc}")
        nested = {}
    if evidence is None:
        reasons.append("return-trace evidence.json missing (asset/data "
                       "binding unverifiable)")
    else:
        expected_sha = nested.get("source_sha256")
        if expected_sha and evidence.get("data_file_hash") != expected_sha:
            reasons.append("asset/data mismatch: evidence data_file_hash "
                           "differs from the pinned nested-split source")
        asset = str(evidence.get("asset") or "")
        # The contract's "asset" is a display label ("ETHUSD");
        # "env_asset" is the environment's asset id the evidence
        # actually records. Comparing the label refused every real
        # cell — caught 2026-08-09 before first real aggregation.
        expected_asset = str(contract.get("env_asset") or "")
        if expected_asset and asset != expected_asset:
            reasons.append(f"asset mismatch: evidence asset {asset!r} != "
                           f"contract env_asset {expected_asset!r}")
    return reasons


# ---------------------------------------------------------------------------
# §7.1 per-seed activity fact — pure given probe + rollout summary
# ---------------------------------------------------------------------------

def activity_facts(*, terminal_probe: dict | None,
                   rollout_summary: dict | None) -> dict:
    """Six direct facts. active=True only when ALL are present and true.

    A missing probe/summary or a missing field is an invalid_reason,
    never "inactive".
    """
    facts: Dict[str, Any] = {}
    invalid: List[str] = []

    if not terminal_probe:
        invalid.append("terminal probe absent (artifact not examined)")
    else:
        if terminal_probe.get("loads") is not True:
            invalid.append("terminal artifact failed to load: "
                           + str(terminal_probe.get("error")))
        facts["terminal_loads"] = terminal_probe.get("loads") is True
        facts["terminal_policy_tensor_sha256"] = terminal_probe.get(
            "terminal_policy_tensor_sha256")
        chain_ok = terminal_probe.get("tensor_chain_consistent")
        if chain_ok is not True:
            invalid.append("terminal tensor digest does not extend the "
                           "recorded boundary chain: "
                           + str(terminal_probe.get("tensor_chain_detail")))
        facts["tensor_chain_consistent"] = chain_ok is True
        p2 = terminal_probe.get("phase2_updates_occurred")
        if p2 is None:
            invalid.append("phase-2 update counter fact unavailable")
        elif p2 is not True:
            invalid.append("phase-2 applied zero gradient updates")
        facts["phase2_updates_occurred"] = p2 is True

    def _need(summary_key: str, fact_key: str, positive: str) -> bool | None:
        if rollout_summary is None:
            return None
        value = rollout_summary.get(summary_key)
        if value is None:
            invalid.append(f"verification rollout lacks {summary_key}")
            return None
        ok = float(value) > 0.0
        facts[fact_key] = ok
        facts[fact_key + "_value"] = value
        if not ok:
            facts.setdefault("inactive_signals", []).append(positive)
        return ok

    if rollout_summary is None:
        invalid.append("verification rollout absent")
    else:
        _need("trades_total", "validation_trades_positive",
              "zero closed trades on inner validation")
        _need("action_raw_std", "raw_action_std_positive",
              "raw action standard deviation is zero")
        _need("action_non_hold_rate", "non_hold_rate_positive",
              "policy held on every step")
        diag = rollout_summary.get("execution_diagnostics")
        if not isinstance(diag, dict):
            invalid.append("verification rollout lacks execution "
                           "diagnostics")
        else:
            protected = sum(int(diag.get(k, 0) or 0) for k in (
                "protected_market_entries", "protected_limit_entries",
                "protected_stop_entries"))
            facts["protected_entries_submitted"] = protected
            facts["protected_entry_positive"] = protected >= 1
            if protected < 1:
                facts.setdefault("inactive_signals", []).append(
                    "no protected entry with native SL/TP was submitted")

    facts["invalid_reasons"] = invalid
    if invalid:
        facts["valid"] = False
        facts["active"] = None  # invalid is NEVER inactive
        return facts
    facts["valid"] = True
    facts["active"] = all((
        facts.get("terminal_loads"),
        facts.get("tensor_chain_consistent"),
        facts.get("phase2_updates_occurred"),
        facts.get("validation_trades_positive"),
        facts.get("raw_action_std_positive"),
        facts.get("non_hold_rate_positive"),
        facts.get("protected_entry_positive"),
    ))
    return facts


# ---------------------------------------------------------------------------
# §7.2 exact ordered outcome — pure
# ---------------------------------------------------------------------------

def decide_outcome(matrix: Dict[float, Dict[int, Dict[str, dict]]],
                   *, refusals: List[str]) -> Tuple[str, str]:
    """matrix[multiplier][seed][factor] = activity facts dict.

    factor is "E" or "N". Evaluated exactly in spec §7.2 order.
    """
    if refusals:
        return "INCONCLUSIVE", ("refusals precede evaluation: "
                                + "; ".join(refusals))
    multipliers = sorted(matrix)
    if len(multipliers) != 2:
        return "INCONCLUSIVE", (f"contract shape unsupported: expected "
                                f"exactly 2 LR levels, saw {multipliers}")
    low, high = multipliers[0], multipliers[1]
    if set(matrix[low]) != set(matrix[high]):
        return "INCONCLUSIVE", ("seed sets differ between LR levels: "
                                f"x{low}={sorted(matrix[low])} vs "
                                f"x{high}={sorted(matrix[high])}")

    # Rule 1 — any invalid or missing cell.
    for mult in multipliers:
        seeds = matrix[mult]
        for seed in sorted(seeds):
            for factor in ("E", "N"):
                cell = seeds[seed].get(factor)
                if cell is None:
                    return "INCONCLUSIVE", (f"missing cell factor={factor} "
                                            f"multiplier={mult} seed={seed}")
                if cell.get("valid") is not True:
                    return "INCONCLUSIVE", (
                        f"invalid cell factor={factor} multiplier={mult} "
                        f"seed={seed}: "
                        + "; ".join(cell.get("invalid_reasons", [])))

    def deltas(mult: float) -> Dict[int, int]:
        return {seed: int(bool(matrix[mult][seed]["E"]["active"]))
                - int(bool(matrix[mult][seed]["N"]["active"]))
                for seed in matrix[mult]}

    def actives(mult: float, factor: str) -> int:
        return sum(1 for seed in matrix[mult]
                   if matrix[mult][seed][factor]["active"])

    n_seeds = len(matrix[low])
    sum_low = sum(deltas(low).values())
    sum_high = sum(deltas(high).values())

    # Rule 2 — INTERACTION.
    if sum_low != 0 and sum_high != 0 and \
            (sum_low > 0) != (sum_high > 0):
        return "INTERACTION", (f"paired delta sums disagree in sign: "
                               f"LR x{high}={sum_high:+d}, "
                               f"LR x{low}={sum_low:+d}")
    # Rules 3-5 bind to the LOW multiplier level (spec "M0.1").
    e_low = actives(low, "E")
    n_low = actives(low, "N")
    three_quarters = (3 * n_seeds + 3) // 4  # 3/4 of seeds, ceil
    one_quarter = n_seeds // 4               # 1/4 of seeds, floor
    if e_low >= three_quarters and sum_low >= 2:
        return "EASY_CONTRIBUTES", (f"at LR x{low}: E active {e_low}/"
                                    f"{n_seeds}, paired delta sum "
                                    f"{sum_low:+d} >= +2")
    if n_low >= three_quarters and e_low <= one_quarter:
        return "EASY_HARMFUL", (f"at LR x{low}: N active {n_low}/{n_seeds} "
                                f"while E active only {e_low}/{n_seeds}")
    if n_low >= three_quarters and sum_low <= 0:
        return "LR_ONLY", (f"at LR x{low}: N active {n_low}/{n_seeds} and "
                           f"paired delta sum {sum_low:+d} <= 0")
    return "INCONCLUSIVE", (
        f"complete pattern matched no rule: at LR x{low} E={e_low}/"
        f"{n_seeds} N={n_low}/{n_seeds} sum={sum_low:+d}; at LR x{high} "
        f"sum={sum_high:+d}")


# ---------------------------------------------------------------------------
# impure probes — injectable so mutation tests never need a GPU
# ---------------------------------------------------------------------------

def probe_terminal(record: dict, *, agent_name: str = "sac_agent") -> dict:
    """Load phase-1 and terminal artifacts; compare counters and digests."""
    from pipeline_plugins.rl_pipeline_with_validation import (
        PipelinePlugin, _load_env_plugin)
    from agent_plugins.sac_agent import _policy_tensor_hash

    out: Dict[str, Any] = {"loads": False}
    try:
        rec_dir = Path(record["_record_path"]).parent
        config = _eval_config(record, rec_dir)
        plugin = PipelinePlugin(config)
        agent = d1._agent_plugin(agent_name)
        csv_path = str(rec_dir / "nested_splits" / "inner_validation.csv")
        plug, env = plugin._make_split_env(
            str(config.get("env_plugin", "gym_fx_env")), config, csv_path,
            agent)
        try:
            post_easy = (record.get("curriculum") or {}).get(
                "post_easy") or {}
            terminal = agent.load(str(record["terminal_model_path"]), env)
            out["loads"] = True
            out["terminal_policy_tensor_sha256"] = _policy_tensor_hash(
                terminal.policy)
            out["terminal_n_updates"] = int(
                getattr(terminal, "_n_updates", 0) or 0)
            phase1 = agent.load(str(post_easy.get("artifact")), env)
            out["phase1_n_updates"] = int(
                getattr(phase1, "_n_updates", 0) or 0)
            # The boundary rebuilds the model via load_for_training, so
            # the terminal's counter restarts at 0 there and counts
            # phase-2 updates exactly; the phase-1 counter lives on a
            # separate lineage and must never be subtracted from it.
            out["phase2_updates_occurred"] = out["terminal_n_updates"] > 0
            boundary = record.get("boundary_transfer_evidence") or {}
            start_hash = boundary.get(
                "target_policy_tensor_hash_after_transfer")
            phase1_hash = _policy_tensor_hash(phase1.policy)
            chain_ok = (phase1_hash == boundary.get(
                "source_policy_tensor_hash"))
            out["tensor_chain_consistent"] = bool(chain_ok)
            out["tensor_chain_detail"] = (
                "phase-1 artifact rehashed to its recorded boundary source"
                if chain_ok else
                f"phase-1 artifact rehash {phase1_hash[:12]}… != recorded "
                f"boundary source")
            out["terminal_differs_from_phase2_start"] = (
                out["terminal_policy_tensor_sha256"] != start_hash)
        finally:
            try:
                plug.close()
            except Exception:
                pass
    except Exception as exc:  # any missing fact is invalid, never inactive
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def verification_rollout(record: dict, *,
                         agent_name: str = "sac_agent") -> dict | None:
    """Deterministic rollout of the terminal policy on the cell's own
    inner_validation split under forced normal_realistic dynamics."""
    from pipeline_plugins.rl_pipeline_with_validation import PipelinePlugin

    rec_dir = Path(record["_record_path"]).parent
    csv_path = rec_dir / "nested_splits" / "inner_validation.csv"
    if not csv_path.is_file():
        return None
    config = _eval_config(record, rec_dir)
    plugin = PipelinePlugin(config)
    agent = d1._agent_plugin(agent_name)
    plug, env = plugin._make_split_env(
        str(config.get("env_plugin", "gym_fx_env")), config, str(csv_path),
        agent)
    try:
        model = agent.load(str(record["terminal_model_path"]), env)
        summary = PipelinePlugin._rollout(
            env, agent, model, int(record["seed"]),
            asset=str(config.get("asset", "unknown_asset")),
            split="aggregator_verification",
            run_id=f"aggregate::{record['cell']}",
            episode_id=f"aggregate::{record['cell']}::seed{record['seed']}",
            continuous_threshold=config.get("continuous_action_threshold"),
        )
        summary.pop("_return_trace_rows", None)
        return summary
    finally:
        try:
            plug.close()
        except Exception:
            pass


def _eval_config(record: dict, rec_dir: Path) -> dict:
    """Rebuild the evaluation-relevant slice of the cell config."""
    config = d1._base_config(rec_dir, str(record["cell"]),
                             int(record["seed"]), epoch_timesteps=20000)
    config["solvency_mode"] = "normal_realistic"
    config.pop("return_trace_dir", None)  # never touch the cell's traces
    return config


# ---------------------------------------------------------------------------
# raw per-seed metrics (§7.2 tail: always emitted, with units)
# ---------------------------------------------------------------------------

def raw_metrics(record: dict, rollout_summary: dict | None) -> dict:
    rec_dir = Path(record["_record_path"]).parent
    metrics: Dict[str, Any] = {"units": RAW_METRIC_UNITS}
    results_path = rec_dir / "results.json"
    try:
        results = json.loads(results_path.read_text())
    except Exception:
        metrics["absent"] = "results.json missing or unreadable"
        return metrics
    initial = 10_000.0
    final_equity = results.get("final_equity")
    metrics["trades_total"] = (None if rollout_summary is None
                               else rollout_summary.get("trades_total"))
    metrics["mean_weekly_return"] = results.get("mean_weekly_return")
    metrics["total_return"] = (
        None if final_equity is None else float(final_equity) / initial - 1.0)
    metrics["max_drawdown_pct"] = results.get("max_drawdown_pct")
    metrics["sharpe_ratio"] = results.get("sharpe_ratio")
    missing = [k for k in ("mean_weekly_return", "total_return",
                           "max_drawdown_pct", "sharpe_ratio",
                           "trades_total") if metrics.get(k) is None]
    if missing:
        metrics["absent"] = f"metrics absent: {', '.join(missing)}"
    return metrics


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def aggregate(output_root: Path, experiment_id: str, *,
              contract: dict | None = None,
              probe_fn: Callable[[dict], dict] = probe_terminal,
              rollout_fn: Callable[[dict], dict | None] =
              verification_rollout) -> dict:
    contract = contract or runner.load_contract()
    records, refusals = discover_records(output_root, experiment_id,
                                         contract)
    matrix: Dict[float, Dict[int, Dict[str, dict]]] = {}
    per_cell: Dict[str, Any] = {}
    raw: Dict[str, Any] = {}

    for (seed, cell), rec_path in sorted(records.items()):
        record = json.loads(rec_path.read_text())
        record["_record_path"] = str(rec_path)
        evidence_path = rec_path.parent / "return_traces" / "evidence.json"
        evidence = None
        if evidence_path.is_file():
            try:
                evidence = json.loads(evidence_path.read_text())
            except Exception:
                evidence = None
        reasons = validate_record_bindings(
            record, contract=contract, seed=seed, cell=cell,
            experiment_id=experiment_id, evidence=evidence)
        if reasons:
            facts = {"valid": False, "active": None,
                     "invalid_reasons": reasons}
            summary = None
        else:
            probe = probe_fn(record)
            summary = rollout_fn(record)
            facts = activity_facts(terminal_probe=probe,
                                   rollout_summary=summary)
        spec = contract["cells"].get(cell, {})
        factor = MODE_TO_FACTOR.get(str(spec.get("phase1_mode")))
        mult = spec.get("phase2_lr_multiplier")
        if factor is None or mult is None:
            refusals.append(f"cell {cell}: contract spec lacks a "
                            "recognizable factor/multiplier")
        else:
            slot = matrix.setdefault(float(mult), {}).setdefault(seed, {})
            if factor in slot:
                refusals.append(f"two cells resolve to factor={factor} "
                                f"multiplier={mult} seed={seed}")
            slot[factor] = facts
        per_cell[f"seed{seed}/{cell}"] = facts
        raw[f"seed{seed}/{cell}"] = raw_metrics(record, summary)

    outcome, rationale = decide_outcome(matrix, refusals=refusals)
    return {
        "schema": AGGREGATION_SCHEMA,
        "experiment_id": experiment_id,
        "contract_sha256": contract.get("_contract_sha256"),
        "outcome": outcome,
        "outcome_rationale": rationale,
        "outcome_domain": "activity survival only; profit does not gate "
                          "this mechanism screen",
        "spec_deviation_declared": (
            "spec §7.2 'M0.1' bound to the contract's lowest "
            "phase2_lr_multiplier level; cell_record.v1 lacks a terminal "
            "tensor digest, so the aggregator computes it directly from "
            "the artifact and publishes it here"),
        "refusals": refusals,
        "cells": per_cell,
        "raw_metrics_per_seed": raw,
        "code_revisions": {r: d1._git_rev(r)
                           for r in ("agent-multi", "gym-fx")},
    }


def write_aggregation(result: dict, output_root: Path) -> Path:
    """Append-only publication: identical re-runs are idempotent,
    divergent overwrites are refused."""
    out_dir = output_root / result["experiment_id"] / "aggregation"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=1, sort_keys=True,
                         default=str) + "\n"
    target = out_dir / "l1_factorial_aggregation.json"
    if target.exists():
        if target.read_text() == payload:
            return target
        raise RuntimeError(
            f"refusing to overwrite divergent aggregation at {target}; "
            "published aggregations are append-only")
    target.write_text(payload)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    contract = runner.load_contract()
    root = Path(args.output_root).expanduser() if args.output_root \
        else Path(contract["output_root"]).expanduser()
    result = aggregate(root, args.experiment_id, contract=contract)
    path = write_aggregation(result, root)
    print(json.dumps({"outcome": result["outcome"],
                      "rationale": result["outcome_rationale"],
                      "refusals": len(result["refusals"]),
                      "aggregation": str(path)}))
    return 0 if result["outcome"] in OUTCOMES else 1


if __name__ == "__main__":
    sys.exit(main())
