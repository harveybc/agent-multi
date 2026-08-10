#!/usr/bin/env python3
"""Generic L1 matched-factorial aggregator (repair §7 + order §4/WP3).

Consumes the 16 ``l1_cell_record.json`` (schema v2) decision records of
one experiment identity and emits ONE typed outcome:

    EASY_CONTRIBUTES | LR_ONLY | INTERACTION | EASY_HARMFUL | INCONCLUSIVE

Evidence discipline (findings 178-187):
- every mandatory identity field must be present and VALIDATED against
  the loaded contract/system manifest/disk facts — never copied;
- the terminal artifact and its policy tensor rehash to the producing
  record BEFORE any verification rollout;
- every required raw metric must be present, finite and unit-typed; a
  missing or unreadable results.json, a missing metric or a non-finite
  value is a refusal and forces INCONCLUSIVE;
- total return uses the record's bound initial cash, never a literal;
- subject execution identity and the aggregator's own identity are
  separate output fields;
- cross-record identity uniformity is enforced — one tampered record
  poisons the whole aggregation into INCONCLUSIVE;
- INCONCLUSIVE or any refusal exits nonzero.

Activity facts (§7.1) are DIRECT evidence via terminal probe +
deterministic verification rollout on the cell's own materialized
inner_validation split under forced normal_realistic dynamics.
Invalid is never inactive. Outcomes concern activity survival only.

Spec note: §7.2 "M0.1" binds to the LOWEST phase2_lr_multiplier level
(0.3 in contract v3); the deviation is declared in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools import l1_factorial_screen as runner  # noqa: E402

AGGREGATION_SCHEMA = "agent_multi.l1_factorial_aggregation.v2"
RECORD_SCHEMA = "agent_multi.l1_factorial_cell_record.v2"
OUTCOMES = ("EASY_CONTRIBUTES", "LR_ONLY", "INTERACTION",
            "EASY_HARMFUL", "INCONCLUSIVE")
PROMOTION_OUTCOMES = ("EASY_CONTRIBUTES", "LR_ONLY", "INTERACTION",
                      "EASY_HARMFUL")

MODE_TO_FACTOR = {
    "easy_chronological_continuation": "E",
    "normal_realistic": "N",
}

MANDATORY_IDENTITY_FIELDS = (
    "system_manifest_sha256", "resolved_config_sha256",
    "observation_manifest_sha256", "terminal_model_sha256",
    "terminal_policy_tensor_sha256", "phase1_requested_epochs",
    "phase2_requested_epochs", "subject_code_identity",
    "initial_cash", "cost_contract", "data_sha256",
    "nested_split_contract_sha256", "metric_schema", "env_asset",
    "cell_identity", "stop_reason", "history_len",
)

UNIFORM_ACROSS_RECORDS = (
    "experiment_id", "contract_sha256", "system_manifest_sha256",
    "data_sha256", "nested_split_contract_sha256", "metric_schema",
    "env_asset", "subject_code_identity", "code_revisions",
)

RAW_METRIC_UNITS = {
    "trades_total": "count (closed trades, verification rollout)",
    "mean_weekly_return": "fraction per week (results.json)",
    "total_return": "fraction of bound initial cash",
    "max_drawdown_pct": "percent of peak equity (results.json)",
    "sharpe_ratio": "dimensionless (results.json)",
}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# discovery — refuses duplicate physical records and mixed identities
# ---------------------------------------------------------------------------

def discover_records(output_root: Path, experiment_id: str,
                     contract: dict) -> Tuple[Dict[Tuple[int, str], Path],
                                              List[str]]:
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
            continue
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
                f"duplicate physical record for seed={key[0]} "
                f"cell={key[1]}")
            continue
        found[key] = rec_path
    for seed in seeds:
        for cell in cells:
            if (seed, cell) not in found:
                refusals.append(f"missing record seed={seed} cell={cell}")
    return found, refusals


def cross_record_uniformity(records: Dict[str, dict]) -> List[str]:
    """All records must share one exact experiment/system/code/data
    identity. A field present on some records and absent on others is
    itself a refusal."""
    reasons: List[str] = []
    for field in UNIFORM_ACROSS_RECORDS:
        values = {}
        for name, rec in sorted(records.items()):
            marker = json.dumps(rec.get(field, "<ABSENT>"),
                                sort_keys=True, default=str)
            values.setdefault(marker, []).append(name)
        if len(values) > 1:
            detail = "; ".join(
                f"{marker[:60]}…×{len(names)}" if len(marker) > 60
                else f"{marker}×{len(names)}"
                for marker, names in sorted(values.items()))
            reasons.append(
                f"identity field {field!r} is not uniform across "
                f"records: {detail}")
    return reasons


# ---------------------------------------------------------------------------
# pure validation of one record's bindings — disk facts injected
# ---------------------------------------------------------------------------

def validate_record_bindings(record: dict, *, contract: dict,
                             seed: int, cell: str,
                             experiment_id: str,
                             evidence: dict | None,
                             manifest: dict | None = None,
                             disk_facts: dict | None = None
                             ) -> List[str]:
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

    for field in MANDATORY_IDENTITY_FIELDS:
        if field not in record or record.get(field) in (None, ""):
            reasons.append(f"mandatory identity field {field!r} absent")

    if manifest is not None and \
            record.get("system_manifest_sha256") not in (None, "") and \
            record.get("system_manifest_sha256") != \
            manifest.get("_manifest_sha256"):
        reasons.append("system-manifest drift: record binds a different "
                       "manifest than the loaded one")
    if int(record.get("seed", -1)) != seed or record.get("cell") != cell:
        reasons.append("record seed/cell disagree with its directory")
    if record.get("phase1_mode") != spec["phase1_mode"]:
        reasons.append("phase1_mode differs from contract cell spec")
    if record.get("phase2_lr_multiplier") != spec["phase2_lr_multiplier"]:
        reasons.append("phase2_lr_multiplier differs from contract "
                       "cell spec")
    anchor = contract["anchors"].get(str(seed), {})
    if record.get("anchor_sha256") != anchor.get("sha256"):
        reasons.append("anchor artifact sha does not match contract "
                       "anchor")

    budget = contract.get("decision_budget", {})
    if record.get("phase1_requested_epochs") is not None and \
            budget.get("phase1_epochs") is not None and \
            int(record["phase1_requested_epochs"]) != \
            int(budget["phase1_epochs"]):
        reasons.append("budget drift: phase1_requested_epochs differs "
                       "from the contract decision budget")
    if record.get("phase2_requested_epochs") is not None and \
            budget.get("phase2_max_epochs") is not None and \
            int(record["phase2_requested_epochs"]) != \
            int(budget["phase2_max_epochs"]):
        reasons.append("budget drift: phase2_requested_epochs differs "
                       "from the contract decision budget")

    subject = record.get("subject_code_identity")
    if isinstance(subject, dict):
        for repo_name, ident in sorted(subject.items()):
            if not isinstance(ident, dict) or not ident.get("commit"):
                reasons.append(f"subject code identity for {repo_name} "
                               "lacks a commit")
            elif ident.get("dirty"):
                reasons.append(
                    f"dirty executing source for {repo_name}: decision "
                    "records must come from a clean committed tree "
                    f"(digest {ident.get('dirty_untracked_digest')})")
    elif subject is not None:
        reasons.append("subject_code_identity is malformed")

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
            reasons.append("boundary: transferred policy does not "
                           "hash-match its source")
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
            reasons.append("phase-1 terminal tensor sha != boundary "
                           "source tensor hash")
        updates = post_easy.get("phase1_gradient_updates")
        if not isinstance(updates, int) or updates <= 0:
            reasons.append("phase-1 required gradient updates fact "
                           "missing or zero")

    for field in ("terminal_model_path", "started_utc", "finished_utc"):
        if not record.get(field):
            reasons.append(f"record field {field} missing")

    # Terminal rehash-to-record: VALIDATED against disk facts, never
    # copied. Absent disk facts are themselves a refusal.
    if disk_facts is None:
        reasons.append("terminal disk facts absent: artifact and tensor "
                       "were not rehashed to the record")
    else:
        if disk_facts.get("terminal_model_sha256") != \
                record.get("terminal_model_sha256"):
            reasons.append(
                "terminal replacement: on-disk artifact sha "
                f"{str(disk_facts.get('terminal_model_sha256'))[:12]}… "
                "does not rehash to the record")
        if disk_facts.get("terminal_policy_tensor_sha256") != \
                record.get("terminal_policy_tensor_sha256"):
            reasons.append("terminal tensor digest does not rehash to "
                           "the record")

    nested_path = REPO / contract["nested_split_contract"]
    try:
        nested = json.loads(nested_path.read_text())
    except Exception as exc:
        reasons.append(f"nested split contract unreadable: {exc}")
        nested = {}
    if record.get("nested_split_contract_sha256") not in (None, ""):
        try:
            actual_nested = sysid.sha_file(nested_path)
        except Exception:
            actual_nested = None
        if actual_nested and \
                record["nested_split_contract_sha256"] != actual_nested:
            reasons.append("nested split contract drift vs record")
    if evidence is None:
        reasons.append("return-trace evidence.json missing (asset/data "
                       "binding unverifiable)")
    else:
        expected_sha = nested.get("source_sha256")
        if expected_sha and evidence.get("data_file_hash") != expected_sha:
            reasons.append("asset/data mismatch: evidence data_file_hash "
                           "differs from the pinned nested-split source")
        if record.get("data_sha256") not in (None, "") and \
                expected_sha and record["data_sha256"] != expected_sha:
            reasons.append("record data_sha256 differs from the pinned "
                           "nested-split source")
        asset = str(evidence.get("asset") or "")
        expected_asset = str(contract.get("env_asset") or "")
        if expected_asset and asset != expected_asset:
            reasons.append(f"asset mismatch: evidence asset {asset!r} != "
                           f"contract env_asset {expected_asset!r}")
        if record.get("env_asset") not in (None, "") and \
                expected_asset and record["env_asset"] != expected_asset:
            reasons.append("record env_asset differs from contract")
    return reasons


# ---------------------------------------------------------------------------
# raw metrics — required, finite, unit-typed; failures are refusal reasons
# ---------------------------------------------------------------------------

def raw_metrics(record: dict, rollout_summary: dict | None
                ) -> Tuple[dict, List[str]]:
    reasons: List[str] = []
    rec_dir = Path(record["_record_path"]).parent
    attempt = record.get("attempt_dir")
    results_path = None
    for base in ([Path(attempt)] if attempt else []) + [rec_dir]:
        candidate = Path(base) / "results.json"
        if candidate.is_file():
            results_path = candidate
            break
    metrics: Dict[str, Any] = {"units": RAW_METRIC_UNITS}
    if results_path is None:
        reasons.append("results.json missing — raw metrics unavailable")
        return metrics, reasons
    try:
        results = json.loads(results_path.read_text())
    except Exception as exc:
        reasons.append(f"results.json unreadable: {exc}")
        return metrics, reasons

    initial_cash = record.get("initial_cash")
    if not _finite(initial_cash) or float(initial_cash) <= 0:
        reasons.append("bound initial_cash missing or non-finite; "
                       "total return cannot be derived")
        initial_cash = None

    final_equity = results.get("final_equity")
    metrics["trades_total"] = (None if rollout_summary is None
                               else rollout_summary.get("trades_total"))
    metrics["mean_weekly_return"] = results.get("mean_weekly_return")
    metrics["total_return"] = (
        None if (final_equity is None or initial_cash is None)
        else float(final_equity) / float(initial_cash) - 1.0)
    metrics["max_drawdown_pct"] = results.get("max_drawdown_pct")
    metrics["sharpe_ratio"] = results.get("sharpe_ratio")
    for key in ("trades_total", "mean_weekly_return", "total_return",
                "max_drawdown_pct", "sharpe_ratio"):
        if metrics.get(key) is None:
            reasons.append(f"raw metric {key} absent")
        elif not _finite(metrics[key]):
            reasons.append(f"raw metric {key} non-finite: "
                           f"{metrics[key]!r}")
    return metrics, reasons


# ---------------------------------------------------------------------------
# §7.1 per-seed activity fact — pure given probe + rollout summary
# ---------------------------------------------------------------------------

def activity_facts(*, terminal_probe: dict | None,
                   rollout_summary: dict | None) -> dict:
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

    def _need(summary_key: str, fact_key: str, positive: str) -> None:
        if rollout_summary is None:
            return
        value = rollout_summary.get(summary_key)
        if value is None:
            invalid.append(f"verification rollout lacks {summary_key}")
            return
        ok = float(value) > 0.0
        facts[fact_key] = ok
        facts[fact_key + "_value"] = value
        if not ok:
            facts.setdefault("inactive_signals", []).append(positive)

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

    for mult in multipliers:
        seeds = matrix[mult]
        for seed in sorted(seeds):
            for factor in ("E", "N"):
                cell = seeds[seed].get(factor)
                if cell is None:
                    return "INCONCLUSIVE", (f"missing cell factor={factor} "
                                            f"multiplier={mult} "
                                            f"seed={seed}")
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

    if sum_low != 0 and sum_high != 0 and \
            (sum_low > 0) != (sum_high > 0):
        return "INTERACTION", (f"paired delta sums disagree in sign: "
                               f"LR x{high}={sum_high:+d}, "
                               f"LR x{low}={sum_low:+d}")
    e_low = actives(low, "E")
    n_low = actives(low, "N")
    three_quarters = (3 * n_seeds + 3) // 4
    one_quarter = n_seeds // 4
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

def terminal_disk_facts(record: dict) -> dict | None:
    """Rehash the on-disk terminal artifact and its tensor digest —
    BEFORE any rollout, per order §4."""
    path = record.get("terminal_model_path")
    if not path or not Path(path).is_file():
        return None
    facts = {"terminal_model_sha256": sysid.sha_file(Path(path))}
    try:
        facts["terminal_policy_tensor_sha256"] = \
            runner._terminal_tensor_sha(path)
    except Exception as exc:
        facts["terminal_policy_tensor_sha256"] = None
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def probe_terminal(record: dict, *, agent_name: str = "sac_agent") -> dict:
    from pipeline_plugins.rl_pipeline_with_validation import PipelinePlugin
    from agent_plugins.sac_agent import _policy_tensor_hash

    out: Dict[str, Any] = {"loads": False}
    try:
        rec_dir = Path(record["_record_path"]).parent
        config = _eval_config(record)
        plugin = PipelinePlugin(config)
        agent = runner._agent_plugin(agent_name)
        csv_path = str(Path(record["attempt_dir"]) / "nested_splits" /
                       "inner_validation.csv")
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
            # The boundary rebuild restarts the counter, so the
            # terminal's counter counts phase-2 updates exactly.
            out["phase2_updates_occurred"] = out["terminal_n_updates"] > 0
            phase1 = agent.load(str(post_easy.get("artifact")), env)
            phase1_hash = _policy_tensor_hash(phase1.policy)
            boundary = record.get("boundary_transfer_evidence") or {}
            chain_ok = (phase1_hash == boundary.get(
                "source_policy_tensor_hash"))
            out["tensor_chain_consistent"] = bool(chain_ok)
            out["tensor_chain_detail"] = (
                "phase-1 artifact rehashed to its recorded boundary "
                "source" if chain_ok else
                f"phase-1 artifact rehash {phase1_hash[:12]}… != "
                "recorded boundary source")
        finally:
            try:
                plug.close()
            except Exception:
                pass
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def verification_rollout(record: dict, *,
                         agent_name: str = "sac_agent") -> dict | None:
    from pipeline_plugins.rl_pipeline_with_validation import PipelinePlugin

    csv_path = Path(record["attempt_dir"]) / "nested_splits" / \
        "inner_validation.csv"
    if not csv_path.is_file():
        return None
    config = _eval_config(record)
    plugin = PipelinePlugin(config)
    agent = runner._agent_plugin(agent_name)
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


def _eval_config(record: dict) -> dict:
    """Evaluation config through the SAME system manifest — never the
    legacy base config resolver."""
    contract = runner.load_contract()
    manifest = runner.load_system_manifest()
    config = sysid.materialize_system_config(
        contract, manifest, str(record["cell"]), int(record["seed"]),
        Path(record["attempt_dir"]))
    config.pop("_identity", None)
    config["solvency_mode"] = "normal_realistic"
    config.pop("return_trace_dir", None)
    return config


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def aggregate(output_root: Path, experiment_id: str, *,
              contract: dict | None = None,
              manifest: dict | None = None,
              probe_fn: Callable[[dict], dict] = probe_terminal,
              rollout_fn: Callable[[dict], dict | None] =
              verification_rollout,
              disk_facts_fn: Callable[[dict], dict | None] =
              terminal_disk_facts) -> dict:
    contract = contract or runner.load_contract()
    manifest = manifest or runner.load_system_manifest()
    records_paths, refusals = discover_records(output_root, experiment_id,
                                               contract)
    matrix: Dict[float, Dict[int, Dict[str, dict]]] = {}
    per_cell: Dict[str, Any] = {}
    raw: Dict[str, Any] = {}
    loaded: Dict[str, dict] = {}

    for (seed, cell), rec_path in sorted(records_paths.items()):
        record = json.loads(rec_path.read_text())
        record["_record_path"] = str(rec_path)
        record.setdefault("attempt_dir", str(rec_path.parent))
        loaded[f"seed{seed}/{cell}"] = record

    refusals.extend(cross_record_uniformity(loaded))

    for name, record in sorted(loaded.items()):
        seed = int(record["seed"])
        cell = str(record["cell"])
        rec_path = Path(record["_record_path"])
        evidence_path = Path(record["attempt_dir"]) / "return_traces" / \
            "evidence.json"
        if not evidence_path.is_file():
            evidence_path = rec_path.parent / "return_traces" / \
                "evidence.json"
        evidence = None
        if evidence_path.is_file():
            try:
                evidence = json.loads(evidence_path.read_text())
            except Exception:
                evidence = None
        disk_facts = disk_facts_fn(record)
        reasons = validate_record_bindings(
            record, contract=contract, manifest=manifest, seed=seed,
            cell=cell, experiment_id=experiment_id, evidence=evidence,
            disk_facts=disk_facts)
        if reasons:
            facts = {"valid": False, "active": None,
                     "invalid_reasons": reasons}
            summary = None
        else:
            probe = probe_fn(record)
            summary = rollout_fn(record)
            facts = activity_facts(terminal_probe=probe,
                                   rollout_summary=summary)
        metrics, metric_reasons = raw_metrics(record, summary)
        if metric_reasons:
            refusals.extend(f"{name}: {r}" for r in metric_reasons)
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
        per_cell[name] = facts
        raw[name] = metrics

    outcome, rationale = decide_outcome(matrix, refusals=refusals)
    subject_revisions = sorted({
        json.dumps(rec.get("subject_code_identity"), sort_keys=True,
                   default=str)
        for rec in loaded.values()})
    return {
        "schema": AGGREGATION_SCHEMA,
        "experiment_id": experiment_id,
        "contract_sha256": contract.get("_contract_sha256"),
        "system_manifest_sha256": manifest.get("_manifest_sha256"),
        "outcome": outcome,
        "outcome_rationale": rationale,
        "outcome_domain": "activity survival only; profit does not gate "
                          "this mechanism screen",
        "spec_deviation_declared": (
            "spec §7.2 'M0.1' bound to the contract's lowest "
            "phase2_lr_multiplier level"),
        "refusals": refusals,
        "cells": per_cell,
        "raw_metrics_per_seed": raw,
        "subject_execution_revisions": [
            json.loads(s) if s != "null" else None
            for s in subject_revisions],
        "aggregator_revisions": {
            "agent-multi": sysid.source_tree_identity(
                sysid.resolve_repo_root(Path(__file__))),
        },
    }


def write_aggregation(result: dict, output_root: Path) -> Path:
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
    # INCONCLUSIVE or any refusal exits nonzero (order §4).
    if result["outcome"] == "INCONCLUSIVE" or result["refusals"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
