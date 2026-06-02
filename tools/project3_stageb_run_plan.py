"""Project 3 Stage B locked run-plan expander.

Generates a deterministic Stage B run plan for a Project 3 candidate
(default: ``ETHUSDT 4h SAC tech_stat``) by expanding a reference
config into one locked config per ``(role, seed, cost_scenario)``
combination plus a manifest. **Nothing is launched.** Every emitted
config is marked ``_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED: true`` and
``stage_c_access: "DENIED"``.

Why this exists
---------------
financial-data needs Stage B return traces (per-bar net/gross
returns, cost columns, etc.) for rigorous DSR/PBO/Reality
Check/SPA. agent-multi already emits those traces and a
run-level ``evidence.json`` per run. This tool produces the
deterministic seed × cost-scenario × baseline grid those
evaluators consume — without ever invoking training.

Schema
------
Manifest schema_version: ``project3_stageb_run_plan_v1``.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "project3_stageb_run_plan_v1"
HELDOUT_START = "2025-01-01"

DEFAULT_SEEDS: Sequence[int] = (0, 1, 2, 3, 4)
DEFAULT_COST_SCENARIOS: Sequence[str] = ("base", "plus_50pct", "plus_100pct")
REQUIRED_COST_SCENARIOS: Sequence[str] = DEFAULT_COST_SCENARIOS
DEFAULT_BASELINES: Sequence[str] = (
    "no_trade",
    "buy_and_hold",
    "random",
    "momentum",
    "reversal",
)

# Cost scenarios are applied as overrides on top of the reference
# config. ``base`` is the reference itself (no change). ``pessimistic``
# inflates commission/slippage to stress-test cost fragility — the
# multipliers are conservative defaults; promotion still requires
# matched-baseline evidence under ``base``.
COST_SCENARIO_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "base": {},
    "plus_50pct": {
        "_cost_scenario": "plus_50pct",
        "commission_multiplier": 1.5,
        "slippage_multiplier": 1.5,
    },
    "plus_100pct": {
        "_cost_scenario": "plus_100pct",
        "commission_multiplier": 2.0,
        "slippage_multiplier": 2.0,
    },
    "pessimistic": {
        "_cost_scenario": "pessimistic",
        "commission_multiplier": 3.0,
        "slippage_bps_floor": 2.0,
    },
}

BASELINE_AGENT_PLUGINS: Dict[str, str] = {
    "no_trade": "no_trade_agent",
    "buy_and_hold": "buy_hold_agent",
    "random": "random_agent",
    "momentum": "momentum_agent",
    "reversal": "reversal_agent",
}

# Required reference-config fields (minimum we need to emit a config
# capable of producing a Stage B return trace).
_REQUIRED_REFERENCE_FIELDS = (
    "asset",
    "agent_plugin",
    "pipeline_plugin",
    "env_plugin",
    "input_data_file",
)


class StageBPlanError(RuntimeError):
    """Raised when the run-plan cannot be safely generated."""


# ---------------------------------------------------------------------------
def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_reference_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise StageBPlanError(f"reference config not found: {path}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except json.JSONDecodeError as e:
        raise StageBPlanError(f"reference config not valid JSON: {e}") from e
    if not isinstance(cfg, dict):
        raise StageBPlanError("reference config must be a JSON object")
    return cfg


def _validate_reference_config(cfg: Mapping[str, Any]) -> None:
    missing = [k for k in _REQUIRED_REFERENCE_FIELDS if not cfg.get(k)]
    if missing:
        raise StageBPlanError(
            f"reference config missing required field(s): {missing}"
        )
    # Stage C must never leak in here.
    if cfg.get("stage_c_access") and str(cfg["stage_c_access"]).upper() != "DENIED":
        raise StageBPlanError(
            "reference config has non-DENIED stage_c_access; refusing to "
            "expand a Stage B plan"
        )
    if cfg.get("final_stage_c_evaluation") or cfg.get("stage_c_acknowledged"):
        raise StageBPlanError(
            "reference config has Stage C authorization flags set; this tool "
            "only generates Stage B (preheldout) plans"
        )


def _parse_iso_date(s: str) -> Optional[_dt.date]:
    s = str(s).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1]
    s = s.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return _dt.datetime.strptime(s.strip()[: len(fmt) + 6], fmt).date()
        except ValueError:
            continue
    try:
        return _dt.datetime.fromisoformat(s.strip()).date()
    except ValueError:
        return None


def _read_first_last_timestamp(
    csv_path: str, date_column: str,
) -> Optional[Dict[str, str]]:
    """Return ``{"first": str, "last": str}`` or None if unavailable."""
    p = Path(csv_path)
    if not p.exists():
        return None
    import csv
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return None
        try:
            idx = header.index(date_column)
        except ValueError:
            return None
        first_row = next(reader, None)
        if first_row is None:
            return None
        first_ts = first_row[idx]
        last_ts = first_ts
        for row in reader:
            if row and len(row) > idx and row[idx]:
                last_ts = row[idx]
        return {"first": first_ts, "last": last_ts}


def _enforce_heldout_safety(
    cfg: Mapping[str, Any], *, allow_missing_data: bool,
) -> Dict[str, Any]:
    """Refuse if the reference config would consume Stage C data.

    Heuristic: read the first/last timestamps of ``input_data_file``,
    compute the end of the (train+val+test) window from
    ``train_years`` / ``val_years`` / ``test_years`` (defaults 4/1/1),
    and require it to be strictly before HELDOUT_START.

    If the data file cannot be read and ``allow_missing_data`` is
    True, returns a report entry but does not raise.
    """
    csv_path = cfg.get("input_data_file")
    date_col = cfg.get("date_column", "DATE_TIME")
    bounds = _read_first_last_timestamp(str(csv_path), str(date_col))
    if bounds is None:
        if not allow_missing_data:
            raise StageBPlanError(
                f"could not read first/last timestamp from {csv_path!r}; "
                "pass --allow-missing-data to skip this check (e.g. for "
                "smoke tests)"
            )
        return {
            "data_first_timestamp": None,
            "data_last_timestamp": None,
            "usable_end_date": None,
            "heldout_check_skipped": True,
        }
    first_date = _parse_iso_date(bounds["first"])
    last_date = _parse_iso_date(bounds["last"])
    if first_date is None or last_date is None:
        raise StageBPlanError(
            f"unparseable {date_col!r} timestamps in {csv_path!r}: "
            f"first={bounds['first']!r} last={bounds['last']!r}"
        )
    train_years = int(cfg.get("train_years", 4))
    val_years = int(cfg.get("val_years", 1))
    test_years = int(cfg.get("test_years", 1))
    span_years = train_years + val_years + test_years
    usable_end = _dt.date(
        first_date.year + span_years, first_date.month, first_date.day,
    )
    heldout_date = _dt.date(2025, 1, 1)
    if usable_end >= heldout_date:
        raise StageBPlanError(
            f"reference config would consume data at or after {HELDOUT_START} "
            f"(first={first_date}, span={span_years}y, "
            f"usable_end={usable_end}); Stage C is forbidden in Stage B "
            "run plans"
        )
    return {
        "data_first_timestamp": bounds["first"],
        "data_last_timestamp": bounds["last"],
        "usable_end_date": usable_end.isoformat(),
        "heldout_check_skipped": False,
    }


# ---------------------------------------------------------------------------
def _candidate_metadata_from_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    asset_field = str(cfg.get("asset", "unknown"))
    # ``asset`` in Project 3 configs is typically "ethusdt_4h" — split.
    asset, _, timeframe = asset_field.partition("_")
    timeframe = timeframe or str(cfg.get("timeframe", ""))
    algo = "sac"
    plugin = str(cfg.get("agent_plugin", ""))
    if "ppo" in plugin.lower():
        algo = "ppo"
    elif "dqn" in plugin.lower():
        algo = "dqn"
    return {
        "asset": asset.upper() or asset_field,
        "timeframe": timeframe,
        "algorithm": algo,
        "feature_preset": cfg.get("features_preset"),
    }


def _config_filename(*, role: str, baseline: Optional[str], seed: int,
                     cost_scenario: str, candidate_id: str) -> str:
    if role == "baseline" and baseline:
        return f"stageb_{candidate_id}_baseline_{baseline}_s{seed}_{cost_scenario}.json"
    return f"stageb_{candidate_id}_candidate_s{seed}_{cost_scenario}.json"


def _build_locked_config(
    *,
    reference: Mapping[str, Any],
    candidate_id: str,
    role: str,
    baseline: Optional[str],
    seed: int,
    cost_scenario: str,
    output_root: Path,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(reference))

    # Stage B lock fields — these must be preserved verbatim.
    cfg["_project3_stage_b_lock"] = True
    cfg["_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"] = True
    cfg["heldout_start"] = HELDOUT_START
    cfg["stage_c_access"] = "DENIED"
    cfg["final_stage_c_evaluation"] = False
    cfg["stage_c_acknowledged"] = False

    # Seeds — explicit and matched across roles.
    cfg["train_seed"] = int(seed)
    cfg["eval_seed"] = int(seed)

    # Cost scenario overrides (transparent; preserved in the locked config).
    cfg["_cost_scenario"] = cost_scenario
    overrides = COST_SCENARIO_OVERRIDES.get(cost_scenario, {})
    for k, v in overrides.items():
        cfg[k] = v

    # Role wiring.
    cfg["_stage_b_role"] = role
    if role == "baseline":
        cfg["_stage_b_baseline_name"] = baseline
        baseline_plugin = BASELINE_AGENT_PLUGINS.get(str(baseline or ""))
        cfg["_baseline_promotion_eligible"] = False
        cfg["_baseline_template_only"] = baseline_plugin is None
        if baseline_plugin is not None:
            cfg["agent_plugin"] = baseline_plugin
            cfg["total_timesteps"] = 0
            cfg["load_model"] = None
            cfg["eval_deterministic"] = True
    elif int(cfg.get("total_timesteps") or 0) <= 0:
        epoch_timesteps = int(cfg.get("epoch_timesteps") or 0)
        max_epochs = int(cfg.get("max_epochs") or 0)
        cfg["total_timesteps"] = max(1, epoch_timesteps * max_epochs)

    # Deterministic per-run output paths.
    run_dirname = (
        f"{candidate_id}_{role}"
        + (f"_{baseline}" if role == "baseline" and baseline else "")
        + f"_s{seed}_{cost_scenario}"
    )
    run_dir = output_root / "runs" / run_dirname
    cfg["save_model"] = str(run_dir / "policy.zip")
    cfg["results_file"] = str(run_dir / "summary.json")
    cfg["save_config"] = str(run_dir / "config_out.json")
    cfg["progress_file"] = str(run_dir / "training_progress.json")
    cfg["training_progress_file"] = str(run_dir / "training_progress.json")

    # Trace + evidence wiring (consumed by financial-data's evaluator).
    trace_dir = run_dir / "return_traces"
    cfg["return_trace_dir"] = str(trace_dir)
    cfg["return_trace_file"] = str(trace_dir / "evaluation_return_trace.csv")

    return cfg


def _expected_evidence_file(cfg: Mapping[str, Any]) -> str:
    trace_dir = cfg.get("return_trace_dir")
    if trace_dir:
        return str(Path(trace_dir) / "evidence.json")
    trace_file = cfg.get("return_trace_file")
    if trace_file:
        return str(Path(trace_file).parent / "evidence.json")
    return ""


# ---------------------------------------------------------------------------
def build_run_plan(
    *,
    reference_config_path: str,
    candidate_id: str,
    output_dir: str,
    seeds: Sequence[int],
    cost_scenarios: Sequence[str],
    baselines: Sequence[str],
    allow_too_few_seeds: bool = False,
    allow_missing_data: bool = False,
    allow_missing_cost_scenarios: bool = False,
    write_files: bool = True,
) -> Dict[str, Any]:
    """Build (and optionally write) the Stage B run plan.

    Returns the manifest dict. When ``write_files`` is True, also
    writes the manifest JSON/MD and one locked config per cell.
    """
    if len(seeds) < 5 and not allow_too_few_seeds:
        raise StageBPlanError(
            f"Stage B requires at least 5 paired seeds, got {len(seeds)}; "
            "pass --allow-too-few-seeds-for-smoke-test to override (the "
            "manifest will mark the plan non-promotable)"
        )
    if not allow_missing_cost_scenarios:
        missing_costs = set(REQUIRED_COST_SCENARIOS) - set(cost_scenarios)
        if missing_costs:
            raise StageBPlanError(
                f"Stage B requires cost scenarios {set(REQUIRED_COST_SCENARIOS)!r}; "
                f"missing {sorted(missing_costs)}; pass "
                "--allow-cost-scenario-override to override"
            )
    unknown_costs = [c for c in cost_scenarios if c not in COST_SCENARIO_OVERRIDES]
    if unknown_costs:
        raise StageBPlanError(
            f"unknown cost scenarios: {unknown_costs!r}; known: "
            f"{sorted(COST_SCENARIO_OVERRIDES)}"
        )

    reference = _load_reference_config(reference_config_path)
    _validate_reference_config(reference)
    heldout_report = _enforce_heldout_safety(
        reference, allow_missing_data=allow_missing_data,
    )

    candidate_meta = _candidate_metadata_from_config(reference)
    output_root = Path(output_dir)
    if write_files:
        (output_root / "configs").mkdir(parents=True, exist_ok=True)
        (output_root / "runs").mkdir(parents=True, exist_ok=True)

    reference_sha = _sha256_file(reference_config_path)

    config_entries: List[Dict[str, Any]] = []

    def _emit(
        *, role: str, baseline: Optional[str], seed: int, cost_scenario: str,
    ) -> None:
        cfg = _build_locked_config(
            reference=reference,
            candidate_id=candidate_id,
            role=role,
            baseline=baseline,
            seed=seed,
            cost_scenario=cost_scenario,
            output_root=output_root,
        )
        # Strict post-build invariants — defense in depth.
        assert cfg.get("_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED") is True
        assert cfg.get("stage_c_access") == "DENIED"
        assert cfg.get("return_trace_dir") or cfg.get("return_trace_file")

        filename = _config_filename(
            role=role, baseline=baseline, seed=seed,
            cost_scenario=cost_scenario, candidate_id=candidate_id,
        )
        config_path = output_root / "configs" / filename
        if write_files:
            with config_path.open("w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2, sort_keys=True, default=str)

        run_dir = str(Path(str(cfg.get("save_model", ""))).parent)
        entry = {
            "role": role,
            "baseline_name": baseline,
            "seed": int(seed),
            "cost_scenario": cost_scenario,
            "config_file": str(config_path),
            "agent_plugin": cfg.get("agent_plugin"),
            "run_dir": run_dir,
            "progress_file": cfg.get("progress_file"),
            "return_trace_dir": cfg.get("return_trace_dir"),
            "return_trace_file": cfg.get("return_trace_file"),
            "expected_evidence_file": _expected_evidence_file(cfg),
            "promotion_eligible": role == "candidate",
            "template_only": bool(cfg.get("_baseline_template_only", False)),
        }
        config_entries.append(entry)

    for seed in seeds:
        for cost in cost_scenarios:
            _emit(role="candidate", baseline=None, seed=seed, cost_scenario=cost)
            for bl in baselines:
                _emit(role="baseline", baseline=bl, seed=seed, cost_scenario=cost)

    promotion_blockers: List[str] = []
    if len(seeds) < 5:
        promotion_blockers.append("TOO_FEW_PAIRED_SEEDS")
    if set(REQUIRED_COST_SCENARIOS) - set(cost_scenarios):
        promotion_blockers.append("INSUFFICIENT_COST_SCENARIOS")
    if heldout_report.get("heldout_check_skipped"):
        promotion_blockers.append("HELDOUT_DATA_CHECK_SKIPPED")
    if any(e["template_only"] for e in config_entries):
        promotion_blockers.append("BASELINES_TEMPLATE_ONLY")

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utcnow_iso(),
        "reference_config": str(Path(reference_config_path).resolve()),
        "reference_config_sha256": reference_sha,
        "candidate_id": candidate_id,
        "algorithm": candidate_meta["algorithm"],
        "asset": candidate_meta["asset"],
        "timeframe": candidate_meta["timeframe"],
        "feature_preset": candidate_meta["feature_preset"],
        "heldout_start": HELDOUT_START,
        "stage_c_access": "DENIED",
        "training_launched": False,
        "seeds": list(seeds),
        "cost_scenarios": list(cost_scenarios),
        "baselines": list(baselines),
        "configs": config_entries,
        "data_bounds": heldout_report,
        "promotion_rules": {
            "minimum_paired_seeds": 5,
            "candidate_must_beat_matched_baseline_under_base_cost": True,
            "stress_costs_must_not_be_catastrophic": True,
            "required_cost_scenarios": list(REQUIRED_COST_SCENARIOS),
            "stage_c_forbidden": True,
        },
        "promotion_blockers": promotion_blockers,
        "promotion_eligible": not promotion_blockers,
    }

    if write_files:
        json_path = output_root / "stageb_run_plan_manifest.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        md_path = output_root / "stageb_run_plan_manifest.md"
        md_path.write_text(_render_manifest_markdown(manifest), encoding="utf-8")
        manifest["manifest_file"] = str(json_path.resolve())
        manifest["manifest_markdown_file"] = str(md_path.resolve())

    return manifest


def _render_manifest_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        f"# Project 3 Stage B Run Plan — {manifest['candidate_id']}",
        "",
        f"- schema_version: `{manifest['schema_version']}`",
        f"- generated_at: {manifest['generated_at']}",
        f"- algorithm: `{manifest['algorithm']}`",
        f"- asset: `{manifest['asset']}` timeframe: `{manifest['timeframe']}`",
        f"- feature_preset: `{manifest['feature_preset']}`",
        f"- heldout_start: `{manifest['heldout_start']}`",
        f"- stage_c_access: **{manifest['stage_c_access']}**",
        f"- training_launched: **{manifest['training_launched']}**",
        f"- seeds: {manifest['seeds']}",
        f"- cost_scenarios: {manifest['cost_scenarios']}",
        f"- baselines: {manifest['baselines']}",
        f"- promotion_eligible: **{manifest['promotion_eligible']}**",
    ]
    if manifest["promotion_blockers"]:
        lines.append(f"- promotion_blockers: {manifest['promotion_blockers']}")
    lines += [
        "",
        f"## Configs ({len(manifest['configs'])})",
        "",
        "| role | baseline | seed | cost | config |",
        "|------|----------|------|------|--------|",
    ]
    for c in manifest["configs"]:
        lines.append(
            f"| {c['role']} | {c.get('baseline_name') or ''} | "
            f"{c['seed']} | {c['cost_scenario']} | "
            f"`{Path(c['config_file']).name}` |"
        )
    lines.append("")
    lines.append("> NOTE: every emitted config is locked with "
                 "`_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED: true` and "
                 "`stage_c_access: DENIED`. This tool never launches "
                 "training.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def _parse_csv_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a Project 3 Stage B locked run plan. "
                    "Does not launch training."
    )
    p.add_argument("--reference-config", required=True,
                   help="Path to the reference Project 3 config JSON.")
    p.add_argument("--candidate-id", default="ethusdt_4h_sac_tech_stat",
                   help="Stable candidate identifier used in filenames.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to receive manifest + locked configs.")
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="Comma-separated seed list (default: 0,1,2,3,4).")
    p.add_argument("--cost-scenarios", default=",".join(DEFAULT_COST_SCENARIOS),
                   help="Comma-separated cost scenarios.")
    p.add_argument("--baselines",
                   default=",".join(DEFAULT_BASELINES),
                   help="Comma-separated baseline names.")
    p.add_argument("--allow-too-few-seeds-for-smoke-test", action="store_true")
    p.add_argument("--allow-cost-scenario-override", action="store_true")
    p.add_argument("--allow-missing-data", action="store_true",
                   help="Skip the data-file heldout check (smoke tests "
                        "only). Will mark the plan non-promotable.")
    p.add_argument("--dry-run-validate-only", action="store_true",
                   help="Validate inputs and print summary; write nothing.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        manifest = build_run_plan(
            reference_config_path=args.reference_config,
            candidate_id=args.candidate_id,
            output_dir=args.output_dir,
            seeds=_parse_csv_int_list(args.seeds),
            cost_scenarios=_parse_csv_str_list(args.cost_scenarios),
            baselines=_parse_csv_str_list(args.baselines),
            allow_too_few_seeds=args.allow_too_few_seeds_for_smoke_test,
            allow_missing_data=args.allow_missing_data,
            allow_missing_cost_scenarios=args.allow_cost_scenario_override,
            write_files=not args.dry_run_validate_only,
        )
    except StageBPlanError as e:
        print(f"[stageb-run-plan] ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps({
        "schema_version": manifest["schema_version"],
        "candidate_id": manifest["candidate_id"],
        "config_count": len(manifest["configs"]),
        "promotion_eligible": manifest["promotion_eligible"],
        "promotion_blockers": manifest["promotion_blockers"],
        "manifest_file": manifest.get("manifest_file"),
        "training_launched": manifest["training_launched"],
        "stage_c_access": manifest["stage_c_access"],
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
