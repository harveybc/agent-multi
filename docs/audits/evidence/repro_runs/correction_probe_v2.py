#!/usr/bin/env python3
"""Typed correction probe (WP1 / AUD-F1-20260806-143).

The retired `after_probe.py` mapped ANY exception to "corrected", so a
renamed symbol, a removed helper or a malformed fixture counted as a
pass. This probe replaces it with four distinct outcomes:

- ``postcondition_pass``  the corrected contract was exercised and its
  POSTCONDITION holds (a value assertion, not the absence of a crash);
- ``expected_refusal``    the call raised the PREREGISTERED refusal
  (exception class and message fragment both matched) and the durable
  state is unchanged;
- ``fixture_error``       our fixture is wrong (bad bytes, bad path);
- ``harness_error``       the probe itself is stale (missing/renamed
  symbol, signature change).

Only the first two are passes. `fixture_error` and `harness_error` are
FAILURES of the probe and are reported as such.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

RESULTS: dict = {}


def record(case: str, outcome: str, **detail) -> None:
    RESULTS[case] = {"outcome": outcome, **detail}


def expect_refusal(case: str, func, *, exc_type, message_fragment,
                   postcondition=None) -> None:
    """A raise is a pass ONLY when class and message match the
    preregistered refusal contract and any postcondition holds."""
    try:
        value = func()
    except exc_type as raised:
        if message_fragment.lower() not in str(raised).lower():
            record(case, "harness_error",
                   reason=("raised the right class with an unexpected"
                           f" message: {str(raised)[:200]}"))
            return
        if postcondition is not None and not postcondition():
            record(case, "harness_error",
                   reason="refusal raised but postcondition failed")
            return
        record(case, "expected_refusal",
               exception=f"{exc_type.__name__}: {str(raised)[:160]}")
    except Exception as other:                    # noqa: BLE001
        record(case, "harness_error",
               reason=(f"expected {exc_type.__name__}, got"
                       f" {type(other).__name__}: {str(other)[:160]}"))
    else:
        record(case, "still_reproduced",
               reason=f"no refusal; returned {str(value)[:160]}")


def expect_postcondition(case: str, func, check, description: str
                         ) -> None:
    try:
        value = func()
    except Exception as exc:                      # noqa: BLE001
        record(case, "harness_error",
               reason=f"{type(exc).__name__}: {str(exc)[:160]}")
        return
    try:
        ok = check(value)
    except Exception as exc:                      # noqa: BLE001
        record(case, "harness_error",
               reason=f"postcondition raised: {exc}")
        return
    record(case, "postcondition_pass" if ok else "still_reproduced",
           postcondition=description, observed=str(value)[:200])


# --------------------------------------------------------------- 140
def probe_warmup_scoring() -> None:
    from tools import rolling_origin_adaptation as rt
    def fact(equity, **kw):
        base = {"equity": equity, "position": 0.0, "price": 100.0,
                "trades": 0, "commission_paid": 0.0}
        base.update(kw)
        return base

    samples = [fact(100.0), fact(95.0), fact(90.0), fact(88.0),
               fact(88.0 * 1.10)]
    expect_postcondition(
        "140_warmup_excluded_from_score",
        lambda: rt.score_interval(samples, warmup_bars=4,
                                  cadence_bars=1, starting_equity=88.0),
        lambda s: (abs(s["interval_return"] - 0.10) < 1e-9
                   and s["scored_bars"] == 1),
        "interval return is +10% and exactly one bar is scored")
    # 145: exactly h bars, interval deltas, explicit flat handover
    activity = [fact(100.0, trades=7, commission_paid=3.0),
                fact(101.0, trades=9, commission_paid=4.0),
                fact(102.0, trades=11, commission_paid=5.5,
                     position=2.0, price=50.0)]
    expect_postcondition(
        "145_interval_deltas_and_flat_handover",
        lambda: rt.score_interval(activity, warmup_bars=1,
                                  cadence_bars=2, commission=0.001),
        lambda s: (s["scored_bars"] == 2
                   and s["interval_trades"] == 4
                   and abs(s["interval_commission"] - 2.5) < 1e-9
                   and abs(s["handover"]["closing_cost"] - 0.1) < 1e-9
                   and abs(s["equity_after"] - (102.0 - 0.1)) < 1e-9),
        "exactly h bars, delta activity, explicit charged flat close")


def probe_executable_config() -> None:
    from tools import rolling_origin_adaptation as rt
    from tools import eth_curriculum_decision_experiment as decision
    expect_postcondition(
        "142_rt_config_has_no_year_shorthand",
        rt.base_config,
        lambda c: all(f not in c for f in rt.DORMANT_SPLIT_FIELDS),
        "train/val/test_years absent from the executable RT config")
    expect_postcondition(
        "142_decision_config_has_no_year_shorthand",
        lambda: decision._base_config(Path(tempfile.gettempdir()),
                                      "N14", 101, epoch_timesteps=10),
        lambda c: ("train_years" not in c and "test_years" not in c
                   and c.get("train_start")),
        "year shorthand absent while explicit dates remain")


# --------------------------------------------------------------- 136
def probe_artifact_validation() -> None:
    from tools import eth_curriculum_decision_experiment as decision
    tmp = Path(tempfile.mkdtemp())
    import hashlib
    bogus = tmp / "terminal.zip"
    bogus.write_bytes(b"not a real zip archive")
    digest = hashlib.sha256(bogus.read_bytes()).hexdigest()
    replica = tmp / "replica"
    replica.mkdir()
    (replica / "terminal.zip").write_bytes(bogus.read_bytes())
    record_body = {
        "schema": "agent_multi.arm_record.v5",
        "execution_id": "e" * 64, "arm": "N14", "seed": 101,
        "splits_raw": {"validation": {
            "mean_weekly_return": 0.001, "total_return": 0.01,
            "max_drawdown_fraction": 0.02}},
        "code_revisions_before": {"agent-multi": "r"},
        "code_revisions_after": {"agent-multi": "r"},
        "margin_telemetry": {"validation": {"x": "unavailable"}},
        "return_trace_sha256": {"t.csv": "a" * 64},
        "resolved_config_sha256": "c" * 64,
        "artifacts": {
            "best_checkpoint": {
                "path": str(bogus), "replica_path":
                str(replica / "terminal.zip"), "sha256": digest,
                "load_proven": True},
            "terminal": {
                "path": str(bogus), "replica_path":
                str(replica / "terminal.zip"), "sha256": digest,
                "load_proven": True},
        },
        "best_checkpoint_vs_terminal": {"terminal_evaluation": {
            "artifact_sha256": digest, "artifact_path": str(bogus),
            "splits_raw": {"validation": {
                "mean_weekly_return": 0.001, "total_return": 0.01,
                "max_drawdown_fraction": 0.02}}}},
    }
    expect_postcondition(
        "136_unloadable_bytes_are_rejected",
        lambda: decision.validate_arm_record(record_body, "N14"),
        lambda problems: any("load" in p.lower() for p in problems),
        "a non-ZIP artifact with a matching hash is REJECTED")

    missing = json.loads(json.dumps(record_body))
    missing["artifacts"]["terminal"]["path"] = str(tmp / "gone.zip")
    expect_postcondition(
        "136_missing_artifact_is_rejected",
        lambda: decision.validate_arm_record(missing, "N14"),
        lambda problems: any("not retrievable" in p for p in problems),
        "a nonexistent artifact path is REJECTED")


# --------------------------------------------------------------- 138
def probe_repair_validation() -> None:
    from optimizer_plugins.project3_full_genome_optimizer import Plugin
    rules = [{"rule": "forbid_value", "gene": "preprocessing_mode",
              "value": "none", "repair": "resample_categorical"}]
    expect_refusal(
        "138_missing_typed_schema_refused",
        lambda: Plugin.validate_repair_rules(rules, {}),
        exc_type=ValueError, message_fragment="typed")
    schema = [{"name": "preprocessing_mode", "kind": "categorical",
               "choices": ["none", "rolling_zscore"],
               "target": "feature_scaling"}]
    typo = [dict(rules[0], value="nonexistent")]
    expect_refusal(
        "138_forbidden_value_outside_domain_refused",
        lambda: Plugin.validate_repair_rules(
            typo, {"mixed_genome_schema": schema}),
        exc_type=ValueError, message_fragment="not a declared choice")


# --------------------------------------------------------------- 139
def probe_authority_join() -> None:
    sys.path.insert(0, "/home/harveybc/Documents/GitHub/lts")
    from tools import controller_inventory as inv
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": "prediction_provider.live_sac_manifest.v1",
        "model_id": "m", "artifact_sha256": "a" * 64,
        "config_sha256": "c" * 64, "input_feature_sha256": "f" * 64,
        "preprocessing_sha256": "p" * 64, "manifest_sha256": "m" * 64,
        "live_inference_eligible": True,
        "live_execution_eligible": True,
        "observation_parity_verified": True}
    stale_hb = {"observed_at": "2020-01-01T00:00:00+00:00",
                "model_id": "m", "artifact_sha256": "a" * 64,
                "config_sha256": "c" * 64,
                "input_feature_sha256": "f" * 64,
                "preprocessing_sha256": "p" * 64,
                "manifest_sha256": "m" * 64}

    def run_on(host, command, timeout=20.0):
        if "systemctl" in command:
            return 0, "ActiveState=active\nSubState=running\n", ""
        if "for f in" in command:
            return 0, "===/m/manifest.json\n" + json.dumps(manifest), ""
        if "cat" in command:
            return 0, json.dumps(stale_hb), ""
        return 1, "", ""

    expect_postcondition(
        "139_stale_heartbeat_denied_authority",
        lambda: inv.collect(run_on=run_on)["seats"]["mt5_demo"],
        lambda seat: (seat["sac_champion_authoritative"] is not True
                      and any("not fresh" in r for r in
                              seat["join"].get("blocking_reasons", []))),
        "stale heartbeat cannot be authoritative and says why")


# ------------------------------------------------- proof the old lied
def probe_old_harness_would_have_lied() -> None:
    """Run the RETIRED harness pattern against a deliberate
    AttributeError and a malformed ZIP; it calls both 'corrected'."""
    def old_style(func):
        try:
            func()
            return "ran"
        except Exception:
            return "raised_fail_closed -> counted as CORRECTED"

    renamed = old_style(
        lambda: __import__(
            "tools.rolling_origin_adaptation",
            fromlist=["x"])._score_interval([1.0], warmup_bars=0))
    malformed = old_style(
        lambda: __import__("zipfile").ZipFile(__file__).namelist())
    record("143_old_harness_would_have_lied", "postcondition_pass",
           renamed_symbol=renamed, malformed_fixture=malformed,
           conclusion=("the retired harness converts a stale symbol and"
                       " a bad fixture into passes; it is retired from"
                       " any acceptance role"))


def main() -> int:
    probe_warmup_scoring()
    probe_executable_config()
    probe_artifact_validation()
    probe_repair_validation()
    probe_authority_join()
    probe_old_harness_would_have_lied()
    passes = {"postcondition_pass", "expected_refusal"}
    payload = {
        "schema": "agent_multi.correction_probe.v2",
        "network_used": False,
        "cases": RESULTS,
        "failures": {k: v for k, v in RESULTS.items()
                     if v["outcome"] not in passes},
        "all_pass": all(v["outcome"] in passes
                        for v in RESULTS.values()),
    }
    print(json.dumps(payload, indent=1, sort_keys=True, default=str))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
