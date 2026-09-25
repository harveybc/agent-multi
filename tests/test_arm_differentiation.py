"""Differentiation precondition: catches the pathology, spares dead arms.

Every fixture below is a reduction of a real observation from the
2026-08-15 D8 diagnosis; the docstrings name the collection so a future
reader can re-derive the expected verdict from evidence rather than from
this test's opinion.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.arm_differentiation import (  # noqa: E402
    ACCEPT_VERDICTS,
    REFUSE_VERDICTS,
    ArmDifferentiationRefusal,
    ArmObservation,
    assert_arms_differentiated,
    evaluate_arms,
    observations_from_json,
    trace_behavior_facts,
    trace_fingerprint,
)

LIVE = {"mean_weekly_return": 0.0005135560996941659,
        "annualized_return": 0.027086991259871684,
        "total_return": 0.027130832950620354,
        "max_drawdown_fraction": 0.026633483877083208,
        "trades_total": 136}
OTHER = {"mean_weekly_return": 0.00028971633205658414,
         "annualized_return": 0.01517913464126397,
         "total_return": 0.015203560286318085,
         "max_drawdown_fraction": 0.019385696967900334,
         "trades_total": 48}
NULL = {"mean_weekly_return": 0.0, "annualized_return": 0.0,
        "total_return": 0.0, "max_drawdown_fraction": 0.0,
        "trades_total": 0}


def _verdict(a: ArmObservation, b: ArmObservation) -> str:
    return evaluate_arms([a, b])["pairs"][0]["verdict"]


# ---------------------------------------------------------------- accept

def test_distinct_metrics_pass():
    """P1LR c0e53cf18b7d60dd LR=3e-5 column: 42 vs 48 trades separate."""
    a = ArmObservation("P1E_LR3E5", {"dynamics": "easy"}, LIVE,
                       scored_policy_tensor_sha256="0d68901e", trades_total=42)
    b = ArmObservation("P1N_LR3E5", {"dynamics": "normal"}, OTHER,
                       scored_policy_tensor_sha256="e1f39918", trades_total=48)
    assert _verdict(a, b) == "OK"
    assert assert_arms_differentiated([a, b])["differentiated"] is True


def test_zero_trade_dead_arms_are_not_flagged():
    """SITE 3. P1E_LR1E4 vs P1N_LR1E4: distinct tensors (4312de6d vs
    908c09bb), both inactive, both 0 trades, identical null tuple.
    Real degenerate outcome — must NOT fire."""
    a = ArmObservation("P1E_LR1E4", {"dynamics": "easy"}, NULL,
                       scored_policy_tensor_sha256="4312de6d",
                       active=False, trades_total=0)
    b = ArmObservation("P1N_LR1E4", {"dynamics": "normal"}, NULL,
                       scored_policy_tensor_sha256="908c09bb",
                       active=False, trades_total=0)
    assert _verdict(a, b) == "DEGENERATE_IDENTICAL"
    assert assert_arms_differentiated([a, b])["differentiated"] is True


def test_saturated_constant_action_arms_are_not_flagged():
    """SITE 4. usdcad_4h ppo s0: eight feature presets, action_raw==1.0
    for all 28 931 steps. A constant policy ignores its features, so
    identical metrics are forced. Must NOT fire."""
    a = ArmObservation("baseline_12", {"preset": "baseline_12"}, LIVE,
                       scored_policy_tensor_sha256="aaaa",
                       behavior_fingerprint="3f100d54",
                       behavior_degenerate=True, trades_total=1727)
    b = ArmObservation("fx_full", {"preset": "fx_full"}, LIVE,
                       scored_policy_tensor_sha256="bbbb",
                       behavior_fingerprint="3f100d54",
                       behavior_degenerate=True, trades_total=1727)
    assert _verdict(a, b) == "DEGENERATE_IDENTICAL"


def test_same_treatment_is_not_a_comparison():
    """Replicas of one treatment may agree freely."""
    a = ArmObservation("seed101/E4", {"arm": "E4"}, LIVE,
                       scored_policy_tensor_sha256="8137b224")
    b = ArmObservation("seed404/E4", {"arm": "E4"}, LIVE,
                       scored_policy_tensor_sha256="8311dc8d")
    assert _verdict(a, b) == "SAME_TREATMENT"


# ---------------------------------------------------------------- refuse

def test_shared_scored_policy_refuses():
    """SITE 1/2. eth_curriculum_decision_20260807_v2 seed101: E4, EN4_10
    and N14 all scored policy tensor 8137b224 (the untouched warm-start
    anchor) because no post-epoch-0 checkpoint passed the trade gate.
    Container digests differ (zip mtimes) — tensors do not."""
    a = ArmObservation("E4", {"arm": "E4"}, LIVE,
                       scored_policy_tensor_sha256="8137b224")
    b = ArmObservation("N14", {"arm": "N14"}, LIVE,
                       scored_policy_tensor_sha256="8137b224")
    assert _verdict(a, b) == "SHARED_SCORED_POLICY"
    with pytest.raises(ArmDifferentiationRefusal) as excinfo:
        assert_arms_differentiated([a, b])
    assert "SHARED_SCORED_POLICY" in str(excinfo.value)
    assert excinfo.value.report["refused"] == 1


def test_shared_scored_policy_beats_differing_container_hashes():
    """The naive 'identical metrics + different artifact hash' rule would
    call site 1 a metric bug. Tensor identity must win."""
    a = ArmObservation("E4", {"arm": "E4"}, LIVE,
                       scored_policy_tensor_sha256="8137b224",
                       behavior_fingerprint="e061afd1",
                       provenance={"container_sha256": "ce67e6f4"})
    b = ArmObservation("EN4_10", {"arm": "EN4_10"}, LIVE,
                       scored_policy_tensor_sha256="8137b224",
                       behavior_fingerprint="e061afd1",
                       provenance={"container_sha256": "a3dad941"})
    assert _verdict(a, b) == "SHARED_SCORED_POLICY"


def test_metric_collapse_refuses():
    """THE pathology: distinct policies, distinct behaviour, identical
    metric tuple. The measurement pipeline lost the distinction."""
    a = ArmObservation("L2_N", {"arm": "L2_N"}, LIVE,
                       scored_policy_tensor_sha256="1111",
                       behavior_fingerprint="aaaa", trades_total=136)
    b = ArmObservation("L2_EN", {"arm": "L2_EN"}, LIVE,
                       scored_policy_tensor_sha256="2222",
                       behavior_fingerprint="bbbb", trades_total=136)
    assert _verdict(a, b) == "METRIC_COLLAPSE"
    with pytest.raises(ArmDifferentiationRefusal):
        assert_arms_differentiated([a, b])


def test_treatment_not_realized_refuses():
    """Live (non-degenerate) policies replaying identical actions under
    different treatments: the treatment never reached the model."""
    a = ArmObservation("L2_N", {"arm": "L2_N"}, LIVE,
                       scored_policy_tensor_sha256="1111",
                       behavior_fingerprint="same",
                       behavior_degenerate=False, trades_total=136)
    b = ArmObservation("L2_EN", {"arm": "L2_EN"}, LIVE,
                       scored_policy_tensor_sha256="2222",
                       behavior_fingerprint="same",
                       behavior_degenerate=False, trades_total=136)
    assert _verdict(a, b) == "TREATMENT_NOT_REALIZED"
    with pytest.raises(ArmDifferentiationRefusal):
        assert_arms_differentiated([a, b])


def test_identical_metrics_without_behaviour_evidence_refuses():
    """Absence of proof is not proof of health."""
    a = ArmObservation("L2_N", {"arm": "L2_N"}, LIVE,
                       scored_policy_tensor_sha256="1111", trades_total=136)
    b = ArmObservation("L2_EN", {"arm": "L2_EN"}, LIVE,
                       scored_policy_tensor_sha256="2222", trades_total=136)
    assert _verdict(a, b) == "UNDECIDABLE"
    with pytest.raises(ArmDifferentiationRefusal):
        assert_arms_differentiated([a, b])


def test_float_equality_is_exact_not_rounded():
    """Values that print alike at 6 dp must still count as different."""
    a = ArmObservation("A", {"arm": "A"}, dict(LIVE, total_return=0.0271308329),
                       scored_policy_tensor_sha256="1111")
    b = ArmObservation("B", {"arm": "B"},
                       dict(LIVE, total_return=0.02713083290000001),
                       scored_policy_tensor_sha256="2222")
    assert _verdict(a, b) == "OK"


def test_require_informative_flags_all_degenerate_campaign():
    """Sound measurement, unanswerable campaign — opt-in refusal only."""
    a = ArmObservation("L2_N", {"arm": "L2_N"}, NULL, active=False,
                       scored_policy_tensor_sha256="1111", trades_total=0)
    b = ArmObservation("L2_EN", {"arm": "L2_EN"}, NULL, active=False,
                       scored_policy_tensor_sha256="2222", trades_total=0)
    assert assert_arms_differentiated([a, b])["differentiated"] is True
    with pytest.raises(ArmDifferentiationRefusal) as excinfo:
        assert_arms_differentiated([a, b], require_informative=True)
    assert "NO_INFORMATIVE_CONTRAST" in str(excinfo.value)


def test_three_arm_report_shape_and_duplicate_rejection():
    arms = [
        ArmObservation("E4", {"arm": "E4"}, LIVE,
                       scored_policy_tensor_sha256="8137b224"),
        ArmObservation("EN4_10", {"arm": "EN4_10"}, LIVE,
                       scored_policy_tensor_sha256="8137b224"),
        ArmObservation("N14", {"arm": "N14"}, LIVE,
                       scored_policy_tensor_sha256="8137b224"),
    ]
    report = evaluate_arms(arms)
    assert len(report["pairs"]) == 3
    assert report["refused"] == 3
    assert report["differentiated"] is False
    assert all(p["verdict"] == "SHARED_SCORED_POLICY"
               for p in report["pairs"])
    with pytest.raises(ValueError, match="duplicate arm"):
        evaluate_arms([arms[0], arms[0]])


def test_every_verdict_is_classified_accept_or_refuse():
    assert not (ACCEPT_VERDICTS & REFUSE_VERDICTS)


# ------------------------------------------------------- trace helpers

HEADER = "step,seed,run_id,action_raw,equity,trades\n"


def _write(path: Path, rows: str) -> Path:
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def test_trace_fingerprint_ignores_identity_columns(tmp_path):
    """seed101/E4 vs seed404/E4 differ ONLY in the seed column; their
    behaviour is byte-identical and must fingerprint alike."""
    a = _write(tmp_path / "a.csv",
               "1,101,E4,1.0,10000.0,0\n2,101,E4,1.0,10010.0,1\n")
    b = _write(tmp_path / "b.csv",
               "1,404,E4,1.0,10000.0,0\n2,404,E4,1.0,10010.0,1\n")
    assert trace_fingerprint(a) == trace_fingerprint(b)


def test_trace_fingerprint_detects_real_behaviour_change(tmp_path):
    a = _write(tmp_path / "a.csv",
               "1,101,E4,1.0,10000.0,0\n2,101,E4,1.0,10010.0,1\n")
    b = _write(tmp_path / "b.csv",
               "1,101,E4,1.0,10000.0,0\n2,101,E4,-1.0,9990.0,1\n")
    assert trace_fingerprint(a) != trace_fingerprint(b)


def test_trace_behavior_facts_marks_constant_action_degenerate(tmp_path):
    path = _write(tmp_path / "sat.csv",
                  "1,0,r,1.0,10000.0,0\n2,0,r,1.0,10010.0,5\n"
                  "3,0,r,1.0,10020.0,9\n")
    facts = trace_behavior_facts(path)
    assert facts["distinct_actions"] == 1
    assert facts["trades_total"] == 9
    assert facts["behavior_degenerate"] is True


def test_trace_behavior_facts_marks_live_policy_non_degenerate(tmp_path):
    path = _write(tmp_path / "live.csv",
                  "1,0,r,1.0,10000.0,0\n2,0,r,-1.0,10010.0,5\n"
                  "3,0,r,0.2,10020.0,9\n")
    facts = trace_behavior_facts(path)
    assert facts["distinct_actions"] == 3
    assert facts["behavior_degenerate"] is False


def test_trace_behavior_facts_marks_zero_trade_degenerate(tmp_path):
    path = _write(tmp_path / "hold.csv",
                  "1,0,r,0.0,10000.0,0\n2,0,r,0.1,10000.0,0\n")
    facts = trace_behavior_facts(path)
    assert facts["trades_total"] == 0
    assert facts["behavior_degenerate"] is True


def test_trace_fingerprint_rejects_headerless_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no header"):
        trace_fingerprint(path)


# ------------------------------------------------------------ JSON/CLI

def test_observations_from_json_requires_fields():
    with pytest.raises(ValueError, match="missing required field"):
        observations_from_json([{"arm": "A", "treatment": {}}])


def _cli(tmp_path, payload, *extra):
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps(payload), encoding="utf-8")
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "arm_differentiation.py"),
         "--arms-json", str(arms), "--report", str(report), *extra],
        capture_output=True, text=True)
    return proc, json.loads(report.read_text())


def test_cli_exits_zero_on_differentiated_arms(tmp_path):
    proc, report = _cli(tmp_path, [
        {"arm": "A", "treatment": {"t": 1}, "metrics": LIVE,
         "scored_policy_tensor_sha256": "1111"},
        {"arm": "B", "treatment": {"t": 2}, "metrics": OTHER,
         "scored_policy_tensor_sha256": "2222"},
    ])
    assert proc.returncode == 0, proc.stderr
    assert report["differentiated"] is True


def test_cli_exits_two_and_names_the_defect(tmp_path):
    proc, report = _cli(tmp_path, [
        {"arm": "E4", "treatment": {"t": 1}, "metrics": LIVE,
         "scored_policy_tensor_sha256": "8137b224"},
        {"arm": "N14", "treatment": {"t": 2}, "metrics": LIVE,
         "scored_policy_tensor_sha256": "8137b224"},
    ])
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr
    assert report["pairs"][0]["verdict"] == "SHARED_SCORED_POLICY"
