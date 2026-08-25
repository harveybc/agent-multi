"""Materialization-negative tests (C1/C2 + F1/F3).

Reproduction-before fixtures are the audited defects verbatim: the
frozen P1 champion admitted on earlier origins; plural finalists on
sealed 2025; a policy trained after the fit boundary but before score
start (SOTA-C03 bypass); authority smuggled through a non-denylisted key
(SOTA-C04); and the REAL executed-84 vs declared-83 observation drift
with `typical_price` prepended (SOTA-C01).
"""
import importlib.util
from pathlib import Path

import pytest

MOD = Path(__file__).resolve().parents[1] / "tools" / "post_p1_screen_contract.py"
spec = importlib.util.spec_from_file_location("screen_contract", MOD)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

P1_CHAMPION = sc.PolicyIdentity(
    name="p1_champion_seed303",
    fit_data_end="2022-12-31", selection_info_end="2024-12-31")

ORIGIN_2022 = sc.Origin("o2022", "2021-12-31", "2021-12-31",
                        "2022-01-01", "2022-12-31")
ORIGIN_2023 = sc.Origin("o2023", "2022-12-31", "2022-12-31",
                        "2023-01-01", "2023-12-31")
ORIGIN_2024 = sc.Origin("o2024", "2023-12-31", "2023-12-31",
                        "2024-01-01", "2024-12-31")


# --- causal eligibility -------------------------------------------------

def test_frozen_p1_policy_refused_on_earlier_origins():
    for origin in (ORIGIN_2022, ORIGIN_2023):
        with pytest.raises(sc.ScreenContractViolation):
            sc.check_causal_eligibility(P1_CHAMPION, origin)


def test_fit_boundary_bypass_refused():
    # SOTA-C03: trained AFTER the declared fit boundary but before score
    # start — the old check passed this; it must now be refused.
    p = sc.PolicyIdentity("late_fit", "2022-06-30", "2022-06-30")
    origin = sc.Origin("o2023", "2021-12-31", "2022-12-31",
                       "2023-01-01", "2023-12-31")
    with pytest.raises(sc.ScreenContractViolation, match="fit boundary"):
        sc.check_causal_eligibility(p, origin)


def test_selection_boundary_enforced():
    p = sc.PolicyIdentity("late_sel", "2021-12-31", "2022-12-30")
    origin = sc.Origin("o2023", "2021-12-31", "2022-06-30",
                       "2023-01-01", "2023-12-31")
    with pytest.raises(sc.ScreenContractViolation, match="selection boundary"):
        sc.check_causal_eligibility(p, origin)


def test_malformed_date_refused():
    p = sc.PolicyIdentity("bad", "2022-13-45", "2022-12-31")
    with pytest.raises(sc.ScreenContractViolation, match="ISO date"):
        sc.check_causal_eligibility(p, ORIGIN_2023)


def test_causally_eligible_policy_accepted():
    p = sc.PolicyIdentity("causal", "2022-12-31", "2022-12-31")
    assert sc.check_causal_eligibility(p, ORIGIN_2023)


def test_diagnostic_label_admits_only_2024():
    diag = sc.PolicyIdentity("p1", "2022-12-31", "2024-12-31",
                             labels=("diagnostic_2024",))
    assert sc.check_causal_eligibility(diag, ORIGIN_2024)
    with pytest.raises(sc.ScreenContractViolation, match="diagnostic_2024"):
        sc.check_causal_eligibility(diag, ORIGIN_2022)


# --- origin set ---------------------------------------------------------

def test_valid_origin_set_accepted():
    assert sc.validate_origins([ORIGIN_2022, ORIGIN_2023, ORIGIN_2024])


def test_overlapping_origins_refused():
    o1 = sc.Origin("a", "2021-12-31", "2021-12-31",
                   "2022-01-01", "2023-06-30")
    o2 = sc.Origin("b", "2022-12-31", "2022-12-31",
                   "2023-01-01", "2023-12-31")
    with pytest.raises(sc.ScreenContractViolation, match="overlap"):
        sc.validate_origins([o1, o2])


def test_disordered_origin_boundaries_refused():
    bad = sc.Origin("x", "2023-06-30", "2022-01-01",
                    "2022-06-01", "2022-12-31")  # fit_end > score_start
    with pytest.raises(sc.ScreenContractViolation, match="not ordered"):
        sc.validate_origins([bad])


# --- sealed absence -----------------------------------------------------

def test_sealed_date_anywhere_refused():
    cfg = {"folds": [{"score_start": "2025-01-01"}]}
    with pytest.raises(sc.ScreenContractViolation, match="sealed-period"):
        sc.check_sealed_absence(cfg)


def test_materialized_sealed_role_refused():
    cfg = {"roles": {"sealed_test": {"csv": "/tmp/sealed.csv"}}}
    with pytest.raises(sc.ScreenContractViolation, match="sealed_test"):
        sc.check_sealed_absence(cfg)


def test_clean_development_config_accepted():
    cfg = {"folds": [{"score_start": "2022-01-01", "score_end": "2022-12-31"}],
           "roles": {"sealed_test": {"csv": None, "materialized": False}}}
    assert sc.check_sealed_absence(cfg)


# --- release packet -----------------------------------------------------

FINALIST = {"name": "a", "decision_authoritative": True,
            "artifact_sha256": "a" * 64, "config_sha256": "b" * 64,
            "code_commit": "6e7bd128f422939cba21d55b3eaed66739ef515f",
            "ensemble_rule_sha256": "c" * 64}


def test_two_authoritative_refused():
    with pytest.raises(sc.ScreenContractViolation, match="exactly one"):
        sc.check_release_packet([FINALIST, dict(FINALIST, name="b")])


def test_finalist_without_frozen_digests_refused():
    with pytest.raises(sc.ScreenContractViolation, match="frozen digest"):
        sc.check_release_packet([{"name": "a",
                                  "decision_authoritative": True}])


def test_report_only_authority_smuggling_refused():
    # SOTA-C04: authority via a key OUTSIDE the old denylist
    with pytest.raises(sc.ScreenContractViolation, match="non-allowlisted"):
        sc.check_release_packet([
            FINALIST,
            {"name": "b", "promotion_condition": "if a drawdown > 10%"}])


def test_typed_report_only_companions_accepted():
    assert sc.check_release_packet([
        FINALIST,
        {"name": "b", "metrics": {"sharpe": 0.4},
         "series_sha256": "d" * 64, "notes": "report-only"}])


# --- observation identity (SOTA-C01) ------------------------------------

DECLARED_83 = {"feature_columns": [f"f{i}" for i in range(83)],
               "include_price_window": True, "include_agent_state": True,
               "window_size": 32}
EXECUTED_84 = {"feature_columns": ["typical_price"] +
               [f"f{i}" for i in range(83)],
               "include_price_window": False, "include_agent_state": True,
               "window_size": 32}


def test_executed_84_vs_declared_83_refused():
    with pytest.raises(sc.ScreenContractViolation,
                       match="84 != declared\n?.*83|feature count"):
        sc.check_observation_identity(EXECUTED_84, DECLARED_83)


def test_same_count_wrong_order_refused():
    exe = dict(DECLARED_83)
    exe["feature_columns"] = list(reversed(DECLARED_83["feature_columns"]))
    with pytest.raises(sc.ScreenContractViolation, match="ORDER"):
        sc.check_observation_identity(exe, DECLARED_83)


def test_price_window_flag_drift_refused():
    exe = dict(DECLARED_83, include_price_window=False)
    with pytest.raises(sc.ScreenContractViolation,
                       match="include_price_window"):
        sc.check_observation_identity(exe, DECLARED_83)


def test_matching_identity_accepted_with_shape():
    out = sc.check_observation_identity(dict(DECLARED_83), DECLARED_83)
    assert out["feature_count"] == 83
    assert out["flattened_shape"] == 32 * 85 + 4  # price window adds 2


def test_executed_identity_labeling_is_honest():
    ident = sc.executed_observation_identity(
        {"effective_config": EXECUTED_84})
    assert ident["executed_feature_count"] == 84
    assert ident["executed_flattened_shape"] == 32 * 84 + 4
    assert ident["executed_include_price_window"] is False
