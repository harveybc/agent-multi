"""Materialization-negative tests (C1/C2): future-trained policies and
sealed-2025 must be unable to enter development screens; release packets
must carry exactly one decision-authoritative finalist.

Reproduction-before: doc 40 v1 admitted frozen P1 champions (fit through
2022, selection informed by 2024) as Screen-B arm B4 on origins scoring
2022/2023 — the exact fixture below; and doc 38 §23 said "surviving
configurations" (plural) evaluate on sealed 2025.
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

ORIGIN_2022 = sc.Origin("o2022", "2021-12-31", "2022-01-01", "2022-12-31")
ORIGIN_2023 = sc.Origin("o2023", "2022-12-31", "2023-01-01", "2023-12-31")
ORIGIN_2024 = sc.Origin("o2024", "2023-12-31", "2024-01-01", "2024-12-31")


def test_frozen_p1_policy_refused_on_2022_origin():
    with pytest.raises(sc.ScreenContractViolation, match="lookahead"):
        sc.check_causal_eligibility(P1_CHAMPION, ORIGIN_2022)


def test_frozen_p1_policy_refused_on_2023_origin():
    with pytest.raises(sc.ScreenContractViolation, match="lookahead"):
        sc.check_causal_eligibility(P1_CHAMPION, ORIGIN_2023)


def test_fit_timestamp_equal_to_score_start_refused():
    p = sc.PolicyIdentity("edge", "2023-01-01", "2022-06-30")
    with pytest.raises(sc.ScreenContractViolation):
        sc.check_causal_eligibility(p, ORIGIN_2023)


def test_causally_eligible_per_origin_policy_accepted():
    p = sc.PolicyIdentity("causal_o2023", "2022-12-31", "2022-12-31")
    assert sc.check_causal_eligibility(p, ORIGIN_2023)


def test_diagnostic_label_admits_only_2024():
    diag = sc.PolicyIdentity("p1", "2022-12-31", "2024-12-31",
                             labels=("diagnostic_2024",))
    assert sc.check_causal_eligibility(diag, ORIGIN_2024)
    with pytest.raises(sc.ScreenContractViolation, match="diagnostic_2024"):
        sc.check_causal_eligibility(diag, ORIGIN_2022)


def test_sealed_date_anywhere_in_config_refused():
    cfg = {"folds": [{"score_start": "2024-01-01"},
                     {"score_start": "2025-01-01"}]}
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


def test_release_packet_with_two_authoritative_refused():
    with pytest.raises(sc.ScreenContractViolation, match="exactly one"):
        sc.check_release_packet([
            {"name": "a", "decision_authoritative": True},
            {"name": "b", "decision_authoritative": True}])


def test_release_packet_with_zero_authoritative_refused():
    with pytest.raises(sc.ScreenContractViolation, match="exactly one"):
        sc.check_release_packet([{"name": "a"}])


def test_reported_companion_with_fallback_trigger_refused():
    with pytest.raises(sc.ScreenContractViolation, match="fallback"):
        sc.check_release_packet([
            {"name": "a", "decision_authoritative": True},
            {"name": "b", "fallback_trigger": "if a degrades"}])


def test_single_finalist_with_report_only_companions_accepted():
    assert sc.check_release_packet([
        {"name": "a", "decision_authoritative": True},
        {"name": "b"}, {"name": "c"}])
