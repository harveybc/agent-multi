from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from pathlib import Path

import pytest

from pipeline_plugins._execution_curriculum import (
    CONTRACT_VERSION,
    COST_KEYS,
    OBSERVATION_NAMES,
    ExecutionCostCurriculum,
    RobustFitnessConfig,
    aggregate_robust_weekly_fitness,
)


PROFILE_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "config"
    / "execution_curriculum"
    / "project3_execution_cost_curriculum_v1.json"
)


def _valid_mapping() -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "curriculum_id": "unit_cost_curriculum_v1",
        "normalization_bounds": {
            "commission_fraction_per_side": 0.001,
            "full_spread_rate": 0.002,
            "slippage_bps_per_side": 10.0,
        },
        "scenarios": [
            {
                "scenario_id": "easy_a",
                "commission_fraction_per_side": 0.00005,
                "full_spread_rate": 0.00010,
                "slippage_bps_per_side": 0.25,
            },
            {
                "scenario_id": "nominal_a",
                "commission_fraction_per_side": 0.00020,
                "full_spread_rate": 0.00040,
                "slippage_bps_per_side": 1.00,
            },
            {
                "scenario_id": "nominal_b",
                "commission_fraction_per_side": 0.00030,
                "full_spread_rate": 0.00060,
                "slippage_bps_per_side": 2.00,
            },
            {
                "scenario_id": "stress_a",
                "commission_fraction_per_side": 0.00075,
                "full_spread_rate": 0.00150,
                "slippage_bps_per_side": 8.00,
            },
        ],
        "phases": [
            {
                "name": "easy_nonzero",
                "start_progress": 0.0,
                "end_progress": 0.3,
                "scenario_weights": [{"scenario_id": "easy_a", "weight": 1.0}],
            },
            {
                "name": "nominal_randomized",
                "start_progress": 0.3,
                "end_progress": 0.8,
                "scenario_weights": [
                    {"scenario_id": "nominal_a", "weight": 1.0},
                    {"scenario_id": "nominal_b", "weight": 2.0},
                ],
            },
            {
                "name": "stress",
                "start_progress": 0.8,
                "end_progress": 1.0,
                "scenario_weights": [{"scenario_id": "stress_a", "weight": 1.0}],
            },
        ],
    }


def test_profile_contract_is_immutable_versioned_and_canonical() -> None:
    contract = ExecutionCostCurriculum.from_json_file(str(PROFILE_PATH))

    assert contract.contract_version == CONTRACT_VERSION
    assert tuple(phase.name for phase in contract.phases) == (
        "easy_nonzero",
        "nominal_randomized",
        "stress",
    )
    assert len(contract.fingerprint) == 64
    assert contract.fingerprint == contract.fingerprint
    assert contract.to_canonical_mapping()["contract_version"] == CONTRACT_VERSION
    with pytest.raises(FrozenInstanceError):
        contract.curriculum_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        contract.scenarios[0] = contract.scenarios[0]  # type: ignore[index]


def test_selection_is_deterministic_bounded_and_exposes_costs() -> None:
    contract = ExecutionCostCurriculum.from_mapping(_valid_mapping())

    first = contract.select(seed=42, training_progress=0.51)
    second = contract.select(seed=42, training_progress=0.51)

    assert first == second
    assert first.phase_name == "nominal_randomized"
    assert first.observable_names == OBSERVATION_NAMES
    assert tuple(first.cost_patch) == (*COST_KEYS, "financing_enabled")
    assert all(0.0 <= value <= 1.0 for value in first.observable_vector)
    assert first.observable_vector[:3] == pytest.approx(
        tuple(
            first.cost_patch[key] / bound
            for key, bound in zip(COST_KEYS, contract.normalization_bounds)
        )
    )
    assert first.observable_vector[3:] == pytest.approx((0.0, 0.51))
    mutable_patch = first.cost_patch
    mutable_patch[COST_KEYS[0]] = 99.0
    assert contract.select(seed=42, training_progress=0.51).cost_patch != mutable_patch


def test_fixed_validation_scenario_bypasses_random_selection() -> None:
    contract = ExecutionCostCurriculum.from_mapping(_valid_mapping())
    first = contract.select(
        seed=1,
        training_progress=0.1,
        episode_index=10,
        scenario_id="stress_a",
    )
    second = contract.select(
        seed=999,
        training_progress=0.1,
        episode_index=0,
        scenario_id="stress_a",
    )
    assert first.scenario_id == second.scenario_id == "stress_a"
    assert first.cost_patch == second.cost_patch


@pytest.mark.parametrize(
    ("progress", "phase"),
    [
        (0.0, "easy_nonzero"),
        (0.299999, "easy_nonzero"),
        (0.3, "nominal_randomized"),
        (0.799999, "nominal_randomized"),
        (0.8, "stress"),
        (1.0, "stress"),
    ],
)
def test_phase_boundaries_are_explicit(progress: float, phase: str) -> None:
    contract = ExecutionCostCurriculum.from_mapping(_valid_mapping())
    assert contract.select(seed=7, training_progress=progress).phase_name == phase


def test_selection_rejects_invalid_seed_or_progress() -> None:
    contract = ExecutionCostCurriculum.from_mapping(_valid_mapping())
    for invalid in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            contract.select(seed=1, training_progress=invalid)
    with pytest.raises(ValueError, match="seed"):
        contract.select(seed=True, training_progress=0.5)
    with pytest.raises(ValueError, match="episode_index"):
        contract.select(seed=1, training_progress=0.5, episode_index=-1)


def test_contract_rejects_negative_nonfinite_or_out_of_bound_costs() -> None:
    for field, invalid in (
        ("commission_fraction_per_side", -0.01),
        ("full_spread_rate", math.nan),
        ("slippage_bps_per_side", math.inf),
        ("slippage_bps_per_side", 11.0),
    ):
        raw = _valid_mapping()
        raw["scenarios"][0][field] = invalid
        with pytest.raises(ValueError):
            ExecutionCostCurriculum.from_mapping(raw)


def test_contract_rejects_bad_phase_names_ranges_weights_and_references() -> None:
    bad_name = _valid_mapping()
    bad_name["phases"][1]["name"] = "something_else"
    with pytest.raises(ValueError, match="exactly"):
        ExecutionCostCurriculum.from_mapping(bad_name)

    gap = _valid_mapping()
    gap["phases"][1]["start_progress"] = 0.4
    with pytest.raises(ValueError, match="contiguous"):
        ExecutionCostCurriculum.from_mapping(gap)

    zero_weights = _valid_mapping()
    zero_weights["phases"][2]["scenario_weights"][0]["weight"] = 0.0
    with pytest.raises(ValueError, match="positive total"):
        ExecutionCostCurriculum.from_mapping(zero_weights)

    unknown = _valid_mapping()
    unknown["phases"][2]["scenario_weights"][0]["scenario_id"] = "missing"
    with pytest.raises(ValueError, match="unknown"):
        ExecutionCostCurriculum.from_mapping(unknown)

    not_randomized = _valid_mapping()
    not_randomized["phases"][1]["scenario_weights"][1]["weight"] = 0.0
    with pytest.raises(ValueError, match="at least two"):
        ExecutionCostCurriculum.from_mapping(not_randomized)


def test_easy_phase_rejects_zero_cost_scenarios() -> None:
    raw = _valid_mapping()
    raw["scenarios"][0]["slippage_bps_per_side"] = 0.0
    with pytest.raises(ValueError, match="easy_nonzero"):
        ExecutionCostCurriculum.from_mapping(raw)


def test_robust_fitness_uses_weekly_units_and_explicit_penalties() -> None:
    metrics = {
        "easy": {
            "mean_weekly_return": 0.02,
            "annualized_return": 0.30,
            "mean_weekly_rap": 0.01,
        },
        "nominal_a": {
            "mean_weekly_return": 0.01,
            "annualized_return": 0.15,
            "mean_weekly_rap": -0.01,
        },
        "nominal_b": {
            "mean_weekly_return": 0.03,
            "annualized_return": 0.45,
            "mean_weekly_rap": 0.00,
        },
        "stress": {
            "mean_weekly_return": 0.00,
            "annualized_return": 0.0,
            "mean_weekly_rap": -0.03,
        },
    }
    config = RobustFitnessConfig(
        lower_tail_fraction=0.5,
        downside_penalty_weight=2.0,
        dispersion_penalty_weight=0.25,
    )

    result = aggregate_robust_weekly_fitness(metrics, config=config)

    mean_return = 0.015
    mean_rap = -0.0075
    cvar = -0.02
    dispersion = math.sqrt(
        sum((value - mean_rap) ** 2 for value in (0.01, -0.01, 0.0, -0.03))
        / 4
    )
    downside_penalty = 2.0 * (mean_rap - cvar)
    dispersion_penalty = 0.25 * dispersion
    assert result["scenario_count"] == 4
    assert result["mean_weekly_return"] == pytest.approx(mean_return)
    assert result["annualized_return"] == pytest.approx(0.225)
    assert result["annual_return"] == pytest.approx(0.225)
    assert result["mean_weekly_rap"] == pytest.approx(mean_rap)
    assert result["annual_rap"] == pytest.approx(52.0 * mean_rap)
    assert result["worst_scenario_id"] == "stress"
    assert result["worst_scenario_weekly_rap"] == pytest.approx(-0.03)
    assert result["lower_tail_cvar_weekly_rap"] == pytest.approx(cvar)
    assert result["scenario_weekly_rap_dispersion"] == pytest.approx(dispersion)
    assert result["downside_weekly_rap_penalty"] == pytest.approx(downside_penalty)
    assert result["dispersion_weekly_rap_penalty"] == pytest.approx(
        dispersion_penalty
    )
    assert result["robust_weekly_rap_fitness"] == pytest.approx(
        mean_rap - downside_penalty - dispersion_penalty
    )


def test_fractional_lower_tail_mass_is_deterministic() -> None:
    metrics = {
        "a": {
            "mean_weekly_return": 0.0,
            "annualized_return": 0.0,
            "mean_weekly_rap": -0.03,
        },
        "b": {
            "mean_weekly_return": 0.0,
            "annualized_return": 0.0,
            "mean_weekly_rap": -0.01,
        },
        "c": {
            "mean_weekly_return": 0.0,
            "annualized_return": 0.0,
            "mean_weekly_rap": 0.02,
        },
    }
    result = aggregate_robust_weekly_fitness(
        metrics,
        config=RobustFitnessConfig(
            lower_tail_fraction=0.5,
            downside_penalty_weight=0.0,
            dispersion_penalty_weight=0.0,
        ),
    )
    assert result["lower_tail_cvar_weekly_rap"] == pytest.approx(
        (-0.03 - 0.5 * 0.01) / 1.5
    )
    assert result["robust_weekly_rap_fitness"] == pytest.approx(
        result["mean_weekly_rap"]
    )


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"a": {"mean_weekly_return": 0.1}},
        {
            "a": {
                "mean_weekly_return": math.nan,
                "annualized_return": 0.1,
                "mean_weekly_rap": 0.1,
            }
        },
        {
            "a": {
                "mean_weekly_return": 0.1,
                "annualized_return": 0.1,
                "mean_weekly_rap": math.inf,
            }
        },
    ],
)
def test_robust_fitness_fails_closed_on_incomplete_evidence(metrics: dict) -> None:
    with pytest.raises(ValueError):
        aggregate_robust_weekly_fitness(metrics)


def test_robust_fitness_config_rejects_invalid_penalties() -> None:
    with pytest.raises(ValueError):
        RobustFitnessConfig(lower_tail_fraction=0.0)
    with pytest.raises(ValueError):
        RobustFitnessConfig(lower_tail_fraction=1.01)
    with pytest.raises(ValueError):
        RobustFitnessConfig(downside_penalty_weight=-1.0)
    with pytest.raises(ValueError):
        RobustFitnessConfig(dispersion_penalty_weight=math.nan)
