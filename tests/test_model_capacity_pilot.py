"""Adversarial tests for the M0-M2 model-information pilot."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import model_capacity_pilot as pilot  # noqa: E402
import model_information_contract as mic  # noqa: E402


def _reseal(document, field):
    body = copy.deepcopy(document)
    body.pop(field, None)
    return mic.seal_document(body, field)


@pytest.fixture(scope="module")
def design():
    return pilot.make_design()


@pytest.fixture(scope="module")
def boolean_record(design):
    compact = copy.deepcopy(design)
    compact["training"] = dict(
        compact["training"],
        max_epochs=24,
        minimum_stop_epoch=4,
        patience=3,
        diagnostic_post_stop_epochs=3,
        checkpoint_epochs=[0, 1, 2, 4, 8, 16, 24],
    )
    spec = {
        "rule": "majority",
        "train_label_noise": 0.2,
        "hidden_width": 4,
        "seed": 11,
    }
    return compact, spec, pilot._run_unit(compact, "boolean_mlp", spec)


def test_bool_is_not_an_integer_or_measurement():
    with pytest.raises(mic.ModelInformationError, match="not bool"):
        mic.require_int("count", True)
    with pytest.raises(mic.ModelInformationError, match="finite number"):
        mic.require_number("metric", False)


def test_nonfinite_arrays_refuse_before_description():
    with pytest.raises(mic.ModelInformationError, match="non-finite"):
        mic.describe_model_arrays({"w": np.array([1.0, np.nan])})
    with pytest.raises(mic.ModelInformationError, match="non-finite"):
        mic.describe_data_arrays(
            {"x": np.array([1.0, np.inf])}, repeated_exposures=1
        )


def test_model_description_mutation_breaks_self_digest():
    description = mic.describe_model_arrays({"w": np.array([1.0, 2.0])})
    description["raw_serialization"]["zlib_bytes"] += 1
    with pytest.raises(mic.ModelInformationError, match="self-digest mismatch"):
        mic.verify_model_description(description)


def test_model_description_contains_pruning_and_low_rank_distortion_curves():
    description = mic.describe_model_arrays(
        {"matrix": np.arange(24, dtype=np.float64).reshape(6, 4)}
    )
    assert [
        point["requested_pruned_fraction"]
        for point in description["magnitude_pruning_curve"]
    ] == [0.0, 0.25, 0.5, 0.75, 0.9]
    assert description["magnitude_pruning_curve"][0]["parameter_mse"] == 0.0
    low_rank = description["low_rank_distortion_curves"][0]
    assert low_rank["name"] == "matrix"
    assert low_rank["points"][-1]["relative_frobenius_error"] < 1e-12


def test_repeated_weights_never_infer_spare_capacity():
    description = mic.describe_model_arrays({"w": np.ones(128)})
    assert description["exact_repeated_value_fraction"] > 0.99
    assert description["residual_capacity_inferred"] is False
    forged = _reseal(
        dict(description, residual_capacity_inferred=True), "description_sha256"
    )
    with pytest.raises(mic.ModelInformationError, match="residual capacity"):
        mic.verify_model_description(forged)


def test_repeated_epochs_are_accounted_but_not_multiplied():
    description = mic.describe_data_arrays(
        {"x": np.arange(16)}, repeated_exposures=100
    )
    assert description["repeated_training_exposures"] == 100
    assert description["independent_information_multiplier_from_repetition"] == 1
    forged = _reseal(
        dict(description, independent_information_multiplier_from_repetition=100),
        "description_sha256",
    )
    with pytest.raises(mic.ModelInformationError, match="not independent"):
        mic.verify_data_description(forged)


def test_stopping_is_a_function_of_calibration_losses_only():
    losses = [1.0, 0.8, 0.7, 0.71, 0.72, 0.73]
    first = pilot.choose_stop_from_calibration(
        losses, minimum_epoch=2, patience=3, minimum_delta=1e-6
    )
    # The production rule deliberately has no evaluation-data argument.
    second = pilot.choose_stop_from_calibration(
        list(losses), minimum_epoch=2, patience=3, minimum_delta=1e-6
    )
    assert first == second == (3, 6)


def test_capacity_endpoint_requires_fit_and_failure_regimes():
    records = []
    for ratio in (1.0, 1.5, 2.0):
        for _seed in (1, 2, 3):
            records.append(
                {
                    "kind": "threshold_neuron",
                    "spec": {
                        "inputs": 8,
                        "association_ratio": ratio,
                        "associations": int(8 * ratio),
                    },
                    "result": {
                        "perfect_fit": True,
                        "train_accuracy": 1.0,
                        "independent_random_label_evaluation_accuracy": 0.5,
                    },
                }
            )
    boundary = pilot._threshold_summary(records)["boundaries"][0]
    assert boundary["status"] == "SATURATION_NOT_IDENTIFIED"
    assert boundary["endpoint_is_never_called_capacity_without_both_regimes"] is True


def test_v3_bank_contains_required_controls_and_structured_noise(design):
    units = pilot.expected_units(design)
    assert len(units) == 195
    boolean_rules = {
        spec["rule"] for kind, spec in units if kind == "boolean_mlp"
    }
    assert "first_bit_identity_control" in boolean_rules
    assert "random_labels_null_control" in boolean_rules
    perturbations = {
        spec["perturbation"] for kind, spec in units if kind == "temporal_mlp"
    }
    assert perturbations == {"white", "colored", "impulsive"}
    assert design["threshold_neuron"]["input_distribution"] == (
        "random_gaussian_points_in_general_position_row_normalized"
    )


def test_code_identity_change_refuses_even_after_resealing(design):
    forged = copy.deepcopy(design)
    forged["code_identity"][pilot.CODE_FILES[0]] = "0" * 64
    forged = _reseal(forged, "design_sha256")
    with pytest.raises(pilot.PilotRefusal, match="code identity"):
        pilot.verify_design(forged)


def test_design_cannot_enable_evaluation_stopping(design):
    forged = copy.deepcopy(design)
    forged["training"]["evaluation_used_for_stopping"] = True
    forged = _reseal(forged, "design_sha256")
    with pytest.raises(pilot.PilotRefusal, match="evaluation split"):
        pilot.verify_design(forged)


def test_unit_identity_swap_refuses(design):
    spec = {
        "inputs": 4,
        "association_ratio": 1.0,
        "associations": 4,
        "seed": 11,
    }
    record = pilot._run_unit(design, "threshold_neuron", spec)
    with pytest.raises(pilot.PilotRefusal, match="identity"):
        pilot.verify_unit_record(
            record, design, "threshold_neuron", dict(spec, seed=12)
        )


def test_diagnostic_branch_cannot_become_selected(boolean_record):
    design, spec, record = boolean_record
    forged = copy.deepcopy(record)
    forged["result"]["stopping"]["diagnostic_branch_may_select"] = True
    forged = _reseal(forged, "record_sha256")
    with pytest.raises(pilot.PilotRefusal, match="cannot select"):
        pilot.verify_unit_record(forged, design, "boolean_mlp", spec)


def test_post_stop_checkpoint_cannot_replace_selection(boolean_record):
    design, spec, record = boolean_record
    forged = copy.deepcopy(record)
    forged["result"]["stopping"]["observed_stop_epoch"] = 4
    forged["result"]["stopping"]["selection_epoch"] = 5
    forged = _reseal(forged, "record_sha256")
    with pytest.raises(pilot.PilotRefusal, match="contaminated selection"):
        pilot.verify_unit_record(forged, design, "boolean_mlp", spec)


def test_checkpoint_mutation_refuses_even_if_wrapper_is_redigested(boolean_record):
    design, spec, record = boolean_record
    forged = copy.deepcopy(record)
    forged["result"]["selection_checkpoint"]["losses"]["evaluation"] = 999.0
    forged = _reseal(forged, "record_sha256")
    with pytest.raises(pilot.PilotRefusal, match="checkpoint self-digest"):
        pilot.verify_unit_record(forged, design, "boolean_mlp", spec)


def test_coherent_checkpoint_forgery_refuses_after_all_self_digests_are_repaired(
    boolean_record,
):
    design, spec, record = boolean_record
    forged = copy.deepcopy(record)
    checkpoint = forged["result"]["selection_checkpoint"]
    checkpoint["losses"]["evaluation"] = 999.0
    forged["result"]["selection_checkpoint"] = _reseal(
        checkpoint, "checkpoint_sha256"
    )
    forged = _reseal(forged, "record_sha256")
    with pytest.raises(pilot.PilotRefusal, match="fresh.*recomputation"):
        pilot.verify_unit_record(forged, design, "boolean_mlp", spec)


def test_undeclared_result_field_refuses_even_after_resealing(boolean_record):
    design, spec, record = boolean_record
    forged = copy.deepcopy(record)
    forged["result"]["producer_claim"] = "looks_good"
    forged = _reseal(forged, "record_sha256")
    with pytest.raises(pilot.PilotRefusal, match="fresh.*recomputation"):
        pilot.verify_unit_record(forged, design, "boolean_mlp", spec)


def test_missing_record_population_refuses(monkeypatch, design):
    first = {
        "inputs": 4,
        "association_ratio": 1.0,
        "associations": 4,
        "seed": 11,
    }
    second = dict(first, seed=12)
    record = pilot._run_unit(design, "threshold_neuron", first)
    monkeypatch.setattr(
        pilot,
        "expected_units",
        lambda _design: [("threshold_neuron", first), ("threshold_neuron", second)],
    )
    with pytest.raises(pilot.PilotRefusal, match="population"):
        pilot.derive_summary([record], design)


def test_nonfinite_wrapper_cannot_be_sealed(boolean_record):
    _, _, record = boolean_record
    forged = copy.deepcopy(record)
    forged["cpu_wall_seconds"] = float("nan")
    with pytest.raises(mic.ModelInformationError, match="canonical JSON"):
        _reseal(forged, "record_sha256")


def test_json_roundtrip_preserves_a_valid_unit(boolean_record):
    design, spec, record = boolean_record
    roundtrip = json.loads(json.dumps(record, allow_nan=False))
    pilot.verify_unit_record(roundtrip, design, "boolean_mlp", spec)
    assert set(roundtrip["diagnostic_cpu_seconds"]) == {
        "activation_geometry",
        "data_description",
        "gradient_fisher_hessian",
        "initial_model_description",
        "losses_and_scores",
        "model_description",
        "sample_specific_memorization",
    }
