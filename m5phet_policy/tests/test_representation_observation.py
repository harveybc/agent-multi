"""WP21(a): the observation built from a DECLARED representation, and the spec that is refused.

Until now the market-data path read the attached bars under exactly one contract -- the fitted
policy's, read from its bundle. That is correct and it is also silent: nothing in a receipt said
which representation had been honoured, and nothing could ask for another one, so the question
"would this policy answer a representation the design job proposed?" had no shape.

`m5phet.representation.v1` gives it one, and the answer is almost always no. A policy was fitted
under one representation; a vector built from another has the right length and a different
meaning, and the policy answers it exactly as confidently. So the tests here come in two halves.
The first is parity: the fitted policy's own representation, exported as a spec and handed back,
must reproduce gym-fx's construction element for element -- if that ever stops holding, the spec
path is describing a different observation while claiming to describe this one. The second is
refusal: a spec that drops a column, reorders one, moves the window or declares something the
env's observation has no block for is refused BY NAME, and the refusal names the lengths and the
columns, because those are what a person can act on.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from agent_multi_m5phet.observation import (
    CONTRACT_FILENAME,
    CONTRACT_SCHEMA,
    REPRESENTATION_SCHEMA,
    build_from_rows,
    load_representation_reader,
    observation_from_spec,
    policy_representation_spec,
)
from agent_multi_m5phet.provider import PolicyProvider, PolicyRefusal, state_ref_for

WINDOW = 4
SCALING_WINDOW = 8
FEATURES = ["f0", "f1", "f2"]
#: 4 window rows x 3 features, plus a 4-row price window, its 4 returns, and 4 agent-state elements.
OBS = WINDOW * len(FEATURES) + WINDOW + WINDOW + 4


def _gym_fx_checkout():
    configured = os.environ.get("M5PHET_GYM_FX")
    if configured and Path(configured, "gym_fx").is_dir():
        return configured
    sibling = Path(__file__).resolve().parents[3] / "gym-fx"
    return str(sibling) if (sibling / "gym_fx").is_dir() else None


CHECKOUT = _gym_fx_checkout()
needs_builder = pytest.mark.skipif(CHECKOUT is None,
                                   reason="requires a gym-fx checkout; the builder lives there and is not stubbed here")


def _environment():
    return {"window_size": WINDOW, "price_column": "CLOSE", "feature_columns": list(FEATURES),
            "feature_binary_columns": ["f2"], "feature_scaling": "rolling_zscore",
            "feature_scaling_window": SCALING_WINDOW, "feature_clip": 10.0,
            "include_price_window": True, "include_agent_state": True, "position_size": 0.01}


def _document(environment=None):
    """The bundle's observation contract: the only place a policy's representation exists."""
    return {"schema": CONTRACT_SCHEMA, "policy_id": "dev_policy_v1", "observation_size": OBS,
            "required_rows": SCALING_WINDOW, "asset": "ETHUSD", "timeframe": "4h",
            "checkpoint_sha256": "a" * 64,
            "environment": environment or _environment(),
            "agent_state_default": {"position": 0, "equity": 10000.0, "initial_cash": 10000.0,
                                    "entry_price": 0.0, "holding_bars": 0,
                                    "bar_index": 0, "total_bars": 1},
            "provenance": "DEVELOPMENT: a fixture"}


def _manifest():
    return {"schema": "m5phet_policy_bundle.v1", "policy_id": "dev_policy_v1",
            "observation_size": OBS, "checkpoint_sha256": "a" * 64}


def rows(count=SCALING_WINDOW, seed=3):
    """Market data as it reaches a provider: row objects of STRINGS, which is what a CSV becomes."""
    made = []
    value = 1000.0
    for index in range(count):
        value += math.sin(index * seed) * 3.0
        made.append({"DATE_TIME": f"2024-01-{index + 1:02d}T00:00:00", "CLOSE": f"{value:.6f}",
                     "f0": f"{math.cos(index) * 2:.6f}", "f1": f"{math.sin(index * 2):.6f}",
                     "f2": f"{float(index % 2):.6f}"})
    return made


# --- the export is a representation, and it is this policy's ------------------------------------

def test_the_exported_spec_is_the_schema_the_design_job_emits():
    spec = policy_representation_spec(_document())
    assert spec["schema"] == REPRESENTATION_SCHEMA
    assert spec["windows"] == [WINDOW] and spec["lags"] == []
    assert spec["target"] == {"column": "CLOSE", "transform": "level"}
    assert spec["exogenous"] == FEATURES, "the fitted order, which is the one the policy is indexed by"
    assert spec["sampling"]["step_seconds"] == 4 * 3600, "the bundle's 4h timeframe, read and not defaulted"
    assert spec["fitted_state_ref"] == "policy:" + "a" * 64


def test_what_the_export_could_not_decide_is_named_rather_than_claimed():
    """Three keys the schema requires decide nothing about an observation. They say so."""
    spec = policy_representation_spec(_document())
    assert set(spec["not_decided"]) == {"holdout", "calendar.clock", "sampling.timezone"}
    assert spec["calendar"]["columns"] == []


def test_an_unreadable_timeframe_is_refused_rather_than_defaulted():
    document = _document()
    document["timeframe"] = "every so often"
    with pytest.raises(PolicyRefusal, match="REPRESENTATION_SAMPLING_UNKNOWN"):
        policy_representation_spec(document)
    assert policy_representation_spec(document, step_seconds=900)["sampling"]["step_seconds"] == 900


def test_a_manifest_is_not_a_representation():
    """The manifest declares a length, and a length cannot say what the numbers mean."""
    with pytest.raises(PolicyRefusal, match="OBSERVATION_CONTRACT_UNAVAILABLE"):
        policy_representation_spec(_manifest())


def test_the_exported_spec_reads_under_the_module_that_declares_the_schema():
    reader = load_representation_reader(os.environ.get("M5PHET_REPRESENTATION"))
    if reader is None:
        pytest.skip("feature_eng_m5phet.representation is not importable here; it is not stubbed")
    assert reader.validate_spec(policy_representation_spec(_document())) is not None


# --- parity: the same observation, element for element ------------------------------------------

@needs_builder
def test_the_policys_own_representation_reproduces_the_builder_element_for_element():
    """The whole point of (a). Anything less and the spec path describes a different observation."""
    table = rows()
    fitted, _ = build_from_rows(table, _document(), _manifest(), CHECKOUT)
    spec = policy_representation_spec(_document())
    from_spec, declaration = observation_from_spec(table, spec, _document(), _manifest(), CHECKOUT)
    assert from_spec == fitted
    assert declaration["observation_length"] == OBS
    assert declaration["built_by"] == "gym_fx.observation_builder"


@needs_builder
def test_the_declaration_says_which_representation_was_honoured_and_what_came_from_the_bundle():
    spec = policy_representation_spec(_document())
    _observation, declaration = observation_from_spec(rows(), spec, _document(), _manifest(), CHECKOUT)
    representation = declaration["representation"]
    assert representation["schema"] == REPRESENTATION_SCHEMA
    assert representation["window"] == WINDOW and representation["feature_columns"] == len(FEATURES)
    assert "feature_scaling" in representation["taken_from_the_fitted_contract"], (
        "the schema has no key for it, so the receipt must say where it came from")


@needs_builder
def test_a_gym_fx_contract_may_be_handed_over_instead_of_the_bundles_document():
    builder = __import__("agent_multi_m5phet.observation", fromlist=["load_builder"]).load_builder(CHECKOUT)
    contract = builder.ObservationContract.from_config(_environment())
    supplied = {"rows": rows(), "agent_state": {"position": 0, "equity": 10000.0,
                                                "initial_cash": 10000.0, "bar_index": 0,
                                                "total_bars": 1}}
    observation, _declaration = observation_from_spec(supplied, policy_representation_spec(_document()),
                                                      contract, None, CHECKOUT)
    assert len(observation) == OBS


# --- refusal: a representation this policy was not fitted on ------------------------------------

@needs_builder
def test_a_dropped_column_is_refused_naming_the_lengths_and_the_column():
    spec = policy_representation_spec(_document())
    spec["exogenous"] = FEATURES[:-1]
    with pytest.raises(PolicyRefusal) as refusal:
        observation_from_spec(rows(), spec, _document(), _manifest(), CHECKOUT)
    message = str(refusal.value)
    assert message.startswith("OBSERVATION_CONTRACT_MISMATCH: ")
    assert "f2" in message, "the column that differs is what a person can act on"
    assert str(OBS) in message and str(OBS - WINDOW) in message, "expected against produced, both named"


@needs_builder
def test_the_same_columns_in_another_order_are_refused_although_the_length_agrees():
    """Same length, same set, a different observation: the case nothing downstream would notice."""
    spec = policy_representation_spec(_document())
    spec["exogenous"] = [FEATURES[1], FEATURES[0], FEATURES[2]]
    with pytest.raises(PolicyRefusal, match="OBSERVATION_CONTRACT_MISMATCH"):
        observation_from_spec(rows(), spec, _document(), _manifest(), CHECKOUT)


@needs_builder
def test_a_different_window_is_refused_naming_the_length_it_would_build():
    spec = policy_representation_spec(_document())
    spec["windows"] = [WINDOW * 2]
    with pytest.raises(PolicyRefusal) as refusal:
        observation_from_spec(rows(count=SCALING_WINDOW * 4), spec, _document(), _manifest(), CHECKOUT)
    assert "window_size" in str(refusal.value) or "length" in str(refusal.value)


@needs_builder
def test_two_windows_are_refused_rather_than_silently_reduced_to_one():
    spec = policy_representation_spec(_document())
    spec["windows"] = [WINDOW, WINDOW * 2]
    with pytest.raises(PolicyRefusal, match="REPRESENTATION_NOT_OBSERVABLE"):
        observation_from_spec(rows(), spec, _document(), _manifest(), CHECKOUT)


@needs_builder
def test_individual_lags_are_refused_because_the_observation_has_no_block_for_one():
    spec = policy_representation_spec(_document())
    spec["lags"] = [1, 2]
    with pytest.raises(PolicyRefusal, match="REPRESENTATION_NOT_OBSERVABLE"):
        observation_from_spec(rows(), spec, _document(), _manifest(), CHECKOUT)


@needs_builder
def test_a_differenced_or_transformed_target_is_refused():
    for change in ({"differencing": {"order": 1}}, {"target": {"column": "CLOSE", "transform": "log_return"}}):
        spec = policy_representation_spec(_document())
        spec.update(change)
        with pytest.raises(PolicyRefusal, match="REPRESENTATION_NOT_OBSERVABLE"):
            observation_from_spec(rows(), spec, _document(), _manifest(), CHECKOUT)


@needs_builder
def test_another_schema_is_refused_as_a_schema_and_not_as_a_missing_key():
    spec = policy_representation_spec(_document())
    spec["schema"] = "m5phet.representation.v2"
    with pytest.raises(PolicyRefusal, match="REPRESENTATION_SPEC_INVALID: WRONG_SCHEMA"):
        observation_from_spec(rows(), spec, _document(), _manifest(), CHECKOUT)


@needs_builder
def test_something_that_is_not_an_object_at_all_is_refused_by_name():
    with pytest.raises(PolicyRefusal, match="REPRESENTATION_SPEC_REQUIRED"):
        observation_from_spec(rows(), "the daily one", _document(), _manifest(), CHECKOUT)


# --- the provider's market-data path carries the spec, and its absence changes nothing -----------

@pytest.fixture
def bundle(tmp_path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"not a real checkpoint, only its bytes matter to the digest check")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {"schema": "m5phet_policy_bundle.v1", "policy_id": "dev_policy_v1",
                "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                "observation_size": OBS, "action_size": 1, "action_low": -1.0, "action_high": 1.0,
                "unit": "target position fraction", "action_space": "Box(-1, 1, (1,), float32)",
                "provenance": "DEVELOPMENT fixture"}
    document = _document()
    document["checkpoint_sha256"] = digest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / CONTRACT_FILENAME).write_text(json.dumps(document))
    return tmp_path, manifest, document


def _provider(bundle):
    directory, _manifest, _document = bundle
    return PolicyProvider(environ={"M5PHET_POLICY_BUNDLE": str(directory),
                                   "M5PHET_POLICY_PYTHON": "/nonexistent",
                                   **({"M5PHET_GYM_FX": CHECKOUT} if CHECKOUT else {})},
                          runner=lambda payload: {"action": [0.25], "seconds": 0.01})


def _config():
    return {"as_of": "2026-09-25T12:00:00Z", "state": ""}


@needs_builder
def test_the_attachment_may_carry_the_spec_beside_the_rows(bundle):
    _directory, _manifest_json, document = bundle
    made = _provider(bundle)
    with_spec = made.chat_request("q", {"rows": rows(), "spec": policy_representation_spec(document)}, _config())
    without = made.chat_request("q", rows(), _config())
    assert with_spec["inputs"]["observation"] == without["inputs"]["observation"]
    assert with_spec["inputs"]["observation_build"]["representation"]["window"] == WINDOW
    assert "representation" not in without["inputs"]["observation_build"], (
        "absent spec is the unchanged path, and a receipt must not claim a declaration nobody made")


@needs_builder
def test_a_foreign_spec_refuses_the_request_instead_of_answering_it(bundle):
    _directory, _manifest_json, document = bundle
    spec = policy_representation_spec(document)
    spec["exogenous"] = FEATURES[:-1]
    with pytest.raises(PolicyRefusal, match="OBSERVATION_CONTRACT_MISMATCH"):
        _provider(bundle).chat_request("q", {"rows": rows(), "spec": spec}, _config())


@needs_builder
def test_a_spec_beside_a_typed_vector_is_refused_because_there_is_nothing_to_read(bundle):
    with pytest.raises(PolicyRefusal, match="REPRESENTATION_SPEC_NEEDS_MARKET_DATA"):
        _provider(bundle).chat_request("q", [0.0] * OBS, _config(), spec=policy_representation_spec(_document()))


@needs_builder
def test_the_question_envelope_carries_the_spec_in_its_state(bundle):
    envelope = pytest.importorskip("m5phet.questions", reason="the envelope module is not importable here")
    _directory, _manifest_json, document = bundle
    made = _provider(bundle)
    state = {"current_observation": rows(), "spec": policy_representation_spec(document)}
    answers = made.answer_questions(state, {"a": {"type": "next_action", "instructions": "propose"}},
                                    None, "2026-09-25T12:00:00Z")
    assert answers["a"].get("status") in (None, "OK"), answers["a"]
    assert envelope is not None


@needs_builder
def test_a_foreign_spec_in_the_state_is_refused_as_a_refusal_and_not_as_an_action(bundle):
    pytest.importorskip("m5phet.questions", reason="the envelope module is not importable here")
    _directory, _manifest_json, document = bundle
    spec = policy_representation_spec(document)
    spec["windows"] = [WINDOW * 2]
    answers = _provider(bundle).answer_questions(
        {"current_observation": rows(), "spec": spec},
        {"a": {"type": "next_action", "instructions": "propose"}}, None, "2026-09-25T12:00:00Z")
    assert answers["a"]["status"] != "OK"
    assert "OBSERVATION_CONTRACT_MISMATCH" in json.dumps(answers["a"])
