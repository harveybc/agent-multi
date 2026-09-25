"""The policy provider under the unified envelope: area "rl", two question types, no invented numbers.

Two things are worth proving. That the envelope reaches the SAME path a chat takes -- `chat_request` then `infer` -- so a
policy named that this bundle does not hold is refused by name and an observation of the wrong length is refused rather
than padded, exactly as before. And that every number in an answer came from the checkpoint: the action from the actor,
the distribution from the actor's own Gaussian, the return from the twin critics. A recording runner that returns only an
action must produce an answer with those fields ABSENT and a value question REFUSED; it must never produce a distribution
built from one action or a return built from nothing.

The real-checkpoint tests run the operator's interpreter against the retained ETH policy and skip, saying so, when the
operator's environment is not configured.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from agent_multi_m5phet.provider import PolicyProvider, read_manifest, state_ref_for

questions = pytest.importorskip("m5phet.questions", reason="requires an m5phet with the question envelope")
from m5phet.questions import NOT_ESTIMABLE, STATE_REQUIRED, TASK_SCHEMA, catalog, run_task  # noqa: E402
from m5phet.runtime import Registry  # noqa: E402

OBS = 4
POLICY_ID = "dev_policy_v1"
CAMPAIGN_CSV = Path("/home/harveybc/Documents/GitHub/predictor/examples/data/project3/"
                    "ethusdt_4h_tech_stat_full_model_ready.csv")
CAMPAIGN_ROWS = 300


@pytest.fixture
def bundle(tmp_path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"not a real checkpoint, only its bytes matter to the digest check")
    manifest = {"schema": "m5phet_policy_bundle.v1", "policy_id": POLICY_ID,
                "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "observation_size": OBS, "action_size": 1, "action_low": -1.0, "action_high": 1.0,
                "unit": "target position fraction", "action_space": "Box(-1, 1, (1,), float32)",
                "provenance": "DEVELOPMENT fixture"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def provider(bundle, engine=None):
    """A provider over a recording runner. By default the engine returns an action and nothing else."""
    directory, _manifest = bundle
    calls = []

    def runner(payload):
        calls.append(payload)
        return {"action": [0.25], "seconds": 0.01, **(engine or {})}

    made = PolicyProvider(environ={"M5PHET_POLICY_BUNDLE": str(directory), "M5PHET_POLICY_PYTHON": "/nonexistent"},
                          runner=runner)
    made.calls = calls
    return made


def registry_with(made):
    r = Registry()
    r.register(made)
    return r


def task(state, **asked):
    return {"schema": TASK_SCHEMA, "area": "rl", "state": state, "as_of": "2026-09-24T12:00:00Z", "questions": asked}


NEXT = {"type": "next_action", "instructions": "What does the policy do next?"}
VALUE = {"type": "value_estimation", "instructions": "What return does it expect?"}

ENGINE_WITH_READINGS = {"actor_distribution": {"family": "gaussian_pre_squash", "squash": "tanh",
                                               "mean": [0.2554], "log_std": [-0.5], "std": [0.6065],
                                               "state_dependent_exploration": False},
                        "critic": {"q_values": [0.31, 0.27], "n_critics": 2, "action_scaled": [0.25], "gamma": 0.99}}


# --- what is declared ------------------------------------------------------------------------------------------------

def test_it_is_the_rl_area_and_declares_both_question_types(bundle):
    cat = catalog(registry_with(provider(bundle)))
    assert cat["rl"]["provider"] == "trading_policy"
    assert set(cat["rl"]["question_types"]) == {"next_action", "value_estimation"}


# --- next_action from a raw vector, through the chat path ---------------------------------------------------------------

def test_next_action_comes_from_the_fitted_policy_and_claims_no_authority(bundle):
    made = provider(bundle)
    out = run_task(task({"policy_id": POLICY_ID, "current_observation": [0.0] * OBS}, accion=NEXT), registry_with(made))
    answer = out["answers"]["accion"]
    assert answer["status"] == "OK" and answer["type"] == "next_action"
    assert answer["action"] == [0.25] and answer["policy_id"] == POLICY_ID
    assert answer["execution_authorized"] is False and out["execution_authorized"] is False
    assert answer["observation_source"] == "SUPPLIED_AS_A_VECTOR"
    assert out["state_ref"] == state_ref_for(made.manifest)
    assert made.calls[0]["observation"] == [0.0] * OBS and made.calls[0]["deterministic"] is True


def test_no_distribution_is_invented_from_one_action(bundle):
    """The engine returned an action. A confidence or a distribution would have to be made up, so neither is."""
    out = run_task(task({"current_observation": [0.0] * OBS}, accion=NEXT), registry_with(provider(bundle)))
    answer = out["answers"]["accion"]
    assert answer["confidence"] is None and "not emitted" in answer["confidence_absent_because"]
    assert answer["action_distribution"] is None and "none is derived" in answer["action_distribution_absent_because"]


def test_the_actors_own_distribution_is_reported_as_the_actors_and_not_as_a_probability(bundle):
    out = run_task(task({"current_observation": [0.0] * OBS}, accion=NEXT),
                   registry_with(provider(bundle, engine=ENGINE_WITH_READINGS)))
    answer = out["answers"]["accion"]
    distribution = answer["action_distribution"]
    assert distribution["mean"] == [0.2554] and distribution["log_std"] == [-0.5]
    assert "NOT a probability" in distribution["reading"]
    assert answer["confidence"] is None, "the spread of the actor is not a confidence, and is not relabelled as one"


# --- value_estimation: the critic, or a typed refusal --------------------------------------------------------------------

def test_a_return_is_estimated_from_the_twin_critics_or_not_at_all(bundle):
    out = run_task(task({"current_observation": [0.0] * OBS}, retorno=VALUE),
                   registry_with(provider(bundle, engine=ENGINE_WITH_READINGS)))
    answer = out["answers"]["retorno"]
    assert answer["status"] == "OK"
    assert answer["expected_return"] == 0.27, "the minimum of the twins, as SAC itself uses it"
    assert answer["uncertainty_bounds"] == [0.27, 0.31] and answer["critic_values"] == [0.31, 0.27]
    assert answer["evaluated_at_action"] == [0.25] and answer["discount_gamma"] == 0.99
    assert "not a realised profit" in answer["reading"]
    assert answer["execution_authorized"] is False and answer["policy_id"] == POLICY_ID


def test_without_a_critic_evaluation_the_value_question_is_refused_by_type(bundle):
    out = run_task(task({"current_observation": [0.0] * OBS}, retorno=VALUE), registry_with(provider(bundle)))
    answer = out["answers"]["retorno"]
    assert answer["status"] == "REFUSED" and answer["refusal"] == NOT_ESTIMABLE and answer["type"] == "value_estimation"
    assert "no critic evaluation" in answer["why"]
    assert "expected_return" not in answer


# --- both questions, one engine call, each answered on its own ------------------------------------------------------------

def test_both_questions_share_one_engine_call_and_are_answered_separately(bundle):
    made = provider(bundle)
    out = run_task(task({"current_observation": [0.0] * OBS}, accion=NEXT, retorno=VALUE), registry_with(made))
    assert list(out["answers"]) == ["accion", "retorno"]
    assert out["answers"]["accion"]["status"] == "OK" and out["answers"]["retorno"]["refusal"] == NOT_ESTIMABLE
    assert out["answered"] == 1 and out["refused"] == 1
    assert len(made.calls) == 1, "one observation, one subprocess; the questions share the evaluation"
    assert sorted(made.calls[0]["evaluate"]) == ["actor_distribution", "critic"]


# --- refusals, by name, before anything reaches the engine -------------------------------------------------------------------

def test_a_policy_this_bundle_does_not_hold_is_refused_by_its_name(bundle):
    made = provider(bundle)
    out = run_task(task({"policy_id": "someone_elses_policy", "current_observation": [0.0] * OBS},
                        accion=NEXT, retorno=VALUE), registry_with(made))
    for answer in out["answers"].values():
        assert answer["status"] == "REFUSED" and answer["refusal"] == STATE_REQUIRED
        assert "UNKNOWN_POLICY" in answer["why"] and "someone_elses_policy" in answer["why"]
    assert made.calls == []


@pytest.mark.parametrize("length", [OBS - 1, OBS + 3])
def test_an_observation_of_the_wrong_length_is_refused_and_never_padded(bundle, length):
    made = provider(bundle)
    out = run_task(task({"current_observation": [0.1] * length}, accion=NEXT), registry_with(made))
    answer = out["answers"]["accion"]
    assert answer["refusal"] == STATE_REQUIRED and "OBSERVATION_SIZE_MISMATCH" in answer["why"]
    assert "padding or trimming would be a different observation" in answer["why"]
    assert made.calls == [], "nothing reaches the engine once the observation has been refused"


def test_a_state_without_an_observation_is_refused(bundle):
    out = run_task(task({"environment_id": "eth-4h"}, accion=NEXT), registry_with(provider(bundle)))
    assert out["answers"]["accion"]["refusal"] == STATE_REQUIRED
    assert "OBSERVATION_REQUIRED" in out["answers"]["accion"]["why"]


# --- the retained ETH checkpoint, when the operator has one ------------------------------------------------------------------

def _real_provider():
    python = os.environ.get("M5PHET_POLICY_PYTHON")
    directory = os.environ.get("M5PHET_POLICY_BUNDLE")
    if not python or not directory or not read_manifest(directory):
        pytest.skip("requires the operator's stable-baselines3 interpreter (M5PHET_POLICY_PYTHON) and a declared "
                    "bundle (M5PHET_POLICY_BUNDLE); the real checkpoint is not exercised here")
    return PolicyProvider()


def _campaign_bars(made):
    readiness = made.market_data_readiness()
    if not readiness.get("supported"):
        pytest.skip(f"the operator's bundle cannot build market data here: {readiness['why']}")
    if not CAMPAIGN_CSV.is_file():
        pytest.skip(f"the campaign CSV is not on this machine: {CAMPAIGN_CSV}")
    with CAMPAIGN_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= CAMPAIGN_ROWS
    return rows[-CAMPAIGN_ROWS:], readiness


def test_the_real_policy_answers_next_action_from_a_raw_vector_with_its_own_distribution():
    made = _real_provider()
    size = made.manifest["observation_size"]
    out = run_task(task({"policy_id": made.manifest["policy_id"], "current_observation": [0.0] * size}, accion=NEXT),
                   registry_with(made))
    answer = out["answers"]["accion"]
    assert answer["status"] == "OK", answer
    assert len(answer["action"]) == made.manifest["action_size"] and answer["execution_authorized"] is False
    distribution = answer["action_distribution"]
    assert distribution["family"] == "gaussian_pre_squash" and distribution["squash"] == "tanh"
    assert len(distribution["mean"]) == len(distribution["log_std"]) == made.manifest["action_size"]
    assert answer["confidence"] is None


def test_the_real_policy_answers_next_action_from_campaign_bars():
    made = _real_provider()
    bars, readiness = _campaign_bars(made)
    out = run_task(task({"policy_id": made.manifest["policy_id"], "current_observation": bars}, accion=NEXT),
                   registry_with(made))
    answer = out["answers"]["accion"]
    assert answer["status"] == "OK", answer
    assert answer["observation_source"] == "BUILT_FROM_MARKET_DATA"
    build = answer["observation_build"]
    assert build["built_by"] == "gym_fx.observation_builder"
    assert build["rows_supplied"] == CAMPAIGN_ROWS and build["rows_consumed"] == readiness["required_rows"]
    assert build["observation_length"] == made.manifest["observation_size"]
    assert answer["execution_authorized"] is False


def test_the_real_policy_answers_value_estimation_from_its_critics_or_refuses_by_type():
    made = _real_provider()
    size = made.manifest["observation_size"]
    out = run_task(task({"current_observation": [0.0] * size}, accion=NEXT, retorno=VALUE), registry_with(made))
    answer = out["answers"]["retorno"]
    if answer["status"] == "REFUSED":
        assert answer["refusal"] == NOT_ESTIMABLE and answer["type"] == "value_estimation"
        return
    assert answer["n_critics"] == 2 and len(answer["critic_values"]) == 2
    assert answer["expected_return"] == min(answer["critic_values"])
    assert answer["uncertainty_bounds"] == [min(answer["critic_values"]), max(answer["critic_values"])]
    assert answer["evaluated_at_action"] == out["answers"]["accion"]["action"]
    assert "not a realised profit" in answer["reading"] and answer["execution_authorized"] is False
