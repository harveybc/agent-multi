"""What this provider lets a person's words select, and what it refuses to let them select.

The workbench resolves ordinary phrasing against the values a provider DECLARES, then hands the result back as
`parameters`. Two things are worth proving here. First, that the declaration is honest: this bundle names one fitted
policy, so one slot with one value is declared and nothing open is invented to look richer. Second, that a policy named
by the person and not held by this bundle is refused by its own name -- the failure mode worth preventing is the helpful
one, where the only policy available is served under a name nobody asked for.
"""

import hashlib
import json

import pytest

from agent_multi_m5phet.provider import PolicyProvider, PolicyRefusal

OBS = 4
POLICY_ID = "dev_policy_v1"


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


def provider(bundle):
    directory, _manifest = bundle
    return PolicyProvider(environ={"M5PHET_POLICY_BUNDLE": str(directory), "M5PHET_POLICY_PYTHON": "/nonexistent"},
                          runner=lambda payload: {"action": [0.25], "seconds": 0.01})


def config():
    return {"as_of": "2026-09-24T12:00:00Z", "state": ""}


# --- what is declared --------------------------------------------------------------------------------------------

def test_it_declares_the_one_policy_the_bundle_holds(bundle):
    slots = provider(bundle).chat_slots()
    assert [slot["name"] for slot in slots] == ["policy_id"], "a bundle names one policy and nothing else is enumerable"
    assert slots[0]["allowed"] == [POLICY_ID]
    assert slots[0]["type"] == "string"
    assert "policy" in slots[0]["aliases"][POLICY_ID], "ordinary words must reach the declared value"


def test_without_a_bundle_nothing_is_declared():
    assert PolicyProvider(environ={}).chat_slots() == [], "a provider with no fitted state declares no vocabulary"


def test_nothing_open_is_declared(bundle):
    """An observation is supplied with the question, not chosen from a list; declaring it would be unvalidatable."""
    assert all(slot["allowed"] for slot in provider(bundle).chat_slots())
    assert "observation" not in {slot["name"] for slot in provider(bundle).chat_slots()}


# --- the declaration against the workbench's own rules -------------------------------------------------------------

def test_the_declaration_satisfies_the_interpreters_contract(bundle):
    interpret = pytest.importorskip("m5phet.interpret")
    assert interpret._check_slots(provider(bundle).chat_slots())


def test_ordinary_words_resolve_to_the_declared_policy(bundle):
    interpret = pytest.importorskip("m5phet.interpret")
    report = interpret.interpret("What action does this policy propose for this observation?",
                                 provider(bundle).chat_slots())
    assert report["status"] == interpret.STATUS_OK
    assert report["parameters"] == {"policy_id": POLICY_ID}
    assert report["sources"]["policy_id"] == "QUESTION_TEXT", "the words settled it; no model was consulted"


def test_an_interpreter_cannot_introduce_a_policy_this_bundle_does_not_have(bundle):
    """The model may choose among declared values. A proposal outside them is refused, never rounded to the neighbour."""
    interpret = pytest.importorskip("m5phet.interpret")

    class Inventing:
        available = True

        def identity(self):
            return {"command": "stub", "model": "stub", "available": True, "reading": "test double"}

        def propose(self, prompt, slots):
            return {"policy_id": "btc_1h_sac_v7"}

    report = interpret.interpret("what would you do with this vector?", provider(bundle).chat_slots(),
                                 interpreter=Inventing())
    assert report["status"] == interpret.STATUS_UNSUPPORTED
    assert "btc_1h_sac_v7" in report["why"]


# --- what the resolved parameters may and may not do ----------------------------------------------------------------

def test_the_resolved_policy_changes_nothing_when_it_is_the_declared_one(bundle):
    made = provider(bundle)
    without = made.chat_request("What action does this policy propose?", [0.0] * OBS, config())
    with_parameters = made.chat_request("What action does this policy propose?", [0.0] * OBS, config(),
                                        parameters={"policy_id": POLICY_ID})
    assert without == with_parameters
    assert with_parameters["task_id"] == POLICY_ID


def test_no_parameters_keeps_the_current_path(bundle):
    request = provider(bundle).chat_request("What action does this policy propose?", [0.0] * OBS, config())
    assert request["operation"] == "infer" and request["output_schema"] == {"targets": ["action"]}


def test_a_policy_that_is_not_this_one_is_refused_by_name(bundle):
    with pytest.raises(PolicyRefusal, match="UNKNOWN_POLICY") as refusal:
        provider(bundle).chat_request("what does the btc policy propose?", [0.0] * OBS, config(),
                                      parameters={"policy_id": "btc_1h_sac_v7"})
    assert "btc_1h_sac_v7" in str(refusal.value), "the name that was asked for must appear in the refusal"
    assert POLICY_ID in str(refusal.value), "and so must the one this bundle actually holds"


def test_a_named_policy_is_refused_before_the_observation_is_even_read(bundle):
    """Otherwise a wrong name would surface as a complaint about the data, and the person would fix the wrong thing."""
    with pytest.raises(PolicyRefusal, match="UNKNOWN_POLICY"):
        provider(bundle).chat_request("go long", "this is prose, not an observation", config(),
                                      parameters={"policy_id": "btc_1h_sac_v7"})


def test_an_undeclared_parameter_is_refused(bundle):
    with pytest.raises(PolicyRefusal, match="UNDECLARED_PARAMETER"):
        provider(bundle).chat_request("What action?", [0.0] * OBS, config(), parameters={"horizon": 60})


def test_parameters_must_be_a_mapping(bundle):
    with pytest.raises(PolicyRefusal, match="PARAMETERS_MUST_BE_A_MAPPING"):
        provider(bundle).chat_request("What action?", [0.0] * OBS, config(), parameters=["dev_policy_v1"])


def test_without_a_bundle_a_named_policy_still_refuses_the_bundle_first():
    with pytest.raises(PolicyRefusal, match="POLICY_BUNDLE_UNAVAILABLE"):
        PolicyProvider(environ={}).chat_request("anything", [0.0], config(), parameters={"policy_id": "whatever"})
