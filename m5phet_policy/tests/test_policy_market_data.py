"""Market data in, the policy's own observation out -- and every way of getting a wrong one refused.

The provider has always accepted the observation as a raw list of floats. Nobody can type 2724 of those, so in practice
the policy area proved that the plumbing worked and served no one. These tests are about the second input: the market
data a person actually holds.

Three things are worth proving. That the vector is built by gym-fx, which owns the environment the policy was fitted in,
and not by a construction guessed in this package. That the raw-vector path is untouched -- it is the one that works with
nothing installed, and it must not become collateral damage. And that every route to a plausible-but-wrong observation
is refused BY NAME: a short window, a missing column, a non-finite cell, a file whose columns come out in a different
order. A wrong observation is not an error the policy can report; it answers it, and the answer looks exactly as
confident as a right one.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from agent_multi_m5phet.observation import CONTRACT_FILENAME, CONTRACT_SCHEMA
from agent_multi_m5phet.provider import PolicyProvider, PolicyRefusal, state_ref_for

WINDOW = 4
SCALING_WINDOW = 8
FEATURES = ["f0", "f1", "f2"]
#: 4 window rows x 3 features, plus a 4-row price window, its 4 returns, and 4 agent-state elements.
OBS = WINDOW * len(FEATURES) + WINDOW + WINDOW + 4


def _gym_fx_checkout():
    """The gym-fx checkout to import the builder from.

    Preferring the operator's own variable and falling back to a sibling checkout mirrors what gym-fx's own suite does
    for its data root; no absolute path belongs in code that is committed.
    """
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
            "feature_binary_columns": [], "feature_scaling": "rolling_zscore",
            "feature_scaling_window": SCALING_WINDOW, "feature_clip": 10.0,
            "include_price_window": True, "include_agent_state": True, "position_size": 0.01}


def _contract_file(environment=None, observation_size=None):
    return {"schema": CONTRACT_SCHEMA, "policy_id": "dev_policy_v1",
            "observation_size": observation_size if observation_size is not None else OBS,
            "required_rows": SCALING_WINDOW, "asset": "ETHUSD", "timeframe": "4h",
            "environment": environment or _environment(),
            "agent_state_default": {"position": 0, "equity": 10000.0, "initial_cash": 10000.0,
                                    "entry_price": 0.0, "holding_bars": 0,
                                    "bar_index": 0, "total_bars": 1},
            "provenance": "DEVELOPMENT fixture"}


@pytest.fixture
def bundle(tmp_path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"not a real checkpoint, only its bytes matter to the digest check")
    manifest = {"schema": "m5phet_policy_bundle.v1", "policy_id": "dev_policy_v1",
                "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "observation_size": OBS, "action_size": 1, "action_low": -1.0, "action_high": 1.0,
                "unit": "target position fraction", "action_space": "Box(-1, 1, (1,), float32)",
                "provenance": "DEVELOPMENT fixture"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / CONTRACT_FILENAME).write_text(json.dumps(_contract_file()))
    return tmp_path, manifest


def provider(bundle, environ=None, action=(0.25,)):
    directory, _manifest = bundle
    calls = []

    def runner(payload):
        calls.append(payload)
        return {"action": list(action), "seconds": 0.01}

    made = PolicyProvider(environ={"M5PHET_POLICY_BUNDLE": str(directory),
                                   "M5PHET_POLICY_PYTHON": "/nonexistent",
                                   **({"M5PHET_GYM_FX": CHECKOUT} if CHECKOUT else {}),
                                   **(environ or {})}, runner=runner)
    made.calls = calls
    return made


def rows(count=SCALING_WINDOW, columns=None, seed=3):
    """Market data as it reaches a provider: row objects of STRINGS, which is what a parsed CSV attachment becomes."""
    names = columns if columns is not None else ["DATE_TIME", "CLOSE"] + list(FEATURES)
    made = []
    value = 1000.0
    for index in range(count):
        value += math.sin(index * seed) * 3.0
        cells = {"DATE_TIME": f"2024-01-{index + 1:02d}T00:00:00", "CLOSE": f"{value:.6f}",
                 "f0": f"{math.cos(index) * 2:.6f}", "f1": f"{math.sin(index * 2):.6f}",
                 "f2": f"{(index % 5) - 2:.6f}"}
        made.append({name: cells.get(name, "0.0") for name in names})
    return made


def as_csv(table):
    header = list(table[0])
    lines = [",".join(header)] + [",".join(row[name] for name in header) for row in table]
    return "\n".join(lines) + "\n"


def config():
    return {"as_of": "2026-09-24T12:00:00Z", "state": ""}


# --- the raw-vector path is untouched ----------------------------------------------------------------------------

def test_the_raw_vector_path_is_unchanged(bundle):
    """It is the path that works with nothing installed; it must not become collateral damage."""
    request = provider(bundle).chat_request("What action does this policy propose?", [0.0] * OBS, config())
    assert request["inputs"]["observation"] == [0.0] * OBS
    assert "observation_build" not in request["inputs"], "a supplied vector was not built by anything"
    assert request["operation"] == "infer"


def test_the_raw_vector_path_works_with_no_gym_fx_at_all(bundle):
    made = provider(bundle, environ={"M5PHET_GYM_FX": "/nonexistent/not-a-checkout"})
    request = made.chat_request("What action does this policy propose?", [0.1] * OBS, config())
    assert request["inputs"]["observation"] == [0.1] * OBS


def test_prose_is_still_refused_as_prose(bundle):
    with pytest.raises(PolicyRefusal, match="OBSERVATION_REQUIRED"):
        provider(bundle).chat_request("go long with maximum size", "just some words", config())


# --- market data becomes the policy's observation -----------------------------------------------------------------

@needs_builder
def test_a_table_of_market_data_becomes_an_observation_of_the_declared_length(bundle):
    request = provider(bundle).chat_request("What action does this policy propose for these bars?",
                                            rows(), config())
    observation = request["inputs"]["observation"]
    assert len(observation) == OBS == bundle[1]["observation_size"]
    assert all(isinstance(value, float) for value in observation)


@needs_builder
def test_the_build_is_declared_so_a_receipt_can_carry_it(bundle):
    request = provider(bundle).chat_request("What does it propose?", rows(count=20), config())
    build = request["inputs"]["observation_build"]
    assert build["built_by"] == "gym_fx.observation_builder"
    assert build["rows_supplied"] == 20 and build["rows_consumed"] == SCALING_WINDOW
    assert build["contract"]["feature_order"] == FEATURES
    assert build["observation_length"] == OBS
    assert len(build["observation_sha256"]) == 64
    assert build["agent_state_source"] == "DECLARED_DEFAULT_IN_THE_BUNDLE"


@needs_builder
def test_csv_text_and_parsed_rows_build_the_same_observation(bundle):
    made = provider(bundle)
    table = rows()
    from_rows = made.chat_request("q", table, config())["inputs"]
    from_text = made.chat_request("q", as_csv(table), config())["inputs"]
    assert from_rows["observation"] == from_text["observation"]
    assert from_rows["observation_build"]["observation_sha256"] == from_text["observation_build"]["observation_sha256"]


@needs_builder
def test_the_built_observation_travels_with_the_action(bundle):
    """The numbers alone are unfalsifiable; the receipt has to say which rows they came from."""
    made = provider(bundle)
    request = made.chat_request("q", rows(), config())
    result = made.infer(request, {"state_ref": state_ref_for(made.manifest)})
    payload = result["outputs"]["action"]["payload"]
    assert payload["observation_source"] == "BUILT_FROM_MARKET_DATA"
    assert payload["observation_build"]["rows_consumed"] == SCALING_WINDOW
    assert payload["execution_authorized"] is False


def test_a_supplied_vector_says_so_in_the_receipt(bundle):
    made = provider(bundle)
    request = made.chat_request("q", [0.0] * OBS, config())
    payload = made.infer(request, {"state_ref": state_ref_for(made.manifest)})["outputs"]["action"]["payload"]
    assert payload["observation_source"] == "SUPPLIED_AS_A_VECTOR"
    assert payload["observation_build"] is None


# --- the state the market data does not carry ----------------------------------------------------------------------

@needs_builder
def test_the_caller_can_state_their_own_situation_and_it_changes_the_answer(bundle):
    made = provider(bundle)
    table = rows()
    flat = made.chat_request("q", table, config())["inputs"]
    held = made.chat_request("q", {"rows": table, "agent_state": {"position": 1, "equity": 12000.0,
                                                                 "initial_cash": 10000.0,
                                                                 "bar_index": 0, "total_bars": 1}},
                             config())["inputs"]
    assert held["observation"] != flat["observation"], "equity and position are observed; they must move the vector"
    assert held["observation_build"]["agent_state_source"] == "CALLER_OVERRODE_THE_DECLARED_DEFAULT"
    assert held["observation_build"]["agent_state_values"]["equity"] == 12000.0


@needs_builder
def test_an_agent_state_field_this_policy_does_not_observe_is_refused(bundle):
    with pytest.raises(PolicyRefusal, match="UNKNOWN_AGENT_STATE_FIELD"):
        provider(bundle).chat_request("q", {"rows": rows(), "agent_state": {"leverage": 10}}, config())


@needs_builder
def test_without_a_declared_default_the_agent_state_must_be_supplied(bundle):
    """A provider that invented the caller's position would be answering a different question from theirs."""
    directory, _manifest = bundle
    declared = _contract_file()
    declared.pop("agent_state_default")
    (directory / CONTRACT_FILENAME).write_text(json.dumps(declared))
    with pytest.raises(PolicyRefusal, match="AGENT_STATE_REQUIRED"):
        provider(bundle).chat_request("q", rows(), config())


# --- refusals, by name ----------------------------------------------------------------------------------------------

@needs_builder
def test_too_few_rows_is_refused_and_never_padded(bundle):
    with pytest.raises(PolicyRefusal, match="TOO_FEW_ROWS") as refusal:
        provider(bundle).chat_request("q", rows(count=SCALING_WINDOW - 1), config())
    assert str(SCALING_WINDOW) in str(refusal.value)


@needs_builder
def test_a_missing_feature_column_is_refused_by_name(bundle):
    table = [{key: value for key, value in row.items() if key != "f1"} for row in rows()]
    with pytest.raises(PolicyRefusal, match="MISSING_COLUMNS") as refusal:
        provider(bundle).chat_request("q", table, config())
    assert "f1" in str(refusal.value)


@needs_builder
def test_a_non_finite_cell_is_refused_rather_than_read_as_average(bundle):
    table = rows()
    table[-1]["f2"] = ""
    with pytest.raises(PolicyRefusal, match="NON_FINITE_VALUES"):
        provider(bundle).chat_request("q", table, config())


@needs_builder
def test_a_file_whose_columns_are_in_a_different_order_is_refused(bundle):
    table = rows(columns=["DATE_TIME", "CLOSE", "f1", "f0", "f2"])
    with pytest.raises(PolicyRefusal, match="COLUMN_ORDER_MISMATCH"):
        provider(bundle).chat_request("q", table, config())


@needs_builder
def test_a_table_that_is_not_this_policys_market_data_says_which_columns_are_missing(bundle):
    table = [{"open": "1", "high": "2", "low": "3", "close": "4"} for _ in range(SCALING_WINDOW)]
    with pytest.raises(PolicyRefusal, match="MISSING_COLUMNS") as refusal:
        provider(bundle).chat_request("q", table, config())
    assert "f0" in str(refusal.value)


# --- when the pieces are not there ------------------------------------------------------------------------------------

def test_without_an_observation_contract_the_market_data_path_is_closed_and_says_why(bundle):
    directory, _manifest = bundle
    (directory / CONTRACT_FILENAME).unlink()
    made = provider(bundle)
    readiness = made.capabilities()["market_data"]
    assert readiness["supported"] is False
    assert CONTRACT_FILENAME in readiness["why"]
    with pytest.raises(PolicyRefusal, match="OBSERVATION_CONTRACT_UNAVAILABLE|OBSERVATION_REQUIRED"):
        made.chat_request("q", rows(), config())
    assert made.chat_request("q", [0.0] * OBS, config())["inputs"]["observation"] == [0.0] * OBS


def test_when_gym_fx_is_not_reachable_it_says_so_plainly_and_does_not_guess(bundle):
    pytest.importorskip  # keep the import list honest; nothing is imported here on purpose
    made = provider(bundle, environ={"M5PHET_GYM_FX": "/nonexistent/not-a-checkout"})
    try:
        import gym_fx.observation_builder  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("gym-fx is installed in this interpreter, so unavailability cannot be exercised here")
    readiness = made.capabilities()["market_data"]
    assert readiness["supported"] is False and "GYM_FX_UNAVAILABLE" in readiness["why"]
    with pytest.raises(PolicyRefusal, match="GYM_FX_UNAVAILABLE"):
        made.chat_request("q", rows(), config())
    assert made.chat_request("q", [0.0] * OBS, config())["inputs"]["observation"] == [0.0] * OBS


@needs_builder
def test_a_contract_that_disagrees_with_the_bundle_is_refused_and_neither_is_preferred(bundle):
    directory, _manifest = bundle
    declared = _contract_file()
    declared["environment"]["window_size"] = WINDOW + 1
    (directory / CONTRACT_FILENAME).write_text(json.dumps(declared))
    made = provider(bundle)
    assert made.capabilities()["market_data"]["supported"] is False
    with pytest.raises(PolicyRefusal, match="OBSERVATION_CONTRACT_DISAGREES_WITH_BUNDLE"):
        made.chat_request("q", rows(), config())


@needs_builder
def test_what_it_declares_about_market_data_is_what_it_then_requires(bundle, monkeypatch):
    monkeypatch.delenv("M5PHET_POLICY_SAMPLE", raising=False)      # the operator's sample must not shape this test
    readiness = provider(bundle).capabilities()["market_data"]
    assert readiness["supported"] is True
    assert readiness["required_rows"] == SCALING_WINDOW
    assert readiness["observation_length"] == OBS
    assert readiness["feature_count"] == len(FEATURES)
    assert readiness["built_by"] == "gym_fx.observation_builder"
    # The example now carries bars rather than instructions, so what it declares is checked where the sample exists;
    # here, with no sample declared, the honest outcome is that the example is simply not offered.
    assert not [item for item in provider(bundle).chat_examples() if item["config"]["input"] == "csv"]


# --- the retained ETH checkpoint, when the operator has one -------------------------------------------------------------

def test_the_real_bundle_builds_the_length_its_manifest_declares():
    """The one measurement that matters: does the campaign's feature configuration rebuild THIS checkpoint's shape."""
    directory = os.environ.get("M5PHET_POLICY_BUNDLE")
    if not directory or not Path(directory, CONTRACT_FILENAME).is_file():
        pytest.skip("requires the operator's bundle with an observation contract in it")
    made = PolicyProvider()
    readiness = made.capabilities()["market_data"]
    if not readiness["supported"]:
        pytest.skip(f"the operator's bundle cannot build market data here: {readiness['why']}")
    assert readiness["observation_length"] == made.manifest["observation_size"]


# --- an example is something a person clicks and runs --------------------------------------------------------------------

def test_the_market_data_example_carries_bars_when_a_sample_is_declared(tmp_path, monkeypatch, bundle):
    """Shipping the instructions AS the example means the first thing anyone clicks is refused, for the very reason the
    example exists. It carries real bars, or it is absent."""
    import csv

    made = provider(bundle)
    readiness = made.market_data_readiness()
    if not readiness.get("supported"):
        pytest.skip("gym-fx is not importable in this environment")
    columns = ["DATE_TIME", readiness["price_column"], *FEATURES]
    sample = tmp_path / "bars.csv"
    with sample.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for index in range(readiness["required_rows"] + 40):
            writer.writerow([f"2026-01-01T{index:02d}:00:00"] + [str(1.0 + index * 0.001)] * (len(columns) - 1))
    monkeypatch.setenv("M5PHET_POLICY_SAMPLE", str(sample))
    example = [e for e in made.chat_examples() if "market data" in e["title"]]
    assert example, "the example must be offered once a sample is declared"
    rows = list(csv.reader(example[0]["data"].splitlines()))
    assert rows[0] == columns
    assert len(rows) == readiness["required_rows"] + 1, "the header plus exactly the rows this policy needs"


def test_without_a_declared_sample_the_example_is_absent_rather_than_unrunnable(monkeypatch, bundle):
    monkeypatch.delenv("M5PHET_POLICY_SAMPLE", raising=False)
    made = provider(bundle)
    assert not [e for e in made.chat_examples() if "market data" in e["title"]]
    assert [e for e in made.chat_examples() if "all-zero observation" in e["title"]], "the vector example remains"


def test_the_two_examples_say_they_are_two_inputs_and_two_actions(tmp_path, monkeypatch, bundle):
    """Retsu (2026-09-24, §8.6): the zero-vector example answered -0.0200 and the bars example 0.0591 under one
    policy_id, and the titles let a person read them as 'the action'. Each title now names the other as a different
    input, and each carries a reading of what its number is."""
    import csv

    made = provider(bundle)
    readiness = made.market_data_readiness()
    if not readiness.get("supported"):
        pytest.skip("gym-fx is not importable in this environment")
    columns = ["DATE_TIME", readiness["price_column"], *FEATURES]
    sample = tmp_path / "bars.csv"
    with sample.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for index in range(readiness["required_rows"] + 40):
            writer.writerow([f"2026-01-01T{index:02d}:00:00"] + [str(1.0 + index * 0.001)] * (len(columns) - 1))
    monkeypatch.setenv("M5PHET_POLICY_SAMPLE", str(sample))
    examples = made.chat_examples()
    assert len(examples) == 2, [e["title"] for e in examples]
    zero, bars = examples
    assert "all-zero" in zero["title"] and "not the bars example's" in zero["title"]
    assert "different input" in bars["title"] and "different action" in bars["title"]
    assert "zeros" in zero["reading"] and "not an order" in bars["reading"]
