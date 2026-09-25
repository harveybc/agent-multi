"""WP21(b): a decision series replayed as a candidate policy, and what the replay refuses.

A first layer that says `long`, `flat` or `short` at each bar is a hypothesis about what to do.
The harness under test hands that hypothesis to the environment the policies of this repository
are fitted in and reports what the environment returned -- beside the `flat` baseline on the same
rows and, when a bundle is configured, the fitted policy's own actions on the same rows.

The tests are about three things. That the numbers are the ENVIRONMENT'S: a short synthetic sweep
is compared, reward for reward, against an independent run of the same environment over the same
rows with the same action values (`tests/_env_reference.py`, which shares no code with the
harness). That the baseline is a baseline: an all-`flat` series never takes a position, so it is
the row of the table a candidate has to beat rather than a second candidate. And that every way of
sweeping the wrong thing -- bars without the fitted columns, fewer bars than the observation
needs, a decision about a bar nobody supplied -- is refused BY NAME before a simulation runs.

Nothing here reaches a broker, and the test at the end reads every key of the output to check that
no realised-P&L word appears in one: the reward is a training signal, and a name is how a number
is read by someone who did not run it.
"""

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_multi_m5phet.observation import CONTRACT_FILENAME, CONTRACT_SCHEMA
from agent_multi_m5phet.provider import PolicyRefusal
from agent_multi_m5phet.sweep import DECISIONS, SCHEMA, instant, sweep

WINDOW = 2
SCALING_WINDOW = 4
FEATURES = ["f0", "f1"]
OBS = WINDOW * len(FEATURES) + WINDOW + WINDOW + 4
BARS = 12


def _gym_fx_checkout():
    configured = os.environ.get("M5PHET_GYM_FX")
    if configured and Path(configured, "gym_fx").is_dir():
        return configured
    sibling = Path(__file__).resolve().parents[3] / "gym-fx"
    return str(sibling) if (sibling / "gym_fx").is_dir() else None


CHECKOUT = _gym_fx_checkout()
_ENGINE = all(importlib.util.find_spec(name) is not None
              for name in ("gymnasium", "backtrader", "pandas", "numpy"))
needs_engine = pytest.mark.skipif(
    CHECKOUT is None or not _ENGINE,
    reason="requires a gym-fx checkout and an interpreter with gymnasium and backtrader; the "
           "simulation is not stubbed here")


def _environment():
    return {"window_size": WINDOW, "price_column": "CLOSE", "feature_columns": list(FEATURES),
            "feature_binary_columns": [], "feature_scaling": "rolling_zscore",
            "feature_scaling_window": SCALING_WINDOW, "feature_clip": 10.0,
            "include_price_window": True, "include_agent_state": True, "position_size": 1.0}


def _env_config(bars_path):
    """A flat gym-fx configuration -- the shape `app/main.py` consumes, taken as it stands."""
    return {**_environment(),
            "input_data_file": str(bars_path), "date_column": "DATE_TIME", "headers": True,
            "env_mode": "training", "mode": "training", "action_space_mode": "continuous",
            "continuous_action_threshold": 0.1, "initial_cash": 10000.0, "min_equity": 0.0,
            "commission": 0.0, "slippage": 0.0, "leverage": 1.0, "timeframe": "4h",
            "simulation_engine": "backtrader", "solvency_mode": "normal_realistic",
            "data_feed_plugin": "default_data_feed", "broker_plugin": "default_broker",
            "strategy_plugin": "default_strategy", "metrics_plugin": "default_metrics",
            "preprocessor_plugin": "feature_window_preprocessor", "reward_plugin": "pnl_reward"}


@pytest.fixture
def bars(tmp_path):
    """A rising series, so a position the environment opens moves the reward it reports."""
    path = tmp_path / "bars.csv"
    lines = ["DATE_TIME,OPEN,HIGH,LOW,CLOSE,VOLUME,f0,f1"]
    for index in range(BARS):
        close = 1000.0 + 25.0 * index
        lines.append(f"2024-01-{index + 1:02d} 00:00:00,{close:.4f},{close * 1.002:.4f},"
                     f"{close * 0.998:.4f},{close:.4f},1000.0,"
                     f"{math.cos(index):.6f},{math.sin(index * 2):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def bundle(tmp_path, bars):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"a fixture checkpoint; only its bytes matter to the digest check")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "m5phet_policy_bundle.v1", "policy_id": "dev_policy_v1",
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "observation_size": OBS,
        "action_size": 1, "action_low": -1.0, "action_high": 1.0,
        "unit": "target position fraction", "action_space": "Box(-1, 1, (1,), float32)",
        "provenance": "DEVELOPMENT fixture"}))
    (tmp_path / CONTRACT_FILENAME).write_text(json.dumps({
        "schema": CONTRACT_SCHEMA, "policy_id": "dev_policy_v1", "observation_size": OBS,
        "required_rows": SCALING_WINDOW, "asset": "ETHUSD", "timeframe": "4h",
        "contract_sha256": "b" * 64, "environment": _environment(),
        "provenance": "DEVELOPMENT fixture"}))
    return tmp_path


def _decisions(tmp_path, chosen, name="decisions.jsonl"):
    """`chosen` maps a bar's ordinal to a word; every other bar is decided `flat`."""
    path = tmp_path / name
    lines = []
    for index in range(BARS):
        lines.append(json.dumps({"t": f"2024-01-{index + 1:02d} 00:00:00",
                                 "action": chosen.get(index, "flat")}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sweep(bundle, bars, decisions, **kw):
    return sweep(str(bars), str(decisions), bundle=str(bundle),
                 env_config=str(kw.pop("env_config")), gym_fx=CHECKOUT,
                 worker_python=sys.executable, include_fitted=False, **kw)


def _reference(env_config, actions):
    """The same environment, stepped with the same values, by code the harness does not share."""
    script = str(Path(__file__).resolve().parent / "_env_reference.py")
    done = subprocess.run([sys.executable, script],
                          input=json.dumps({"env_config": env_config, "actions": actions}),
                          text=True, capture_output=True, cwd=CHECKOUT, timeout=600,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    assert lines, f"the reference run produced nothing: {done.stderr[-2000:]}"
    return json.loads(lines[-1])


@pytest.fixture
def config_file(tmp_path, bars):
    path = tmp_path / "env.json"
    path.write_text(json.dumps(_env_config(bars)), encoding="utf-8")
    return path


# --- the numbers are the environment's -----------------------------------------------------------

@needs_engine
def test_a_long_then_flat_path_returns_the_environments_own_numbers(tmp_path, bundle, bars, config_file):
    """The whole point of (b): the harness reports what the env returned, not what it computed."""
    decisions = _decisions(tmp_path, {1: "long"})
    result = _sweep(bundle, bars, decisions, env_config=config_file, max_steps=3)
    steps = result["runs"]["candidate_decisions"]["steps"]

    assert [step["action"] for step in steps][0] == DECISIONS["long"], (
        "the first decided bar was long and the environment was handed the long value")
    assert DECISIONS["flat"] in [step["action"] for step in steps], "and a flat one after it"

    reference = _reference(_env_config(bars), [step["action"] for step in steps])
    assert [step["environment_return_training_reward"] for step in steps] == reference["rewards"]
    assert [step["position"] for step in steps] == reference["positions"]
    assert result["runs"]["candidate_decisions"]["environment_return_training_reward_total"] == \
        pytest.approx(reference["total"])
    assert [step["bar_index"] for step in steps] == reference["bars"], (
        "the decision is applied to the bar the environment says it is at, not to a count kept here")


@needs_engine
def test_the_decision_applied_at_each_step_is_the_one_written_for_that_bar(tmp_path, bundle, bars, config_file):
    decisions = _decisions(tmp_path, {1: "long", 2: "short"})
    written = {instant(json.loads(line)["t"]): json.loads(line)["action"]
               for line in decisions.read_text().splitlines()}
    result = _sweep(bundle, bars, decisions, env_config=config_file, max_steps=4)
    for step in result["runs"]["candidate_decisions"]["steps"]:
        assert step["action"] == DECISIONS[written[step["t"]]], step


@needs_engine
def test_the_flat_baseline_never_takes_a_position(tmp_path, bundle, bars, config_file):
    """A baseline that traded would be a second candidate, and the table would compare nothing."""
    result = _sweep(bundle, bars, _decisions(tmp_path, {1: "long"}), env_config=config_file, max_steps=6)
    flat = result["runs"]["flat"]
    assert {step["position"] for step in flat["steps"]} == {0}
    assert {step["position_fraction"] for step in flat["steps"]} == {0.0}
    assert {step["action"] for step in flat["steps"]} == {DECISIONS["flat"]}
    assert flat["environment_return_training_reward_total"] == 0.0


@needs_engine
def test_the_runs_are_compared_only_when_they_visited_the_same_bars(tmp_path, bundle, bars, config_file):
    result = _sweep(bundle, bars, _decisions(tmp_path, {1: "long"}), env_config=config_file, max_steps=5)
    assert result["comparison"]["verdict"] == "COMPARABLE"
    assert result["comparison"]["same_rows"] is True
    assert result["comparison"]["rows_compared"] == 5


@needs_engine
def test_what_the_sweep_declares_about_how_it_got_its_numbers(tmp_path, bundle, bars, config_file):
    result = _sweep(bundle, bars, _decisions(tmp_path, {1: "long"}), env_config=config_file, max_steps=3)
    assert result["schema"] == SCHEMA
    assert result["execution_authorized"] is False
    assert result["environment"]["reward_plugin"] == "pnl_reward"
    assert result["environment"]["env_mode"] == "training"
    assert result["decisions"]["mapping"] == dict(DECISIONS)
    assert len(result["bars"]["sha256"]) == 64 and len(result["decisions"]["sha256"]) == 64
    assert "NO_NEW_MEASUREMENT" in result["not_measured"]


@needs_engine
def test_no_key_in_the_output_is_a_realised_profit_word(tmp_path, bundle, bars, config_file):
    """A reward is a training signal. The name is how it gets read by whoever did not run it."""
    result = _sweep(bundle, bars, _decisions(tmp_path, {1: "long"}), env_config=config_file, max_steps=3)
    forbidden = ("pnl", "profit", "loss", "gain", "earnings", "return_pct", "roi", "realised",
                 "realized", "equity_curve")

    def keys(value):
        if isinstance(value, dict):
            for name, inner in value.items():
                yield str(name)
                yield from keys(inner)
        elif isinstance(value, list):
            for inner in value:
                yield from keys(inner)

    offending = [name for name in keys(result)
                 if any(word in name.lower() for word in forbidden)]
    assert offending == [], offending


# --- refusals, before anything is simulated --------------------------------------------------------

@needs_engine
def test_bars_without_the_fitted_columns_are_refused_by_name(tmp_path, bundle, bars, config_file):
    thin = tmp_path / "thin.csv"
    thin.write_text("DATE_TIME,OPEN,HIGH,LOW,CLOSE,VOLUME\n"
                    + "\n".join(f"2024-01-{i + 1:02d} 00:00:00,1,1,1,1,1" for i in range(BARS)) + "\n")
    with pytest.raises(PolicyRefusal, match="BARS_MISSING_FITTED_COLUMNS"):
        _sweep(bundle, thin, _decisions(tmp_path, {}), env_config=config_file)


@needs_engine
def test_fewer_bars_than_the_observation_needs_are_refused_by_name(tmp_path, bundle, bars, config_file):
    short = tmp_path / "short.csv"
    lines = bars.read_text().splitlines()
    short.write_text("\n".join(lines[:SCALING_WINDOW]) + "\n")
    with pytest.raises(PolicyRefusal, match="TOO_FEW_BARS"):
        _sweep(bundle, short, _decisions(tmp_path, {}), env_config=config_file)


@needs_engine
def test_a_decision_about_a_bar_nobody_supplied_is_refused_by_name(tmp_path, bundle, bars, config_file):
    path = tmp_path / "elsewhere.jsonl"
    path.write_text(json.dumps({"t": "2019-05-05 00:00:00", "action": "long"}) + "\n")
    with pytest.raises(PolicyRefusal, match="DECISION_TIMESTAMP_NOT_IN_BARS"):
        _sweep(bundle, bars, path, env_config=config_file)


@needs_engine
def test_a_word_this_harness_has_never_heard_of_is_refused_rather_than_mapped(tmp_path, bundle, bars, config_file):
    path = tmp_path / "odd.jsonl"
    path.write_text(json.dumps({"t": "2024-01-02 00:00:00", "action": "buy the dip"}) + "\n")
    with pytest.raises(PolicyRefusal, match="UNKNOWN_DECISION_ACTION"):
        _sweep(bundle, bars, path, env_config=config_file)


@needs_engine
def test_the_same_bar_decided_twice_is_refused(tmp_path, bundle, bars, config_file):
    path = tmp_path / "twice.jsonl"
    path.write_text("\n".join([json.dumps({"t": "2024-01-02 00:00:00", "action": "long"}),
                               json.dumps({"t": "2024-01-02T00:00:00", "action": "short"})]) + "\n")
    with pytest.raises(PolicyRefusal, match="DUPLICATE_DECISION_TIMESTAMP"):
        _sweep(bundle, bars, path, env_config=config_file)


@needs_engine
def test_a_bar_the_replay_reaches_with_no_decision_is_refused_and_not_defaulted(tmp_path, bundle, bars, config_file):
    """A default here would be an action nobody decided, scored as if somebody had."""
    path = tmp_path / "partial.jsonl"
    path.write_text(json.dumps({"t": "2024-01-02 00:00:00", "action": "long"}) + "\n")
    with pytest.raises(PolicyRefusal, match="DECISION_MISSING_FOR_BAR"):
        _sweep(bundle, bars, path, env_config=config_file, max_steps=BARS)


def test_an_environment_that_is_not_in_training_mode_is_refused(tmp_path, bundle, bars):
    config = _env_config(bars)
    config["env_mode"] = "inference"
    path = tmp_path / "inference.json"
    path.write_text(json.dumps(config))
    with pytest.raises(PolicyRefusal, match="ENVIRONMENT_NOT_IN_TRAINING_MODE"):
        _sweep(bundle, bars, _decisions(tmp_path, {}), env_config=path)


def test_a_bundle_without_an_observation_contract_cannot_say_which_columns_to_sweep(tmp_path, bars, config_file):
    naked = tmp_path / "naked"
    naked.mkdir()
    checkpoint = naked / "policy.zip"
    checkpoint.write_bytes(b"x")
    (naked / "manifest.json").write_text(json.dumps({
        "schema": "m5phet_policy_bundle.v1", "policy_id": "p", "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(b"x").hexdigest(), "observation_size": OBS,
        "action_size": 1, "action_low": -1.0, "action_high": 1.0, "unit": "u",
        "action_space": "Box(-1, 1, (1,), float32)", "provenance": "DEVELOPMENT fixture"}))
    with pytest.raises(PolicyRefusal, match="OBSERVATION_CONTRACT_UNAVAILABLE"):
        _sweep(naked, bars, _decisions(tmp_path, {}), env_config=config_file)
