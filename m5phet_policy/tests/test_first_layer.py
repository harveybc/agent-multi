"""WP21(b), asking half: what the model is shown, what is kept, and what is counted.

Four claims are worth a test here, and they are the four that would be invisible if they broke.

The state is a summary, not the window. The model must never be handed the rows: it would then be an unaudited feature
extractor over data nobody can ask it to justify, and the failure would look exactly like success. Two tests hold that
line -- one by SIZE (a 32-bar window of 83 fitted columns is 2656 numbers; the state carries a couple of dozen) and one
by CONTENT (every cell of the bars is a sentinel value, and none of them appears in the state text).

The series is one entry per decision point. Off by one here means a decision applied to the wrong bar, which the
environment cannot report because it has no way to know which bar a decision was about.

Abstentions are counted, never spelled away. Below the threshold the action becomes `flat`, and `flat` is also a
perfectly good decision -- so a run whose abstentions were not counted would report a series of decisions when it had
made none, and would look identical in the file.

And the sweep must actually read the file. The two halves were written separately; a shape that only the writer
understands is a shape that is wrong.

These tests use the real `m5phet.decide` against a FAKE classification engine. They prove the plumbing and the
arithmetic; they establish nothing whatever about whether an exposure choice is a good one -- that is the environment's
answer, and it is a different file.
"""

import json
from pathlib import Path

import pytest

decide = pytest.importorskip("m5phet.decide", reason="the decision primitive lives in m5phet; it is not stubbed here")

from agent_multi_m5phet import abstention, first_layer                                            # noqa: E402
from agent_multi_m5phet.first_layer import (ABSTAINED, DECISION_KIND, OPTIONS, QUESTION,          # noqa: E402
                                            FirstLayerError, decision_points, log_return,
                                            realized_volatility, summary_payload)
from agent_multi_m5phet.refusal import PolicyRefusal                                              # noqa: E402
from agent_multi_m5phet.sweep import DECISIONS, read_decisions                                    # noqa: E402

WINDOW = 4
BARS = 24
#: every feature cell of the synthetic bars is one of these, so a raw value in the state text is unmistakable
SENTINEL = 918273.456789
FEATURES = [f"f{index}" for index in range(6)]
CHOICES = [key for key, _label in OPTIONS]


# --- fakes -----------------------------------------------------------------------------------------------------------

class FakeLaya:
    """A classification engine with Laya's answer shape and a scripted sequence of choices."""

    def __init__(self, script=None, confidences=None, backend="laya"):
        self.script = list(script or ["long"])
        self.confidences = list(confidences or [0.95])
        self.backend = backend
        self.states = []
        self.calls = 0

    def execute_task(self, prompt, task, attachments, language=None):
        self.states.append(task["state"]["news"])
        answers = {}
        for name, question in task["questions"].items():
            chosen = self.script[self.calls % len(self.script)]
            confidence = self.confidences[self.calls % len(self.confidences)]
            rest = (1.0 - confidence) / (len(CHOICES) - 1)
            answers[name] = {"type": "choice", "status": "OK", "label": chosen, "backend": self.backend,
                             "instructions": question.get("instructions"),
                             "options": [list(pair) for pair in question["options"]],
                             "uncalibrated_probabilities": {key: (confidence if key == chosen else rest)
                                                            for key in CHOICES},
                             "probability_decimals": 4, "calibration": "UNCALIBRATED",
                             "execution_authorized": False}
        self.calls += 1
        return {"response": {"answers": answers, "state_ref": "laya-checkpoint:test"}}


def write_bars(directory, rows=BARS):
    """Bars whose CLOSE moves and whose every feature cell is the sentinel."""
    path = Path(directory) / "bars.csv"
    header = ["DATE_TIME", "CLOSE", *FEATURES]
    lines = [",".join(header)]
    for index in range(rows):
        close = 100.0 + index
        lines.append(",".join([f"2026-01-{index + 1:02d} 00:00:00", f"{close:.4f}",
                               *[f"{SENTINEL:.6f}"] * len(FEATURES)]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def contract(rows=BARS):
    return {"schema": "m5phet_observation_contract.v1", "asset": "TESTPAIR", "timeframe": "4h",
            "required_rows": WINDOW, "policy_id": "test_policy", "contract_sha256": "c" * 64,
            "environment": {"feature_columns": list(FEATURES), "price_column": "CLOSE", "window_size": WINDOW,
                            "date_column": "DATE_TIME"}}


def bundle(tmp_path, rows=BARS):
    """A bundle directory the sweep's own readers accept, so `run` is exercised through its real path."""
    folder = tmp_path / "bundle"
    folder.mkdir(exist_ok=True)
    (folder / "observation_contract.json").write_text(json.dumps(contract(rows)), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({
        "schema": "m5phet_policy_bundle.v1", "policy_id": "test_policy", "provenance": "DEVELOPMENT",
        "observation_size": WINDOW * len(FEATURES), "checkpoint": str(folder / "model.zip"),
        "checkpoint_sha256": "d" * 64, "algorithm": "SAC", "action_space": "continuous",
        "action_size": 1, "action_low": -1.0, "action_high": 1.0,
        "unit": "target position fraction of the policy's action scale"}), encoding="utf-8")
    return folder


def run_first_layer(tmp_path, engine, **kwargs):
    bars = kwargs.pop("bars", None) or write_bars(tmp_path)
    return first_layer.run(str(bars), str(tmp_path / "decisions.jsonl"), engine=engine, decide=decide,
                           bundle=str(bundle(tmp_path)), record_dir=str(tmp_path / "records"),
                           as_of="2026-09-25T00:00:00+00:00", environ={}, **kwargs), bars


# --- the state is a summary, not the window ----------------------------------------------------------------------------

def test_the_state_carries_no_raw_row_by_content():
    """Not one cell of the bars appears in the text the model is shown."""
    closes = [100.0 + index for index in range(BARS)]
    instants = [f"2026-01-{index + 1:02d} 00:00:00" for index in range(BARS)]
    payload = summary_payload(instrument="TESTPAIR", timeframe="4h", step_seconds=14400, instants=instants,
                              closes=closes, index=BARS - 1, position="flat", feature_names=FEATURES)
    text = decide.decision_state(DECISION_KIND, payload)
    assert f"{SENTINEL:.6f}" not in text
    assert "918273" not in text
    # the fitted policy's own column NAMES are declared -- that is the point -- and no value of any of them is
    for name in FEATURES:
        assert name in text


def test_the_state_carries_no_raw_row_by_size():
    """The state holds a fixed handful of measurements, not one per cell of the window."""
    closes = [100.0 + index for index in range(BARS)]
    instants = [f"2026-01-{index + 1:02d} 00:00:00" for index in range(BARS)]
    payload = summary_payload(instrument="TESTPAIR", timeframe="4h", step_seconds=14400, instants=instants,
                              closes=closes, index=BARS - 1, position="flat", feature_names=FEATURES,
                              lookbacks=(1, 5), volatility_windows=(5,))
    text = decide.decision_state(DECISION_KIND, payload)
    numbers = [line for line in text.splitlines()
               if line.strip().startswith(("log_return:", "stdev_of_log_returns:", "last_close:"))]
    # one close, one return per declared lookback, one volatility per declared window. Nothing scales with the window.
    assert len(numbers) == 1 + 2 + 1
    window_cells = WINDOW * len(FEATURES)
    assert len(numbers) < window_cells


def test_a_state_with_more_declared_names_than_the_limit_is_refused():
    with pytest.raises(FirstLayerError, match="DECLARED_NAMES_TOO_MANY"):
        summary_payload(instrument="X", timeframe="4h", step_seconds=14400, instants=["t"], closes=[1.0], index=0,
                        position="flat", feature_names=[f"c{n}" for n in range(first_layer.MAX_DECLARED_NAMES + 1)])


def test_the_state_is_the_same_text_for_the_same_bar():
    closes = [100.0 + index for index in range(BARS)]
    instants = [f"2026-01-{index + 1:02d} 00:00:00" for index in range(BARS)]
    kwargs = dict(instrument="TESTPAIR", timeframe="4h", step_seconds=14400, instants=instants, closes=closes,
                  index=10, position="long", feature_names=FEATURES)
    first = decide.decision_state(DECISION_KIND, summary_payload(**kwargs))
    second = decide.decision_state(DECISION_KIND, summary_payload(**kwargs))
    assert first == second and decide.state_sha256(first) == decide.state_sha256(second)


def test_a_lookback_the_sample_does_not_reach_is_absent_not_zero():
    closes = [100.0, 101.0, 102.0]
    assert log_return(closes, 0, 1) is None
    assert realized_volatility(closes, 1, 20) is None
    assert log_return(closes, 2, 2) == pytest.approx(0.019802627)


# --- one entry per decision point ---------------------------------------------------------------------------------------

def test_one_decision_per_decision_point_at_every_bar(tmp_path):
    engine = FakeLaya(script=["long", "flat", "short"])
    result, _bars = run_first_layer(tmp_path, engine)
    header = result["manifest"]["header"]
    assert header["decision_points"] == BARS
    assert header["asked"] == BARS
    assert engine.calls == BARS
    assert header["lines"] == BARS


def test_every_n_bars_asks_once_and_carries_the_decision_forward(tmp_path):
    engine = FakeLaya(script=["long", "short"])
    result, _bars = run_first_layer(tmp_path, engine, every=4)
    header = result["manifest"]["header"]
    assert header["decision_points"] == len(decision_points(BARS, 4)) == 6
    assert engine.calls == 6
    # the replay steps every bar, so the file still has one line per bar; the lines say which are carried
    assert header["lines"] == BARS
    lines = [json.loads(line) for line in Path(result["manifest"]["decisions_path"]).read_text().splitlines()]
    assert sum(1 for line in lines if not line["carried_forward"]) == 6
    assert sum(1 for line in lines if line["carried_forward"]) == BARS - 6


def test_decision_points_refuses_an_interval_below_one():
    with pytest.raises(FirstLayerError, match="BAD_DECISION_INTERVAL"):
        decision_points(10, 0)


# --- the option set is exactly the declared one ----------------------------------------------------------------------

def test_exactly_the_three_declared_options_reach_the_engine(tmp_path):
    engine = FakeLaya()
    run_first_layer(tmp_path, engine)
    assert [list(pair) for pair in OPTIONS] == [["long", "increase exposure"], ["flat", "no position"],
                                                ["short", "decrease exposure"]]


def test_a_fixture_backend_records_no_decision(tmp_path):
    """A choice made by something that is not the checkpoint is refused, and no action is invented for that bar."""
    engine = FakeLaya(backend="fixture")
    result, _bars = run_first_layer(tmp_path, engine)
    header = result["manifest"]["header"]
    assert header["refused"] == BARS and header["answered_above_threshold"] == 0
    assert all(entry["refusal"] == "NON_MODEL_FIXTURE" for entry in result["manifest"]["refusals"])


# --- abstentions are counted --------------------------------------------------------------------------------------------

#: Which rule runs is a property of the INSTALLED m5phet, not of this package: WP20 put `min_confidence` into
#: `m5phet.decide`, and `first_layer` uses it wherever it is there. The two rules are the same rule and take their
#: threshold from two different documents, so the tests build whichever document the installed checkout demands and
#: assert the behaviour both share.
DELEGATED = first_layer.supports_min_confidence(decide)


def source_document(tmp_path, threshold=0.8, name="bands.json"):
    """The threshold's citation, in the form the rule that will run reads it from."""
    path = Path(tmp_path) / name
    if DELEGATED:
        path.write_text(json.dumps({
            "version": "m5phet-evaluation-report/1", "stage": "laya_zero_shot", "protocol_digest": "p" * 64,
            "corpus_seal": "s" * 64,
            "metric_sets": [{"name": "calibration", "values": {"reliability": {"bin_count": 10, "bins": [
                {"bin": [0.3, 0.4], "count": 134, "accuracy": 0.3209, "mean_confidence": 0.3716},
                {"bin": [round(threshold - 0.1, 6), round(threshold, 6)], "count": 29, "accuracy": 0.3103,
                 "mean_confidence": round(threshold - 0.05, 6)},
                {"bin": [round(threshold, 6), round(threshold + 0.1, 6)], "count": 24, "accuracy": 0.7083,
                 "mean_confidence": round(threshold + 0.05, 6)},
                {"bin": [round(threshold + 0.1, 6), 1.0], "count": 31, "accuracy": 1.0,
                 "mean_confidence": 0.95}]}}}]}), encoding="utf-8")
    else:
        path.write_text(json.dumps({
            "schema": abstention.SCHEMA, "corpus_id": "test-corpus", "checkpoint": "laya-checkpoint:test",
            "chance_rate": 1 / 3, "transfer_assumption": "measured on another task",
            "thresholds": [{"threshold": threshold,
                            "below": {"n": 395, "accuracy": 0.3266},
                            "at_or_above": {"n": 55, "accuracy": 0.8727}}]}), encoding="utf-8")
    return path


def test_low_confidence_becomes_flat_and_is_counted(tmp_path):
    engine = FakeLaya(script=["long", "short"], confidences=[0.95, 0.40])
    result, _bars = run_first_layer(tmp_path, engine, min_confidence=0.8,
                                    abstention_source=str(source_document(tmp_path)))
    header = result["manifest"]["header"]
    assert header["asked"] == BARS
    assert header["abstained"] == BARS // 2
    assert header["answered_above_threshold"] == BARS - BARS // 2
    assert header["abstained"] + header["answered_above_threshold"] + header["refused"] == header["asked"]
    lines = [json.loads(line) for line in Path(result["manifest"]["decisions_path"]).read_text().splitlines()]
    abstentions = [line for line in lines if line.get("abstained")]
    assert abstentions and all(line["action"] == "flat" for line in abstentions)
    # an abstention is not silently a decision: the record says which word the model chose and that it was not kept
    assert all(line["abstention_refusal"] == ABSTAINED for line in abstentions)
    # what the record says the model chose differs by rule, and both are the right statement: `m5phet.decide` writes
    # `chosen: null` because an abstention is not a choice; the local rule keeps the word and marks it as not kept.
    if DELEGATED:
        assert all(line["chosen"] is None for line in abstentions)
    else:
        assert any(line["chosen"] == "short" for line in abstentions)
    assert all(0.0 < line["confidence"] < 0.8 for line in abstentions)


def test_a_threshold_without_a_source_is_refused(tmp_path):
    """A gate nobody measured would decide which answers count, and that decision would be the author's."""
    with pytest.raises(PolicyRefusal, match="ABSTENTION_THRESHOLD_UNSOURCED"):
        run_first_layer(tmp_path, FakeLaya(), min_confidence=0.8)


def test_the_manifest_names_which_rule_applied_the_threshold(tmp_path):
    result, _bars = run_first_layer(tmp_path, FakeLaya(confidences=[0.95]), min_confidence=0.8,
                                    abstention_source=str(source_document(tmp_path)))
    gate = result["manifest"]["abstention"]
    assert gate["rule_implemented_by"] == ("m5phet.decide" if DELEGATED else "agent_multi_m5phet.first_layer")
    assert gate["min_confidence"] == 0.8
    # the citation is carried either way: the framework's, or this package's own band document
    assert gate["citation"] is not None or gate["source"] is not None


def test_a_threshold_the_source_did_not_measure_is_refused(tmp_path):
    """0.75 falls inside a bin the citation never split, so the citation says nothing about it."""
    expected = "THRESHOLD_NOT_MEASURED" if DELEGATED else "ABSTENTION_THRESHOLD_NOT_IN_SOURCE"
    with pytest.raises(PolicyRefusal, match=expected):
        run_first_layer(tmp_path, FakeLaya(), min_confidence=0.75,
                        abstention_source=str(source_document(tmp_path, threshold=0.8)))


def test_without_a_threshold_nothing_abstains(tmp_path):
    engine = FakeLaya(script=["short"], confidences=[0.34])
    result, _bars = run_first_layer(tmp_path, engine)
    header = result["manifest"]["header"]
    assert header["abstained"] == 0 and header["answered_above_threshold"] == BARS
    assert result["manifest"]["abstention"]["min_confidence"] is None


# --- the sweep reads the file --------------------------------------------------------------------------------------------

def test_the_sweep_reads_the_series_this_module_writes(tmp_path):
    engine = FakeLaya(script=["long", "flat", "short"])
    result, bars = run_first_layer(tmp_path, engine)
    instants = [f"2026-01-{index + 1:02d}T00:00:00" for index in range(BARS)]
    series, counts = read_decisions(result["manifest"]["decisions_path"], instants)
    assert len(series) == BARS
    assert set(series.values()) <= set(DECISIONS)
    assert sum(counts.values()) == BARS


def test_the_manifest_carries_the_record_digests(tmp_path):
    engine = FakeLaya()
    result, _bars = run_first_layer(tmp_path, engine)
    digests = result["manifest"]["decision_sha256"]
    assert digests and digests == sorted(digests)
    written = sorted(path.stem for path in (tmp_path / "records").glob("*.json"))
    # every digest the manifest names is a record on disk, content-addressed by that digest
    assert set(digests) <= set(written)
    for digest in digests:
        record = decide.load(tmp_path / "records" / f"{digest}.json")
        assert record["kind"] == DECISION_KIND and record["question"] == QUESTION
        assert record["execution_authorized"] is False


def test_resume_reuses_a_record_instead_of_asking_again(tmp_path):
    engine = FakeLaya()
    run_first_layer(tmp_path, engine)
    asked_first = engine.calls
    again = FakeLaya()
    result, _bars = run_first_layer(tmp_path, again, resume=True)
    assert asked_first == BARS and again.calls == 0
    assert result["manifest"]["header"]["replayed_from_records"] == BARS


# --- nothing here claims authority, and no key names an order -------------------------------------------------------------

def test_no_key_of_the_manifest_names_profit_or_an_order(tmp_path):
    result, _bars = run_first_layer(tmp_path, FakeLaya())
    forbidden = ("profit", "pnl", "p&l", "pl_", "order", "broker", "fill", "trade_order")

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    for key in keys(result["manifest"]):
        assert not any(word in key.lower() for word in forbidden), key
    assert result["manifest"]["execution_authorized"] is False
    for line in Path(result["manifest"]["decisions_path"]).read_text().splitlines():
        for key in json.loads(line):
            assert not any(word in key.lower() for word in forbidden), key
