"""WP21(b), reporting half: one seal over three arms, or no table at all.

The whole claim a closure table makes is "these rows, all of them, for every arm". Everything else in it -- the
totals, the ranks, the verdicts -- is downstream of that one claim, and it is the claim that is cheapest to break by
accident: an arm that stopped a step earlier, a bar edited between two runs, a decision moved to the next bar. None of
those shows up in a total.

So these tests are about the seal and nothing else. It covers BOTH arms' rows, because there is one seal and the three
reports carry it. It covers the BARS, so editing one breaks it. It covers the DECISION TIMESTAMPS, so moving a
decision to another bar breaks it even though every bar is untouched. And when the arms did not visit the same rows,
the module refuses instead of sealing three corpora under one name -- which would produce a table that looks
comparable and is not.

`policy_profitability` stays refused throughout: these reports carry action statistics and the environment's own
training-reward total, and the closure table says in its own words that neither is a quality claim.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("m5phet_evaluation", reason="the evaluation package is not stubbed here")

from m5phet_evaluation.protocol import SealBroken                                                 # noqa: E402

from agent_multi_m5phet import sweep_report                                                       # noqa: E402
from agent_multi_m5phet.refusal import PolicyRefusal                                              # noqa: E402

STEPS = 6
ARMS = ("candidate_decisions", "flat", "fitted_sac")


def write_bars(tmp_path, closes=None):
    path = Path(tmp_path) / "bars.csv"
    closes = closes or [100.0 + index for index in range(STEPS + 1)]
    lines = ["DATE_TIME,CLOSE,f0"]
    for index, close in enumerate(closes):
        lines.append(f"2026-01-{index + 1:02d} 00:00:00,{close:.4f},0.5")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_decisions(tmp_path, every=1, name="decisions.jsonl"):
    path = Path(tmp_path) / name
    lines = []
    for index in range(STEPS + 1):
        moment = f"2026-01-{index + 1:02d} 00:00:00"
        governing = f"2026-01-{(index // every) * every + 1:02d} 00:00:00"
        lines.append(json.dumps({"t": moment, "action": "long", "decided_at": governing,
                                 "carried_forward": governing != moment}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sweep_document(actions=None):
    def steps(value):
        return [{"t": f"2026-01-{index + 1:02d}T00:00:00", "bar_index": index, "action": value,
                 "environment_return_training_reward": 0.5 if value else 0.0, "position": 1 if value else 0,
                 "position_units": 0.0, "position_fraction": 0.0}
                for index in range(STEPS)]

    values = actions or {"candidate_decisions": 1.0, "flat": 0.0, "fitted_sac": 0.25}
    runs = {name: {"run": name, "steps": steps(values[name]), "steps_taken": STEPS, "stopped_because": "TERMINATED",
                   "environment_return_training_reward_total": sum(
                       step["environment_return_training_reward"] for step in steps(values[name]))}
            for name in ARMS}
    return {"schema": "m5phet.policy_sweep.v1", "runs": runs,
            "bars": {"sha256": "a" * 64, "last": "2026-01-06T00:00:00"},
            "decisions": {"sha256": "b" * 64},
            "comparison": {"same_rows": True, "rows_compared": STEPS, "verdict": "COMPARABLE"},
            "totals": {name: runs[name]["environment_return_training_reward_total"] for name in ARMS}}


def build(tmp_path, **kwargs):
    bars = kwargs.pop("bars", None) or write_bars(tmp_path)
    decisions = kwargs.pop("decisions", None) or write_decisions(tmp_path)
    document = kwargs.pop("document", None) or sweep_document()
    return sweep_report.build(document, str(bars), str(decisions), generated_at="2026-09-25T00:00:00Z",
                              sealed_at="2026-09-25T00:00:00Z", **kwargs)


# --- one seal, over both arms' rows ---------------------------------------------------------------------------------

def test_one_seal_covers_every_arm(tmp_path):
    built = build(tmp_path)
    stages = sorted(built["reports"])
    assert stages == ["fitted_sac", "flat", "laya_first_layer"]
    seals = {stage: payload["corpus_seal"] for stage, payload in built["reports"].items()}
    assert len(set(seals.values())) == 1 and set(seals.values()) == {built["seal"].seal}
    assert {payload["sealed_row_count"] for payload in built["reports"].values()} == {STEPS}
    assert {payload["protocol_digest"] for payload in built["reports"].values()} == {built["protocol"].digest}


def test_the_seal_covers_the_bars(tmp_path):
    """One bar edited, and the same rows no longer verify under the seal the reports carry."""
    built = build(tmp_path)
    edited = write_bars(tmp_path, closes=[100.0, 101.0, 999.0, *[103.0 + n for n in range(STEPS - 2)]])
    _population, labels = sweep_report.population_and_labels(sweep_document(), str(edited),
                                                             str(tmp_path / "decisions.jsonl"))
    with pytest.raises(SealBroken):
        built["seal"].verify(labels)


def test_the_seal_covers_the_decision_timestamps(tmp_path):
    """Every bar is untouched; only which decision point governs which bar has moved."""
    built = build(tmp_path)
    moved = write_decisions(tmp_path, every=3, name="moved.jsonl")
    _population, labels = sweep_report.population_and_labels(sweep_document(), str(tmp_path / "bars.csv"),
                                                             str(moved))
    with pytest.raises(SealBroken):
        built["seal"].verify(labels)


def test_arms_on_different_rows_are_refused_not_sealed_together(tmp_path):
    document = sweep_document()
    document["runs"]["fitted_sac"]["steps"] = document["runs"]["fitted_sac"]["steps"][:-1]
    document["comparison"] = {"same_rows": False, "why": "fitted_sac 5 step(s)"}
    with pytest.raises(PolicyRefusal, match="ARMS_NOT_ON_SAME_ROWS"):
        build(tmp_path, document=document)


def test_a_sweep_without_the_naive_arm_is_refused(tmp_path):
    document = sweep_document()
    del document["runs"]["flat"]
    with pytest.raises(PolicyRefusal, match="NAIVE_ARM_ABSENT"):
        build(tmp_path, document=document)


# --- what the reports carry, and what they refuse ----------------------------------------------------------------------

def test_the_reward_is_carried_with_the_flat_arm_on_the_same_rows(tmp_path):
    built = build(tmp_path)
    candidate = built["reports"]["laya_first_layer"]
    reward = next(metrics for metrics in candidate["metric_sets"] if metrics["name"] == "environment_return")
    assert reward["values"]["environment_return_training_reward_total"] == pytest.approx(STEPS * 0.5)
    assert reward["baseline"]["name"] == "flat"
    assert reward["baseline"]["same_rows_as_model"] is True
    assert reward["baseline"]["environment_return_training_reward_total"] == 0.0
    # the naive arm is its own reference and carries no baseline against itself
    flat = next(metrics for metrics in built["reports"]["flat"]["metric_sets"] if metrics["name"] == "environment_return")
    assert flat["baseline"] is None


def test_policy_profitability_is_refused_in_every_report(tmp_path):
    built = build(tmp_path)
    for payload in built["reports"].values():
        policy = next(metrics for metrics in payload["metric_sets"] if metrics["name"] == "policy")
        assert any("no order was placed" in note for note in policy["notes"])
        assert "profit" not in json.dumps(policy["values"]).lower()


def test_every_report_flags_itself_underpowered_on_this_sample(tmp_path):
    built = build(tmp_path)
    for payload in built["reports"].values():
        assert any(flag.startswith("UNDERPOWERED") for flag in payload["flags"])
    assert built["protocol"].minimum_rows == sweep_report.MINIMUM_ROWS > STEPS


def test_no_key_of_a_report_names_profit_or_an_order(tmp_path):
    built = build(tmp_path)
    forbidden = ("profit", "pnl", "p&l", "order", "broker", "fill")
    # the evaluation package's own key: "order" there is the order of the rows, not an instruction to a broker
    allowed = {"mean_turnover_in_population_order"}

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    for payload in built["reports"].values():
        for key in keys(payload):
            if key in allowed:
                continue
            assert not any(word in key.lower() for word in forbidden), key


def test_the_population_identifies_a_step_by_ordinal_and_instant(tmp_path):
    """An environment may stop twice on one bar; two rows sharing an identity make every count ambiguous."""
    built = build(tmp_path)
    assert built["population"][0] == "0000@2026-01-01T00:00:00"
    assert len(set(built["population"])) == len(built["population"]) == STEPS
