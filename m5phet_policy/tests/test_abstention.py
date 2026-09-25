"""Where a confidence threshold comes from: counts in buckets, and nothing that is not a count.

The threshold is the only free parameter of WP21(b)'s asking half, and a free parameter typed into a source file is
indistinguishable from one that was measured. So the document this module builds is what `first_layer` demands before
it will abstain at all, and these tests hold the three properties that make it worth demanding: the buckets are counts
over the rows (not a fit), a threshold the document did not measure is never answered with the nearest one it did, and
a corpus that is really two corpora is refused rather than pooled.
"""

import json
from pathlib import Path

import pytest

from agent_multi_m5phet.abstention import (SCHEMA, THRESHOLDS, AbstentionSourceError, bands, build, load,
                                           read_answers, split_at, threshold_entry)


def answers(tmp_path, rows):
    path = Path(tmp_path) / "answers.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def row(confidence, correct, *, truth="a", corpus="c1", checkpoint="k1"):
    predicted = truth if correct else "b"
    rest = (1.0 - confidence) / 2
    return {"predicted": predicted, "truth": truth, "corpus_id": corpus, "state_ref": checkpoint,
            "probabilities": {predicted: confidence,
                              **{name: rest for name in ("a", "b", "c") if name != predicted}}}


def test_a_bucket_is_a_count_of_its_rows(tmp_path):
    rows, _identity = read_answers(answers(tmp_path, [row(0.95, True), row(0.85, True), row(0.85, False),
                                                      row(0.40, False), row(0.40, True)]))
    table = {(entry["from"], entry["to"]): entry for entry in bands(rows)}
    assert table[(0.9, 1.0)] == {"from": 0.9, "to": 1.0, "n": 1, "accuracy": 1.0}
    assert table[(0.8, 0.9)] == {"from": 0.8, "to": 0.9, "n": 2, "accuracy": 0.5}
    assert table[(0.0, 0.5)] == {"from": 0.0, "to": 0.5, "n": 2, "accuracy": 0.5}
    # an empty bucket keeps its row with a null accuracy; dropping it would read as a bucket that was never possible
    assert table[(0.5, 0.6)] == {"from": 0.5, "to": 0.6, "n": 0, "accuracy": None}


def test_the_split_reports_both_halves_with_their_counts(tmp_path):
    rows, _identity = read_answers(answers(tmp_path, [row(0.95, True), row(0.9, True), row(0.3, False),
                                                      row(0.3, False), row(0.3, True)]))
    entry = split_at(rows, 0.8)
    assert entry == {"threshold": 0.8,
                     "below": {"n": 3, "accuracy": pytest.approx(1 / 3)},
                     "at_or_above": {"n": 2, "accuracy": 1.0}}


def test_a_threshold_the_document_did_not_measure_is_not_answered_with_the_nearest(tmp_path):
    document = build(answers(tmp_path, [row(0.95, True), row(0.3, False)]))
    assert threshold_entry(document, 0.8)["threshold"] == 0.8
    assert threshold_entry(document, 0.75) is None
    assert sorted(entry["threshold"] for entry in document["thresholds"]) == sorted(THRESHOLDS)


def test_answers_without_truth_are_refused_not_dropped(tmp_path):
    path = answers(tmp_path, [row(0.95, True)])
    path.write_text(path.read_text() + json.dumps({"predicted": "a", "probabilities": {"a": 0.9}}) + "\n")
    with pytest.raises(AbstentionSourceError, match="ANSWERS_WITHOUT_TRUTH"):
        read_answers(path)


def test_two_corpora_are_not_pooled(tmp_path):
    with pytest.raises(AbstentionSourceError, match="MIXED_CORPORA"):
        read_answers(answers(tmp_path, [row(0.9, True, corpus="c1"), row(0.9, True, corpus="c2")]))


def test_two_checkpoints_are_not_pooled(tmp_path):
    with pytest.raises(AbstentionSourceError, match="MIXED_CHECKPOINTS"):
        read_answers(answers(tmp_path, [row(0.9, True, checkpoint="k1"), row(0.9, True, checkpoint="k2")]))


def test_the_document_says_what_it_does_not_establish(tmp_path):
    document = build(answers(tmp_path, [row(0.95, True), row(0.3, False, truth="c")]))
    assert document["schema"] == SCHEMA and document["execution_authorized"] is False
    assert "not on any trading question" in document["transfer_assumption"]
    assert document["chance_rate"] == pytest.approx(1 / len(document["classes"]))
    assert document["rows_per_class"] == {"a": 1, "c": 1}


def test_a_document_of_another_schema_is_not_read(tmp_path):
    path = Path(tmp_path) / "other.json"
    path.write_text(json.dumps({"schema": "something.else", "thresholds": []}), encoding="utf-8")
    with pytest.raises(AbstentionSourceError, match="ABSTENTION_SOURCE_UNREADABLE"):
        load(path)
