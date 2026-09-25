"""Where a confidence threshold comes from, so that a threshold is a measurement and not a preference.

WP21(b) asks a checkpoint for one `choice` per bar and then throws away the answers it is not entitled to keep. The
number that decides which ones those are -- 0.8 -- is the whole difference between a series of decisions and a series
of coin flips, so it may not be typed. It is read off an already-sealed corpus: WP09 ran 450 independently labelled
rows of one classification task through the same checkpoint and wrote every answer with its uncalibrated
probabilities. Bucketing those answers by the probability the model put on the option it chose, and reporting accuracy
per bucket, says at which confidence the checkpoint stops being at chance -- on that corpus, for that task.

Two honesties this module is built around.

The first: it computes nothing that is not a count. Accuracy in a bucket is the fraction of rows in the bucket whose
chosen option was the labelled one, and `chance` is `1/len(classes)` for a corpus whose classes are balanced (the
count per class is reported so a reader can see whether they are). Nothing is smoothed, fitted or interpolated, and no
bucket is merged into its neighbour to make a boundary look sharper than the rows make it.

The second, and the reason this file exists rather than a constant: **the corpus is a classification corpus, not a
market**. It establishes that this checkpoint's probabilities carry information about its own accuracy *on that task*.
Carrying the threshold to a trading `choice` is an assumption, it is named as one in every document that uses it
(`transfer_assumption`), and it is not evidence that the checkpoint is any good at the trading question. A threshold
transferred across tasks is a hypothesis about the checkpoint's calibration, and the only thing it buys is that the
decisions kept are the ones the model was least uncertain about.

    python -m agent_multi_m5phet.abstention --answers <answers.jsonl> --out confidence_bands.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "m5phet.abstention_source.v1"

#: The bucket edges reported, and the thresholds a caller may ask for. Declared here rather than taken from the
#: command line so that two runs over the same corpus produce the same document and the table can be diffed.
BANDS = ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0000001))
THRESHOLDS = (0.6, 0.7, 0.8, 0.9)

TRANSFER_ASSUMPTION = (
    "measured on a classification corpus (which economy a calendar release names), not on any trading question. "
    "Using this threshold to decide which trading choices to keep assumes this checkpoint's confidence means the "
    "same thing on both tasks. That assumption is not tested here and nothing in this document supports it")

READING = (
    "accuracy per bucket of the probability the checkpoint put on the option it chose, over the sealed WP09 corpus. "
    "A bucket whose accuracy is the chance rate is a bucket in which the checkpoint's answer carries no information "
    "about the label; nothing here is calibrated or recalibrated")


class AbstentionSourceError(ValueError):
    """The answers cannot be read as a confidence-band source. Names the file and the field."""


def _sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_answers(path):
    """`[{chosen, confidence, correct}, ...]` plus the corpus identity the rows agree on.

    Every row must carry the truth it was scored against; a row without one is not evidence about accuracy and is
    refused rather than dropped, because dropping it would change the denominator without saying so.
    """
    rows, corpus, checkpoint = [], set(), set()
    try:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise AbstentionSourceError(f"ANSWERS_UNREADABLE: {path} could not be read ({exc})") from exc
    for number, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise AbstentionSourceError(f"ANSWERS_UNREADABLE: line {number} of {Path(path).name} is not JSON "
                                        f"({exc})") from exc
        probabilities = entry.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            raise AbstentionSourceError(f"ANSWERS_WITHOUT_PROBABILITIES: line {number} carries no uncalibrated "
                                        f"probabilities, and a confidence band is a statement about them")
        if "truth" not in entry or "predicted" not in entry:
            raise AbstentionSourceError(f"ANSWERS_WITHOUT_TRUTH: line {number} carries no `truth`/`predicted` pair; "
                                        f"accuracy is not computed from answers that were never scored")
        rows.append({"chosen": str(entry["predicted"]),
                     "confidence": float(probabilities[str(entry["predicted"])])
                     if str(entry["predicted"]) in probabilities else float(max(probabilities.values())),
                     "correct": str(entry["predicted"]) == str(entry["truth"]),
                     "truth": str(entry["truth"])})
        if entry.get("corpus_id"):
            corpus.add(str(entry["corpus_id"]))
        if entry.get("state_ref"):
            checkpoint.add(str(entry["state_ref"]))
    if not rows:
        raise AbstentionSourceError(f"ANSWERS_UNREADABLE: {Path(path).name} carries no answers")
    if len(corpus) > 1:
        raise AbstentionSourceError(f"MIXED_CORPORA: these answers name {len(corpus)} corpora; a band table is a "
                                    f"statement about one sealed corpus")
    if len(checkpoint) > 1:
        raise AbstentionSourceError(f"MIXED_CHECKPOINTS: these answers name {len(checkpoint)} checkpoints; a band "
                                    f"table is a statement about one of them")
    return rows, (next(iter(corpus), None), next(iter(checkpoint), None))


def bands(rows):
    """One entry per declared bucket: its edges, its count, its accuracy. Empty buckets keep their row, with `null`."""
    out = []
    for low, high in BANDS:
        inside = [row for row in rows if low <= row["confidence"] < high]
        out.append({"from": low, "to": min(high, 1.0), "n": len(inside),
                    "accuracy": (sum(row["correct"] for row in inside) / len(inside)) if inside else None})
    return out


def split_at(rows, threshold):
    """What the threshold does to this corpus: the two halves it makes, each with its count and its accuracy."""
    below = [row for row in rows if row["confidence"] < threshold]
    kept = [row for row in rows if row["confidence"] >= threshold]
    return {"threshold": float(threshold),
            "below": {"n": len(below),
                      "accuracy": (sum(row["correct"] for row in below) / len(below)) if below else None},
            "at_or_above": {"n": len(kept),
                            "accuracy": (sum(row["correct"] for row in kept) / len(kept)) if kept else None}}


def build(answers_path, *, quality_path=None):
    """The `m5phet.abstention_source.v1` document: counts, bands, thresholds, and what it does not establish."""
    rows, (corpus_id, checkpoint) = read_answers(answers_path)
    classes = sorted({row["truth"] for row in rows})
    per_class = {name: sum(1 for row in rows if row["truth"] == name) for name in classes}
    document = {
        "schema": SCHEMA,
        "execution_authorized": False,
        "reading": READING,
        "transfer_assumption": TRANSFER_ASSUMPTION,
        "answers": {"path": str(Path(answers_path).resolve()), "sha256": _sha256_file(answers_path), "n": len(rows)},
        "corpus_id": corpus_id,
        "checkpoint": checkpoint,
        "classes": classes,
        "rows_per_class": per_class,
        "chance_rate": (1.0 / len(classes)) if classes else None,
        "chance_rate_reading": ("1/len(classes); it is the rate of a uniform guess and it is the right reference only "
                                "where the classes are balanced, which `rows_per_class` shows"),
        "bands": bands(rows),
        "thresholds": [split_at(rows, value) for value in THRESHOLDS],
    }
    if quality_path:
        document["quality"] = {"path": str(Path(quality_path).resolve()), "sha256": _sha256_file(quality_path)}
    return document


def threshold_entry(document, threshold):
    """The declared split for exactly this threshold, or `None` -- never the nearest one.

    A caller that asked to keep decisions at 0.8 and was silently handed the evidence for 0.75 would be reading a
    number about a rule it is not applying.
    """
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise AbstentionSourceError(f"ABSTENTION_SOURCE_UNREADABLE: schema {(document or {}).get('schema')!r} is not "
                                    f"{SCHEMA!r}")
    for entry in document.get("thresholds") or ():
        if abs(float(entry["threshold"]) - float(threshold)) < 1e-12:
            return entry
    return None


def load(path):
    try:
        document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AbstentionSourceError(f"ABSTENTION_SOURCE_UNREADABLE: {path} ({exc})") from exc
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise AbstentionSourceError(f"ABSTENTION_SOURCE_UNREADABLE: {path} is not a {SCHEMA} document")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m agent_multi_m5phet.abstention",
        description="Read a scored answers file and report accuracy per confidence bucket, so a threshold is a "
                    "measurement on a named corpus instead of a preference.")
    parser.add_argument("--answers", required=True, help="JSONL of scored answers (predicted, truth, probabilities)")
    parser.add_argument("--quality", default=None, help="the quality document of the same run, bound by digest")
    parser.add_argument("--out", required=True, help="where the m5phet.abstention_source.v1 document is written")
    args = parser.parse_args(argv)
    try:
        document = build(args.answers, quality_path=args.quality)
    except AbstentionSourceError as error:
        print(json.dumps({"schema": SCHEMA, "execution_authorized": False, "refused": str(error)}, indent=2),
              file=sys.stderr)
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(Path(args.out).resolve()), "n": document["answers"]["n"],
                      "chance_rate": document["chance_rate"],
                      "thresholds": {str(entry["threshold"]): entry for entry in document["thresholds"]},
                      "execution_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
