"""WP21(b), reporting half: one `m5phet-evaluation-report/1` per arm, over one seal, so the three can be a table.

`sweep.py` returns three totals. Three totals printed together are not a comparison -- what makes them one is that
they were produced on the same rows, that the rows were fixed before the totals were seen, and that anyone reading the
table can check both claims without rerunning anything. That is what the evaluation package's protocol and seal are
for, and this module is the adapter between a sweep and them.

**What the seal covers, and why it is built this way.** The population is the replay's own steps, each identified by
its ordinal and its bar instant (`0007@2025-11-20T12:00:00`): an ordinal because an environment may stop on the same
bar twice and two rows sharing an identity make every count ambiguous. Each row's sealed label is the digest of the
bar's own CSV content together with the instant of the decision point that governs it -- so the seal breaks if a bar
is edited AND if a decision is moved to another bar. The three arms carry the same seal because they ran on the same
rows; that identity is what `compare_stages` reads to decide `COMPARABLE`, and if the sweep says the arms did not
visit the same bars, this module refuses rather than sealing three different corpora under one name.

**What the reports claim: nothing about quality.** `policy_profitability` is refused by the evaluation package by
name, and the closure table therefore carries that refusal for every row of this area. What the reports carry is the
package's own `score_policy` action statistics and one metric set of the environment's own training-reward total,
beside the `flat` arm's total on the same rows as the declared naive reference. A training-reward total over 256
development bars of one instrument is a property of a simulation; it is not profit, not a backtest and not evidence
about any market, and the declared minimum row count is deliberately larger than this sample so that every report
carries `UNDERPOWERED` in its own flags.

    python -m agent_multi_m5phet.sweep_report --sweep sweep.json --bars bars.csv \
        --decisions decisions.jsonl --out-dir reports/
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from .refusal import PolicyRefusal

#: sweep run name -> the stage name the closure table shows. `candidate_decisions` is the sweep's generic word for
#: "whatever series was handed in"; in this run that series is the first layer's, and the table says so.
ARM_STAGES = {"candidate_decisions": "laya_first_layer", "flat": "flat", "fitted_sac": "fitted_sac"}

NAIVE_STAGE = "flat"
FAMILY = "policy"
REWARD_KEY = "environment_return_training_reward_total"

#: Declared BEFORE the run and deliberately above what a 256-bar development sample can supply, so that a report over
#: this sample flags itself. A minimum the sample happens to meet is not a minimum, it is a description of the sample.
MINIMUM_ROWS = 500

DECLARED_METRICS = ("action_dimension", "mean_absolute_action", "nonzero_share",
                    "mean_turnover_in_population_order", REWARD_KEY)

ANNOTATION_RULES = (
    "No row of this corpus is annotated and no ground truth exists for it: a bar has no correct action. The sealed "
    "per-row label is the digest binding the bar's own CSV content to the instant of the decision point that governs "
    "it, so the seal detects an edited bar and a moved decision point alike.",
    "The three arms are sealed once, together, because they replayed the same rows; a per-arm seal would make three "
    "corpora that could not be compared and would look exactly like one that could.")

AMBIGUITY = ("not applicable: no row carries a label, so no row can be ambiguous. This corpus fixes the rows a "
             "simulation ran over; it does not encode anyone's judgement about what should have been done on them")

REWARD_READING = (
    "environment_return_training_reward_total is the gym-fx environment's own reward under its configured reward "
    "plugin, summed over the replayed steps. It is a training signal inside a simulation with the campaign's own "
    "commission, slippage and execution rules; it is not profit, not a backtest result and not evidence about any "
    "market")

LITERATURE_NOT_CARRIED = (
    "no literature value is carried: the quantity reported here is one environment configuration's own training "
    "reward, which no published result is expressed in")


class SweepReportError(PolicyRefusal):
    """The sweep cannot be turned into reports as it stands. Named, never a partial table."""


#: The annotation keys `evaluation/compare_stages.py` reads off a report. Named here so a producer cannot invent a
#: key the table will never look at; the generator's own `annotate` is used when the M5PHET checkout is importable,
#: and the identical local writer otherwise, so this module does not require that checkout to be on the path.
ANNOTATIONS = ("stage", "target", "horizon", "scale", "literature")


def _annotate(report, **fields):
    try:
        from evaluation.compare_stages import annotate                    # the generator's own writer, when reachable
    except ImportError:
        pass
    else:
        return annotate(report, **fields)
    payload = json.loads(report.to_json())
    for key, value in fields.items():
        if key not in ANNOTATIONS:
            raise SweepReportError(f"UNKNOWN_ANNOTATION: {key!r} is not one of {ANNOTATIONS}")
        if value is not None:
            payload[key] = value
    return payload


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def bar_digests(bars_path, date_column="DATE_TIME"):
    """`instant -> digest of that bar's whole CSV row`. The row is digested as written, field for field."""
    from .sweep import instant

    with Path(bars_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise SweepReportError(f"BARS_UNREADABLE: {bars_path} carries no rows under a header")
    header = rows[0]
    if date_column not in header:
        raise SweepReportError(f"BARS_UNREADABLE: the bars carry no {date_column!r} column")
    where = header.index(date_column)
    return {instant(row[where]): _digest({"header": header, "row": row}) for row in rows[1:]}


def decision_stamps(decisions_path):
    """`bar instant -> the instant of the decision point that governs it`, from the series the sweep replayed."""
    from .sweep import instant

    stamps = {}
    for line in Path(decisions_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        stamps[instant(entry["t"])] = instant(entry.get("decided_at", entry["t"]))
    return stamps


def population_and_labels(sweep_document, bars_path, decisions_path):
    """The rows the three arms ran over, and the label each one is sealed under. Refuses three different corpora."""
    runs = sweep_document.get("runs") or {}
    if not runs:
        raise SweepReportError("SWEEP_WITHOUT_RUNS: this sweep carries no run, so there is nothing to report")
    comparison = sweep_document.get("comparison") or {}
    if comparison.get("same_rows") is not True:
        raise SweepReportError(
            "ARMS_NOT_ON_SAME_ROWS: the sweep reports that the arms did not visit the same bars "
            f"({comparison.get('why')}). Sealing them under one corpus would make a table that looks comparable and "
            "is not")
    paths = {name: [step["t"] for step in result["steps"]] for name, result in runs.items()}
    reference = paths[next(iter(paths))]
    for name, path in paths.items():
        if path != reference:
            raise SweepReportError(f"ARMS_NOT_ON_SAME_ROWS: arm {name!r} visited a different bar sequence")
    digests = bar_digests(bars_path)
    stamps = decision_stamps(decisions_path)
    population, labels = [], {}
    for ordinal, (moment, step) in enumerate(zip(reference, runs[next(iter(runs))]["steps"])):
        row = f"{ordinal:04d}@{moment}"
        if moment not in digests:
            raise SweepReportError(f"STEP_NOT_IN_BARS: the replay reported a step at {moment!r}, which is not a bar "
                                   f"of {Path(bars_path).name}")
        population.append(row)
        labels[row] = _digest({"bar_sha256": digests[moment], "bar_index": step.get("bar_index"),
                               "decided_at": stamps.get(moment)})
    return tuple(population), labels


def protocol_for(population, *, bars_path, bars_sha256, decisions_path, decisions_sha256, split_frozen_at,
                 split_frozen_by):
    from m5phet_evaluation.protocol import EvaluationProtocol

    return EvaluationProtocol(
        family=FAMILY,
        population=population,
        label_source=(f"{Path(bars_path).name} sha256 {bars_sha256}; decision timestamps from "
                      f"{Path(decisions_path).name} sha256 {decisions_sha256}"),
        label_producer=("the ETHUSD 4h model-ready dataset export for the bars; the decision timestamps are this "
                        "run's own decision points. Neither is a judgement about what action a bar deserved: this "
                        "corpus has no ground truth, which is why policy quality stays refused"),
        label_provenance="INDEPENDENT_SYSTEM",
        annotation_rules=ANNOTATION_RULES,
        ambiguity_adjudication=AMBIGUITY,
        split=(("replay", population),),
        split_frozen_at=split_frozen_at,
        split_frozen_by=split_frozen_by,
        metrics=DECLARED_METRICS,
        baseline=("flat: the all-flat decision series replayed in the same environment configuration on the same "
                  "rows"),
        minimum_rows=MINIMUM_ROWS,
    )


def reward_metric_set(population, steps, *, naive_total, is_naive):
    """The environment's own training-reward total, with the `flat` arm on the same rows as the declared reference."""
    from m5phet_evaluation.scoring import MetricSet

    total = float(sum(float(step["environment_return_training_reward"]) for step in steps))
    baseline = None if is_naive else {
        "name": NAIVE_STAGE, REWARD_KEY: float(naive_total), "rows": len(population),
        "same_rows_as_model": True,
        "note": "the all-flat series replayed in the same environment configuration over the same sealed rows"}
    return MetricSet(
        name="environment_return", family=FAMILY, population=population,
        values={REWARD_KEY: total,
                "environment_return_training_reward_mean_per_step": total / len(steps) if steps else None,
                "steps": len(steps)},
        counts={"declared_rows": len(population), "scored_rows": len(steps)},
        baseline=baseline,
        notes=(REWARD_READING,
               "This arm is one series of actions in one simulation over one development sample. Nothing here is "
               "held out, nothing here is repeated, and a difference between two arms is not a measured effect."))


def build(sweep_document, bars_path, decisions_path, *, generated_at=None, sealed_at=None, split_frozen_at=None,
          split_frozen_by="agent_multi_m5phet.sweep_report: the population is the replay's own steps, fixed by the "
                          "bars and the decision series before any total was read"):
    """One annotated report payload per arm, all under one protocol and one seal."""
    from m5phet_evaluation.freeze import seal_corpus
    from m5phet_evaluation.report import build_report
    from m5phet_evaluation.scoring import score_policy

    runs = sweep_document["runs"]
    population, labels = population_and_labels(sweep_document, bars_path, decisions_path)
    stamp = split_frozen_at or sweep_document.get("bars", {}).get("last") or "unstated"
    protocol = protocol_for(population, bars_path=bars_path,
                            bars_sha256=sweep_document.get("bars", {}).get("sha256"),
                            decisions_path=decisions_path,
                            decisions_sha256=sweep_document.get("decisions", {}).get("sha256"),
                            split_frozen_at=stamp, split_frozen_by=split_frozen_by)
    seal = seal_corpus(labels, protocol=protocol, sealed_at=sealed_at)

    if NAIVE_STAGE not in runs:
        raise SweepReportError("NAIVE_ARM_ABSENT: the `flat` arm is the declared naive reference and it is not in "
                               "this sweep; a skill column against a missing baseline is the comparison the "
                               "evaluation package exists to prevent")
    naive_total = float(runs[NAIVE_STAGE]["environment_return_training_reward_total"])

    payloads = {}
    for name, result in sorted(runs.items()):
        stage = ARM_STAGES.get(name, name)
        actions = {row: float(step["action"]) for row, step in zip(population, result["steps"])}
        metric_sets = [score_policy(protocol=protocol, seal=seal, corpus=labels, actions=actions),
                       reward_metric_set(population, result["steps"], naive_total=naive_total,
                                         is_naive=(name == NAIVE_STAGE))]
        report = build_report(protocol=protocol, seal=seal, metric_sets=metric_sets, generated_at=generated_at)
        payloads[stage] = _annotate(
            report, stage=stage, target="exposure chosen per bar, replayed in the fitted policy's own environment",
            horizon="one bar (4h)", scale=REWARD_KEY)
    return {"protocol": protocol, "seal": seal, "reports": payloads, "population": population}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m agent_multi_m5phet.sweep_report",
        description="Turn one sweep into one evaluation report per arm, all under one protocol and one seal.")
    parser.add_argument("--sweep", required=True, help="the sweep document (m5phet.policy_sweep.v1)")
    parser.add_argument("--bars", required=True, help="the bars the sweep ran over")
    parser.add_argument("--decisions", required=True, help="the decision series the sweep replayed")
    parser.add_argument("--out-dir", required=True, help="where report_<stage>.json files are written")
    parser.add_argument("--sealed-at", default=None, help="stamp the seal with this instant instead of now")
    parser.add_argument("--generated-at", default=None, help="stamp the reports with this instant instead of now")
    args = parser.parse_args(argv)

    document = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    try:
        built = build(document, args.bars, args.decisions, generated_at=args.generated_at, sealed_at=args.sealed_at)
    except PolicyRefusal as refusal:
        print(json.dumps({"execution_authorized": False, "refused": str(refusal)}, indent=2), file=sys.stderr)
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for stage, payload in built["reports"].items():
        path = out / f"report_{stage}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[stage] = str(path)
    (out / "corpus_seal.json").write_text(json.dumps(built["seal"].as_dict(), indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    (out / "protocol.json").write_text(json.dumps(built["protocol"].as_dict(), indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"reports": written, "corpus_seal": built["seal"].seal,
                      "protocol_digest": built["protocol"].digest,
                      "sealed_rows": built["seal"].row_count,
                      "execution_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
