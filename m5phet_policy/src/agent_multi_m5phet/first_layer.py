"""WP21(b), asking half: Laya chooses `long`, `flat` or `short` per bar, and the choices become a decision series.

`sweep.py` already replays a decision series in the environment the policies of this repository were fitted in. It
takes a file of `{"t", "action"}` lines and asks the environment what its own training reward did. What it never had
was the half that PRODUCES those lines from a model. This module is that half, and it is deliberately the smaller of
the two, because almost everything interesting about it is a restriction.

**The model is shown a description, never the data.** At every decision point the window is reduced to a summary --
the last close, the log return over each declared lookback, the realized volatility over each declared window, the
position the first layer is currently holding, the instrument, the bar step, and the fitted policy's own declared
feature names -- and that summary is rendered by `m5phet.decide.decision_state`, which sorts the keys, fixes the
decimals and refuses a list longer than its declared limit. A 32-bar window of 83 fitted columns is 2656 numbers; the
state carries a couple of dozen, and `tests/test_first_layer.py` asserts that by size and by content. Handing the rows
over would make the model an unaudited feature extractor over data it cannot be asked to justify.

**A choice under 0.8 confidence is not kept, and not hidden either.** WP09 ran 450 independently labelled rows through
this checkpoint and wrote every answer's uncalibrated probabilities. Split at 0.8, the rows it answered less
confidently are right 0.3266 of the time against a chance rate of 0.3333 -- that is, not at all -- and the rows at or
above are right 0.8727 of the time. So a decision below the threshold becomes `flat` and is COUNTED: `asked`,
`answered_above_threshold` and `abstained` are in the manifest's header and on stdout, because a series whose
abstentions were silently spelled `flat` would be reported as a series of decisions.

The threshold is never typed into this file. WP20 put the rule into `m5phet.decide` itself (`min_confidence` +
`abstention_source`, refusing `UNCITED_THRESHOLD` and `THRESHOLD_NOT_MEASURED`, writing the abstention as a record
with `chosen: null`), and where the installed `m5phet` carries it, that is what runs and `--abstention-source` is the
`m5phet-evaluation-report/1` whose reliability bins measured this checkpoint. Where it does not, the same rule runs
here against a `m5phet.abstention_source.v1` document (`agent_multi_m5phet.abstention`), and the manifest says which
of the two decided the run. Either citation was measured on a CLASSIFICATION corpus and not on any trading question;
carrying it here is an assumption about this checkpoint's calibration, named as one, and not evidence that it is any
good at the trading question.

**What the number at the end is not.** The sweep reports `environment_return_training_reward`: the reward the
environment's configured reward plugin emitted, summed over the replayed steps of a simulation. It is not profit, not
a backtest, and not evidence about any market; no key here names profit, P&L or an order, and `execution_authorized`
is false throughout. Nothing in this module reaches a broker: it writes a file.

    python -m agent_multi_m5phet.first_layer --bars bars.csv --out decisions.jsonl \
        [--every N] [--min-confidence 0.8 --abstention-source confidence_bands.json]
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import abstention
from . import observation as market
from . import sweep
from .refusal import PolicyRefusal

SCHEMA = "m5phet.policy_first_layer.v1"

#: The kind every record of this module is filed under, so WP23's calibration can find them by name.
DECISION_KIND = "trading_decision"
QUESTION = "exposure"

#: Exactly the option set WP21(b) declares. It travels in the question, the answer is checked back against it by
#: `m5phet.decide`, and `sweep.DECISIONS` maps each key onto the environment's own continuous action scale.
OPTIONS = [["long", "increase exposure"],
           ["flat", "no position"],
           ["short", "decrease exposure"]]

#: Kept short on purpose. The provider budgets head, options and state against one token limit, so every word here is
#: a word the summary cannot have; the summary is the evidence and the instruction is only the question.
INSTRUCTIONS = ("From this summary, choose the exposure for the next bar: long to increase exposure, flat to hold no "
                "position, short to decrease exposure.")

#: A decision kept below this much confidence is a coin flip; see `abstention.py` for where the number comes from.
ABSTAINED = "LOW_CONFIDENCE_ABSTAINED"
ABSTENTION_ACTION = "flat"

DEFAULT_LOOKBACKS = (1, 5, 20, 60)
DEFAULT_VOLATILITY_WINDOWS = (20, 60)
DEFAULT_RECORDS = "~/.local/state/m5phet/decisions"

#: The same limit `m5phet.decide` enforces on any list in a state. Declared here too so that a bundle with more
#: columns than this is refused by name here, before a state is built that the renderer would refuse anyway.
MAX_DECLARED_NAMES = 128

#: `position_held` is the first layer's OWN running position, implied by the decisions it has made so far. It is not
#: read from the environment, because the whole series is produced before any replay: there is no environment yet.
POSITION_READING = (
    "position_held is the first layer's own running position, implied by the decisions it has already made in this "
    "series, starting from the declared opening position. It is not the environment's position: the series is "
    "produced before the replay, so no environment exists while it is being produced")

REWARD_READING = sweep.REWARD_READING

STATE_READING = (
    "the state is a summary of the window, not the window: the last close, one log return per declared lookback, one "
    "realized volatility per declared window, the held position, the instrument, the bar step and the fitted "
    "policy's declared column names. No row of the bars is shown to the model")


class FirstLayerError(PolicyRefusal):
    """A first-layer run cannot be made as it stands. Named refusals, never a default decision."""


# --- what the model is shown ----------------------------------------------------------------------------------------

def _finite(value, what):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FirstLayerError(f"BARS_UNREADABLE: {what} is not a number ({value!r})") from exc
    if not math.isfinite(number):
        raise FirstLayerError(f"BARS_UNREADABLE: {what} is {value!r}, and a summary is not built over a "
                              f"non-finite price")
    return number


def log_return(closes, index, lookback):
    """The log return over `lookback` bars ending at `index`, or `None` when the history is not there.

    `None` rather than 0.0: a lookback the sample does not reach is a measurement that was not made, and a zero would
    be read as a flat market.
    """
    start = index - int(lookback)
    if start < 0:
        return None
    before, now = closes[start], closes[index]
    if before <= 0 or now <= 0:
        return None
    return math.log(now / before)


def realized_volatility(closes, index, window):
    """The sample standard deviation of the 1-bar log returns over `window` bars ending at `index`, or `None`."""
    window = int(window)
    if index - window < 0 or window < 2:
        return None
    steps = []
    for position in range(index - window + 1, index + 1):
        before, now = closes[position - 1], closes[position]
        if before <= 0 or now <= 0:
            return None
        steps.append(math.log(now / before))
    mean = sum(steps) / len(steps)
    variance = sum((step - mean) ** 2 for step in steps) / (len(steps) - 1)
    return math.sqrt(variance)


def summary_payload(*, instrument, timeframe, step_seconds, instants, closes, index, position, feature_names,
                    lookbacks=DEFAULT_LOOKBACKS, volatility_windows=DEFAULT_VOLATILITY_WINDOWS,
                    max_declared_columns=None):
    """The whole of what the model is shown, as a mapping `decide.decision_state` renders deterministically.

    Lists keep their declared order (the lookbacks, the windows, the column names) and everything else is a scalar, so
    the rendered text is the same text for the same bar however this mapping was built.

    `max_declared_columns` exists because the classification provider enforces a token budget and refuses
    `TOKEN_BUDGET_EXCEEDED` rather than truncating a state behind the caller's back. This policy declares 83 columns
    and the whole list does not fit beside the rest of the summary. So the list is cut HERE, where the cut can be
    declared: the state carries the true count of declared columns and the number of them shown, and a state that
    shows fewer than all of them says so in its own text. What is never done is to pass the full list and let the
    provider decide what to keep.
    """
    if not 0 <= index < len(closes):
        raise FirstLayerError(f"DECISION_POINT_OUT_OF_BARS: bar {index} is not one of the {len(closes)} supplied")
    if len(feature_names) > MAX_DECLARED_NAMES:
        raise FirstLayerError(
            f"DECLARED_NAMES_TOO_MANY: this bundle declares {len(feature_names)} columns and a decision state "
            f"carries at most {MAX_DECLARED_NAMES}; a longer list is a table, and a state describes")
    if position not in {key for key, _label in OPTIONS}:
        raise FirstLayerError(f"UNKNOWN_POSITION: {position!r} is not one of {[key for key, _ in OPTIONS]}")
    names = [str(name) for name in feature_names]
    shown = names if max_declared_columns is None else names[:int(max_declared_columns)]
    payload = {
        "instrument": str(instrument),
        "bar_timeframe": str(timeframe),
        "bar_step_seconds": int(step_seconds),
        "as_of_bar": str(instants[index]),
        "bars_seen": index + 1,
        "last_close": _finite(closes[index], f"the close of bar {index}"),
        "position_held": str(position),
        "log_returns": [{"lookback_bars": int(lookback), "log_return": log_return(closes, index, lookback)}
                        for lookback in lookbacks],
        "realized_volatility": [{"window_bars": int(window),
                                 "stdev_of_log_returns": realized_volatility(closes, index, window)}
                                for window in volatility_windows],
        "fitted_policy_declared_columns": shown,
        "fitted_policy_declared_column_count": len(names),
    }
    if len(shown) != len(names):
        payload["fitted_policy_declared_columns_shown"] = len(shown)
        payload["fitted_policy_declared_columns_cut"] = "first names in fitted order; cut to fit the token budget"
    return payload


# --- the bars, and the bundle that says which bars they must be -------------------------------------------------------

def read_bundle(bundle):
    """(manifest, observation contract). The same two documents the sweep replays under, read the same way."""
    return sweep.read_bundle(bundle)


def read_closes(path, document, date_column="DATE_TIME"):
    """(instants, closes) after `sweep.read_bars` has checked these are the columns the policy was fitted on.

    The price column is the environment's own (`price_column`, defaulting as the sweep defaults it), because the
    summary must describe the series the replay will be scored on and not some other column of the same file.
    """
    header, instants = sweep.read_bars(path, document, date_column)
    price = str((document.get("environment") or {}).get("price_column") or "CLOSE")
    if price not in header:
        raise FirstLayerError(f"BARS_MISSING_FITTED_COLUMNS: the bars carry no {price!r} column, and the summary "
                              f"describes the price series the replay is scored on")
    where = header.index(price)
    import csv

    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    closes = [_finite(row[where], f"{price} of bar {number}") for number, row in enumerate(rows[1:])]
    return instants, closes, price


def decision_points(total, every):
    """The bar indices a question is asked at: every `every`-th bar, and always the first one."""
    every = int(every)
    if every < 1:
        raise FirstLayerError(f"BAD_DECISION_INTERVAL: --every is a number of bars, at least 1; {every} is not")
    return list(range(0, int(total), every))


# --- asking -----------------------------------------------------------------------------------------------------------

def _decide_module():
    try:
        from m5phet import decide
    except ImportError as exc:                                                              # pragma: no cover
        raise FirstLayerError(
            "M5PHET_NOT_IMPORTABLE: the decision primitive lives in m5phet (`m5phet.decide`), and this interpreter "
            f"cannot import it ({exc}); no decision is invented in its absence") from exc
    return decide


def supports_min_confidence(decide=None):
    """Whether `m5phet.decide.ask` already takes `min_confidence` (WP20's branch), so the rule is not duplicated."""
    import inspect

    decide = decide or _decide_module()
    try:
        return "min_confidence" in inspect.signature(decide.ask).parameters
    except (TypeError, ValueError):                                                         # pragma: no cover
        return False


def confidence_of(decision):
    """The TOP uncalibrated probability over the declared options. Copied, never rescaled.

    The top probability, not the probability of the chosen key, because that is the quantity WP09's reliability bins
    were built from: each row was binned by the confidence of the option the checkpoint put first. `m5phet.decide`
    reads it the same way, so the local rule below and the framework's own rule are the same rule.
    """
    probabilities = decision.get("probabilities") or {}
    values = [float(value) for value in probabilities.values()
              if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return max(values) if values else None


def apply_threshold(decision, min_confidence):
    """`(action, abstained, confidence)` -- the same rule `m5phet.decide` applies, for the paths that do not go there.

    Used in two places: when the installed `m5phet.decide` has no `min_confidence` (older checkouts), and when a run
    replays a record already on disk, which was answered before any threshold existed.

    An abstention is `flat` because the harness has three words and no fourth; it is NOT a decision to hold no
    position, and the series says so through `abstained`. The decision record itself is kept exactly as the model made
    it: what the threshold changes is which action is replayed, never what was recorded as having been chosen. A tie
    at the top has no single argmax, so it abstains -- again as `decide` does.
    """
    chosen = decision.get("chosen")
    probabilities = decision.get("probabilities") or {}
    confidence = confidence_of(decision)
    if chosen is None:                                  # an abstention record: `decide` already applied the rule
        return ABSTENTION_ACTION, True, confidence
    if min_confidence is None:
        return chosen, False, confidence
    top = [key for key, value in probabilities.items() if float(value) == confidence]
    if confidence is None or confidence < float(min_confidence) or len(top) != 1:
        return ABSTENTION_ACTION, True, confidence
    return chosen, False, confidence


def ask_one(engine, decide, state_text, *, as_of, record_dir, replay=None, gate=None):
    """One `choice` for one decision point: a decision record, an abstention, or a typed refusal.

    `gate` is `{"min_confidence", "abstention_source"}` when the installed `m5phet.decide` applies the rule itself;
    then an abstention comes back as a REFUSED entry named `LOW_CONFIDENCE_ABSTAINED` that also carries its recorded
    decision, which is why it is passed up rather than folded into the other refusals.
    """
    if replay is not None:
        found = replay.get((DECISION_KIND, QUESTION, decide.state_sha256(state_text)))
        if found is not None:
            record, path = found
            return {"status": "OK", "decision": record, "record_path": path, "replayed": True}
    answered = decide.ask(engine, state_text, {QUESTION: {"options": OPTIONS, "instructions": INSTRUCTIONS}},
                          kind=DECISION_KIND, as_of=as_of, record_dir=record_dir, **(gate or {}))
    entry = answered[QUESTION]
    if entry.get("status") != "OK":
        out = {"status": "REFUSED", "refusal": entry.get("refusal"), "why": entry.get("why"),
               "state_sha256": decide.state_sha256(state_text), "replayed": False}
        if entry.get("decision") is not None:
            out["decision"], out["record_path"] = entry["decision"], entry.get("record_path")
        return out
    entry["replayed"] = False
    return entry


def load_records(record_dir, decide=None):
    """Index the records already on disk by `(kind, question, state_sha256)`, so a resumed run asks nothing twice."""
    decide = decide or _decide_module()
    folder = Path(os.path.expanduser(str(record_dir)))
    index = {}
    if not folder.is_dir():
        return index
    for path in sorted(folder.glob("*.json")):
        try:
            record = decide.load(path)
        except Exception:                                                                   # noqa: BLE001
            continue                                                # a file that is not a valid record is not one
        if record.get("kind") == DECISION_KIND:
            index[(record["kind"], record["question"], record["state_sha256"])] = (record, str(path))
    return index


def run(bars, out, *, engine=None, decide=None, bundle=None, every=1, min_confidence=None, abstention_source=None,
        lookbacks=DEFAULT_LOOKBACKS, volatility_windows=DEFAULT_VOLATILITY_WINDOWS, record_dir=DEFAULT_RECORDS,
        opening_position="flat", resume=False, environ=None, as_of=None, progress=None,
        max_declared_columns=None):
    """Ask one `choice` per decision point and write the decision series `sweep.py` reads, plus its manifest.

    The series carries one line per BAR, because the replay steps every bar and a bar with no action would be a bar
    the harness had to default. Between decision points the last decision is carried forward, each line saying which
    decision point it came from and whether it was carried -- so `asked` and the number of lines are two different
    numbers and neither is mistaken for the other.
    """
    decide = decide or _decide_module()
    env = os.environ if environ is None else environ
    bundle = bundle or env.get("M5PHET_POLICY_BUNDLE")
    manifest, document = read_bundle(bundle)
    environment = document["environment"]
    instants, closes, price_column = read_closes(bars, document, environment.get("date_column", "DATE_TIME"))

    step_seconds = market._seconds_of(document.get("timeframe"))
    if step_seconds is None:
        raise FirstLayerError(f"REPRESENTATION_SAMPLING_UNKNOWN: the bundle declares timeframe "
                              f"{document.get('timeframe')!r}, which cannot be read as a bar step; the step is a "
                              f"property of the data and is not defaulted here")

    # WP20 landed `min_confidence`/`abstention_source` in `m5phet.decide`. Where they are there, they are what runs:
    # the framework's rule, its refusal names, its citation, its abstention records. The local rule below is kept for
    # a checkout that does not carry them yet, and the manifest says which of the two decided this run.
    delegated = supports_min_confidence(decide)
    gate = None
    source_document, threshold_evidence, citation = None, None, None
    if min_confidence is not None:
        if not abstention_source:
            raise FirstLayerError(
                "ABSTENTION_THRESHOLD_UNSOURCED: --min-confidence decides which answers are kept and which become "
                "abstentions, so it is a measurement or it is a preference. Pass --abstention-source naming the "
                "report that measured this checkpoint at this threshold")
        if delegated:
            gate = {"min_confidence": float(min_confidence), "abstention_source": str(abstention_source)}
            citation, unresolved = decide.abstention_threshold(min_confidence, abstention_source)
            if unresolved is not None:
                # resolved before a single question is asked, so a gate nobody measured costs no model call
                raise FirstLayerError(f"{unresolved[0]}: {unresolved[1]}")
        else:
            source_document = abstention.load(abstention_source)
            threshold_evidence = abstention.threshold_entry(source_document, min_confidence)
            if threshold_evidence is None:
                declared = [entry["threshold"] for entry in source_document.get("thresholds") or ()]
                raise FirstLayerError(
                    f"ABSTENTION_THRESHOLD_NOT_IN_SOURCE: {abstention_source} measures {declared} and not "
                    f"{min_confidence}; the nearest one is not substituted, because a rule applied at one threshold "
                    f"is not evidenced by the split at another")

    points = decision_points(len(instants), every)
    stamped = as_of if as_of is not None else datetime.now(timezone.utc).isoformat()
    replay = load_records(record_dir, decide) if resume else None

    records = Path(os.path.expanduser(str(record_dir)))
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(out_path.suffix + ".partial")

    asked = answered = abstained = replayed = 0
    refusals, decided, position = [], {}, str(opening_position)
    # The series is written as it is produced: ~1 question per second against a real checkpoint means a run of a few
    # hundred bars is minutes long, and a failure in the middle of it must not cost the answers already paid for.
    with partial.open("w", encoding="utf-8") as handle:
        for index in points:
            payload = summary_payload(instrument=document.get("asset"), timeframe=document.get("timeframe"),
                                      step_seconds=step_seconds, instants=instants, closes=closes, index=index,
                                      position=position, feature_names=environment.get("feature_columns") or (),
                                      lookbacks=lookbacks, volatility_windows=volatility_windows,
                                      max_declared_columns=max_declared_columns)
            state_text = decide.decision_state(DECISION_KIND, payload)
            asked += 1
            entry = ask_one(engine, decide, state_text, as_of=stamped, record_dir=records, replay=replay, gate=gate)
            record = entry.get("decision")
            if entry.get("status") != "OK" and record is None:
                refusals.append({"bar": instants[index], "bar_index": index, "refusal": entry.get("refusal"),
                                 "why": entry.get("why")})
                # A refused question is not an action. The position is carried, the bar keeps the standing action, and
                # the refusal is counted -- a default `flat` here would be a decision nobody made.
                point = {"t": instants[index], "bar_index": index, "action": position,
                         "status": "REFUSED", "refusal": entry.get("refusal"), "decided_at": instants[index]}
            else:
                if entry.get("status") != "OK":
                    # `m5phet.decide` applied the threshold and recorded the abstention; it is a refusal that carries
                    # its record, and it is counted as an abstention and not as a failure to ask
                    action, is_abstention, confidence = ABSTENTION_ACTION, True, confidence_of(record)
                else:
                    action, is_abstention, confidence = apply_threshold(record, min_confidence)
                replayed += 1 if entry.get("replayed") else 0
                if is_abstention:
                    abstained += 1
                else:
                    answered += 1
                position = action
                point = {"t": instants[index], "bar_index": index, "action": action,
                         "status": "OK" if not is_abstention else "ABSTAINED",
                         "chosen": record.get("chosen"), "confidence": confidence,
                         "abstained": is_abstention,
                         "abstention_refusal": ABSTAINED if is_abstention else None,
                         "decision_sha256": decide.decision_sha256(record),
                         "state_sha256": record["state_sha256"],
                         "record_path": entry.get("record_path"),
                         "replayed_from_record": bool(entry.get("replayed")),
                         "decided_at": instants[index]}
            decided[index] = point
            handle.write(json.dumps(point, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if progress is not None:
                progress(asked, len(points), point)

    # One line per bar: the decision point's action, carried forward until the next one. `sweep.read_decisions` reads
    # `t` and `action` and ignores the rest, and the rest is what makes the file auditable.
    standing, lines = str(opening_position), []
    for index, moment in enumerate(instants):
        if index in decided:
            standing = decided[index]["action"]
            entry = dict(decided[index])
            entry["carried_forward"] = False
        else:
            entry = {"t": moment, "bar_index": index, "action": standing, "status": "CARRIED",
                     "decided_at": decided[max(k for k in decided if k <= index)]["t"] if decided else moment,
                     "carried_forward": True}
        entry["t"] = moment
        lines.append(entry)
    out_path.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8")
    partial.unlink(missing_ok=True)

    header = {"asked": asked, "answered_above_threshold": answered, "abstained": abstained,
              "refused": len(refusals), "replayed_from_records": replayed,
              "decision_points": len(points), "bars": len(instants), "lines": len(lines)}
    document_manifest = {
        "schema": SCHEMA,
        "execution_authorized": False,
        "header": header,
        "state_reading": STATE_READING,
        "position_reading": POSITION_READING,
        "reward_reading": REWARD_READING,
        "kind": DECISION_KIND,
        "question": QUESTION,
        "options": [list(pair) for pair in OPTIONS],
        "instructions": INSTRUCTIONS,
        "as_of": stamped,
        "policy_id": manifest["policy_id"],
        "provenance": manifest["provenance"],
        "bundle_observation_contract_sha256": document.get("contract_sha256"),
        "bars": {"path": str(Path(bars).resolve()), "sha256": sweep._sha256(bars), "rows": len(instants),
                 "first": instants[0], "last": instants[-1], "price_column": price_column},
        "summary": {"lookbacks_bars": [int(value) for value in lookbacks],
                    "volatility_windows_bars": [int(value) for value in volatility_windows],
                    "declared_columns": len(environment.get("feature_columns") or ()),
                    "declared_columns_shown": (len(environment.get("feature_columns") or ())
                                               if max_declared_columns is None
                                               else min(int(max_declared_columns),
                                                        len(environment.get("feature_columns") or ()))),
                    "declared_columns_cut_reason": (None if max_declared_columns is None else
                                                    "the classification provider enforces a token budget and refuses "
                                                    "TOKEN_BUDGET_EXCEEDED rather than truncating a state silently; "
                                                    "the list is cut here, where the cut is declared"),
                    "opening_position": str(opening_position)},
        "every_n_bars": int(every),
        "abstention": {
            "min_confidence": None if min_confidence is None else float(min_confidence),
            "refusal_name": ABSTAINED,
            "action_when_abstained": ABSTENTION_ACTION,
            "rule_implemented_by": "m5phet.decide" if delegated else "agent_multi_m5phet.first_layer",
            "rule_note": ("m5phet.decide declares `min_confidence`/`abstention_source` in this checkout, so the "
                          "framework's own rule, refusal names and citation are what ran here"
                          if delegated else
                          "m5phet.decide carries no `min_confidence` parameter in this checkout, so the same rule is "
                          "implemented in agent_multi_m5phet.first_layer and is named as local"),
            "replayed_records_note": ("a record replayed from disk was answered before any threshold existed, so the "
                                      "threshold is applied to it here, by the same rule, on the top probability it "
                                      "recorded"),
            "citation": citation,
            "source": None if source_document is None else {
                "path": str(Path(abstention_source).resolve()),
                "corpus_id": source_document.get("corpus_id"),
                "checkpoint": source_document.get("checkpoint"),
                "chance_rate": source_document.get("chance_rate"),
                "split_at_threshold": threshold_evidence,
                "transfer_assumption": source_document.get("transfer_assumption")},
        },
        "records_dir": str(records),
        "decision_sha256": sorted(point["decision_sha256"] for point in decided.values()
                                  if point.get("decision_sha256")),
        "decisions_path": str(out_path.resolve()),
        "refusals": refusals,
        "not_measured": ("NO_NEW_MEASUREMENT of quality is made here. This file is a series of choices; what they do "
                         "is what the environment says they do, and that is the sweep's number"),
    }
    document_manifest["decisions_sha256"] = sweep._sha256(out_path)
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(document_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": document_manifest, "manifest_path": str(manifest_path), "decisions_path": str(out_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m agent_multi_m5phet.first_layer",
        description="Ask the classification checkpoint for one exposure choice per decision point and write the "
                    "decision series the sweep replays. No broker, no order, no authority to execute anything.")
    parser.add_argument("--bars", required=True, help="CSV of bars carrying the fitted columns")
    parser.add_argument("--out", required=True, help="where the decision series is written (JSONL)")
    parser.add_argument("--every", type=int, default=1, help="ask every N bars (default 1: every bar of the sample)")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="below this uncalibrated confidence the choice is an abstention, counted and recorded")
    parser.add_argument("--abstention-source", default=None,
                        help="the report that measured this checkpoint at --min-confidence; required with it. The "
                             "m5phet-evaluation-report/1 with reliability bins where m5phet.decide applies the rule, "
                             f"else the {abstention.SCHEMA} document this package builds")
    parser.add_argument("--bundle", default=None, help="the fitted bundle (default: M5PHET_POLICY_BUNDLE)")
    parser.add_argument("--records", default=DEFAULT_RECORDS,
                        help=f"where decision records are written (default {DEFAULT_RECORDS})")
    parser.add_argument("--lookbacks", type=int, nargs="+", default=list(DEFAULT_LOOKBACKS),
                        help="lookbacks, in bars, the log returns are taken over")
    parser.add_argument("--volatility-windows", type=int, nargs="+", default=list(DEFAULT_VOLATILITY_WINDOWS),
                        help="windows, in bars, the realized volatility is taken over")
    parser.add_argument("--max-declared-columns", type=int, default=None,
                        help="show only the first N of the fitted policy's declared column names, so the state fits "
                             "the provider's token budget; the state declares the true count and the number shown")
    parser.add_argument("--opening-position", default="flat", choices=[key for key, _label in OPTIONS],
                        help="the position the first layer starts holding (default flat)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse the decision already recorded for an identical state instead of asking again")
    parser.add_argument("--quiet", action="store_true", help="do not print one line per decision point")
    args = parser.parse_args(argv)

    def progress(done, total, point):
        if args.quiet:
            return
        mark = point.get("chosen") or point.get("refusal") or "-"
        confidence = point.get("confidence")
        print(f"[{done}/{total}] {point['t']} {mark:<8} "
              f"{'' if confidence is None else f'{confidence:.4f} '}-> {point['action']}"
              f"{' (abstained)' if point.get('abstained') else ''}", flush=True)

    try:
        engine = _engine()
        result = run(args.bars, args.out, engine=engine, bundle=args.bundle, every=args.every,
                     min_confidence=args.min_confidence, abstention_source=args.abstention_source,
                     lookbacks=tuple(args.lookbacks), volatility_windows=tuple(args.volatility_windows),
                     record_dir=args.records, opening_position=args.opening_position, resume=args.resume,
                     max_declared_columns=args.max_declared_columns, progress=progress)
    except PolicyRefusal as refusal:
        print(json.dumps({"schema": SCHEMA, "execution_authorized": False, "refused": str(refusal)}, indent=2),
              file=sys.stderr)
        return 2
    header = result["manifest"]["header"]
    print(json.dumps({"out": result["decisions_path"], "manifest": result["manifest_path"],
                      "asked": header["asked"], "answered_above_threshold": header["answered_above_threshold"],
                      "abstained": header["abstained"], "refused": header["refused"],
                      "lines": header["lines"], "execution_authorized": False}))
    return 0


def _engine():
    """The worker route, exactly as the workbench takes it. A first layer without the real checkpoint is not one."""
    try:
        from m5phet.web.engine import Engine
    except ImportError as exc:
        raise FirstLayerError(
            "M5PHET_NOT_IMPORTABLE: the classification route lives in m5phet (`m5phet.web.engine.Engine`), and this "
            f"interpreter cannot import it ({exc}); no decision is invented in its absence") from exc
    return Engine()


if __name__ == "__main__":
    raise SystemExit(main())
