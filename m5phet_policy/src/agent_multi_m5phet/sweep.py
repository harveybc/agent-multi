"""WP21(b), replay half: a decision series replayed as a candidate policy in the gym-fx simulation.

What this is for. A first layer that chooses `long`, `flat` or `short` at each bar -- a Laya
decision series, once WP17's `decide` writes them, or any other series today -- is a hypothesis.
The only way to find out what that hypothesis does is to hand it to the environment the policies
of this repository are fitted in and let the environment answer, bar by bar, under its OWN reward.
This module does that and nothing else: it reads bars, reads decisions, replays them through
gym-fx, and writes what the environment returned, beside the `flat` baseline on the same rows and,
when the operator's bundle is configured, the fitted SAC's own actions on the same rows.

What the number is NOT. `environment_return_training_reward` is the reward the env's configured
reward plugin emitted, summed over the replayed steps. It is a training signal computed inside a
simulation with the campaign's own commission, slippage and execution rules; it is not profit, not
a backtest result and not evidence about any market. The word does not appear as a key here, and
no key in the output is a realised P&L word, because a name is how a number gets read six months
later by someone who did not run it. Nothing here reaches a broker: `execution_authorized` is
false in the output, there is no order path, and a score is not an order.

Everything that decides a number is declared in the output: the bars and their digest, the
resolved environment configuration and its digest, the decision file and its digest, and the
mapping from each decision word to the value the environment was actually given.

Configuration, all from the operator's environment, never from a request:

    M5PHET_POLICY_PYTHON   an interpreter with gymnasium, backtrader and stable-baselines3
    M5PHET_POLICY_BUNDLE   the fitted bundle; its observation contract names the columns the bars
                           must carry and the campaign configuration that fitted the policy
    M5PHET_GYM_FX          the gym-fx checkout the environment is imported from

    python -m agent_multi_m5phet.sweep --bars bars.csv --decisions decisions.jsonl --out sweep.json
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import observation as market
from .refusal import PolicyRefusal

SCHEMA = "m5phet.policy_sweep.v1"
DEFAULT_TIMEOUT_SECONDS = 1800

#: The three words a first layer may choose between, and the value each one hands the environment.
#: The environment's own action contract then decides what that value DOES -- under
#: `legacy_directional_v1` a 0.0 is HOLD and not a close, which is why the mapping is written into
#: the output rather than left as an assumption about what "flat" means.
DECISIONS = {"long": 1.0, "flat": 0.0, "short": -1.0}

#: Sections of a campaign configuration that are flattened into one environment configuration, and
#: the keys the risk section owns even when the environment section repeats them. Transcribed from
#: agent-multi `app/canonical_config.py::canonical_to_runtime`, which is what a real run uses; the
#: transcription is declared in the output so a reader can check it against that function.
CAMPAIGN_SECTIONS = ("data", "environment", "asset_policy", "lifecycle_policy", "risk")
RISK_OWNED = ("initial_cash", "position_size", "commission", "slippage", "leverage", "rel_volume",
              "min_order_volume", "max_order_volume", "size_mode", "atr_period", "k_sl", "k_tp")
SECTION_ALIASES = {("lifecycle_policy", "plugin"): "strategy_plugin",
                   ("environment", "plugin"): "env_plugin",
                   ("asset_policy", "plugin"): "agent_plugin"}

REWARD_READING = (
    "environment_return_training_reward is the gym-fx environment's own reward, summed over the "
    "replayed steps under the configured reward plugin. It is a training signal inside a "
    "simulation, not a profit, not a backtest result and not evidence about any market")
POSITION_READING = (
    "position is the environment's own signed position state; position_fraction is that state "
    "times the configured position_size, which is the exposure the configuration declares per "
    "unit of position -- not a measured allocation")


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def instant(text):
    """One text form for an instant, so a decision file and a CSV can name the same bar.

    Both spellings a person and a DataFrame produce are read; anything else is handed back as it
    stands, so an unparseable stamp fails as "not in the bars" and names itself.
    """
    try:
        return datetime.fromisoformat(str(text).strip()).isoformat()
    except ValueError:
        return str(text).strip()


# --- what the operator declared -----------------------------------------------------------------

def read_bundle(bundle):
    """(manifest, observation contract document) for a bundle, or a refusal by name."""
    from .provider import read_manifest

    manifest = read_manifest(bundle)
    if manifest is None:
        raise PolicyRefusal(
            "POLICY_BUNDLE_UNAVAILABLE: a sweep is replayed in the environment a policy was fitted "
            "in, and no operator-declared bundle names that environment")
    document = market.read_observation_contract(bundle)
    if document is None:
        raise PolicyRefusal(
            "OBSERVATION_CONTRACT_UNAVAILABLE: this bundle declares an observation SIZE and not an "
            f"observation CONTRACT; without {market.CONTRACT_FILENAME} there is no statement of "
            "which columns the bars must carry, and a sweep over the wrong columns is not an error "
            "the environment can report")
    return manifest, document


def campaign_config(path):
    """A campaign configuration flattened into the one mapping the environment reads.

    A flat configuration -- one that already names `feature_columns` at its top level -- is taken
    as it stands: that is what `gym-fx`'s own CLI consumes and there is nothing to flatten.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyRefusal(
            f"ENVIRONMENT_CONFIG_UNAVAILABLE: {path} could not be read as JSON ({exc})") from exc
    if not isinstance(document, dict):
        raise PolicyRefusal(f"ENVIRONMENT_CONFIG_UNAVAILABLE: {path} is not a JSON object")
    if "feature_columns" in document:
        return dict(document), "flat gym-fx environment configuration, taken as it stands"
    if not isinstance(document.get("environment"), dict):
        raise PolicyRefusal(
            f"ENVIRONMENT_CONFIG_UNAVAILABLE: {path} declares neither a flat environment "
            "configuration nor an `environment` section; there is nothing here to run a simulation "
            "under, and defaults chosen here would be a different simulation from the fitted one")
    flat = {}
    for section in CAMPAIGN_SECTIONS:
        values = document.get(section)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if section == "environment" and key in RISK_OWNED and isinstance(document.get("risk"), dict) \
                    and key in document["risk"]:
                continue        # the risk section owns it; the environment's copy is the legacy one
            flat[SECTION_ALIASES.get((section, key), key)] = value
    return flat, ("sections " + ", ".join(CAMPAIGN_SECTIONS) + " flattened as agent-multi "
                  "app/canonical_config.py::canonical_to_runtime flattens them, the risk section "
                  "owning " + ", ".join(RISK_OWNED))


def resolve_environment(document, env_config=None):
    """The environment configuration to replay in, and where it came from."""
    source = env_config or (document or {}).get("source_config")
    if not source:
        raise PolicyRefusal(
            "ENVIRONMENT_CONFIG_UNAVAILABLE: the bundle's observation contract names no "
            "`source_config`, so the campaign configuration that fitted this policy is not "
            "reachable from here; pass --env-config. A simulation assembled from defaults would "
            "not be the one the policy was fitted in, and its numbers would answer another "
            "question")
    if not Path(source).is_file():
        raise PolicyRefusal(
            f"ENVIRONMENT_CONFIG_UNAVAILABLE: {source} is not on disk; it is the campaign "
            "configuration that fitted this policy and nothing here substitutes for it")
    flat, how = campaign_config(source)
    if str(flat.get("env_mode", "")) != "training":
        raise PolicyRefusal(
            f"ENVIRONMENT_NOT_IN_TRAINING_MODE: this configuration declares env_mode "
            f"{flat.get('env_mode')!r}, and a sweep reports the environment's TRAINING reward; "
            "running it under another mode and calling the result a training reward would be a "
            "different number under the same name")
    if str(flat.get("action_space_mode", "")).lower() != "continuous":
        raise PolicyRefusal(
            f"ENVIRONMENT_ACTION_SPACE_NOT_CONTINUOUS: this configuration declares "
            f"action_space_mode {flat.get('action_space_mode')!r}. The decision words are mapped "
            f"onto the continuous scale {DECISIONS}; on another action space the same word would "
            "mean something else and this harness will not guess which")
    flat["mode"] = flat.get("env_mode")
    return flat, {"source_config": str(source), "source_config_sha256": _sha256(source),
                  "flattened_by": how}


# --- the bars -------------------------------------------------------------------------------------

def read_bars(path, document, date_column="DATE_TIME"):
    """(header, bar instants), after checking the bars are the ones this policy reads."""
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise PolicyRefusal(f"BARS_UNREADABLE: {path} could not be read ({exc})") from exc
    if len(rows) < 2:
        raise PolicyRefusal(f"BARS_UNREADABLE: {path} carries no rows under a header")
    header = rows[0]
    environment = document["environment"]
    needed = list(environment.get("feature_columns") or ())
    price = str(environment.get("price_column", "CLOSE"))
    date_column = str(date_column or "DATE_TIME")
    missing = [name for name in needed + [price] if name not in set(header)]
    if missing:
        raise PolicyRefusal(
            f"BARS_MISSING_FITTED_COLUMNS: {len(missing)} of the {len(needed) + 1} columns this "
            f"policy was fitted on are not in {Path(path).name}, the first being {missing[:5]}; "
            "these are engineered features and they cannot be derived from OHLC here")
    if date_column not in header:
        raise PolicyRefusal(
            f"BARS_MISSING_FITTED_COLUMNS: the bars carry no {date_column!r} column, and a "
            "decision is applied to a bar by its instant")
    required = int(document.get("required_rows") or environment.get("window_size") or 1)
    if len(rows) - 1 < required:
        raise PolicyRefusal(
            f"TOO_FEW_BARS: {len(rows) - 1} bar(s) supplied and this policy's observation needs "
            f"{required}; the environment would run, the first observations would be built on a "
            "shorter history than the policy was fitted with, and nothing would say so")
    where = header.index(date_column)
    return header, [instant(row[where]) for row in rows[1:]]


def read_decisions(path, instants):
    """The decision series, checked against the bars it claims to be about."""
    try:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise PolicyRefusal(f"DECISIONS_UNREADABLE: {path} could not be read ({exc})") from exc
    known = set(instants)
    series = {}
    counts = {word: 0 for word in DECISIONS}
    for number, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise PolicyRefusal(
                f"DECISIONS_UNREADABLE: line {number} of {Path(path).name} is not JSON ({exc}); "
                "one decision per line") from exc
        if not isinstance(entry, dict) or "t" not in entry or "action" not in entry:
            raise PolicyRefusal(
                f"DECISIONS_UNREADABLE: line {number} is not a decision; each line names an "
                '"t" (the bar\'s instant) and an "action"')
        word = str(entry["action"]).strip().lower()
        if word not in DECISIONS:
            raise PolicyRefusal(
                f"UNKNOWN_DECISION_ACTION: line {number} chooses {entry['action']!r} and the "
                f"declared choices are {sorted(DECISIONS)}; a word this harness has never heard "
                "of is not mapped onto an action scale by guessing")
        key = instant(entry["t"])
        if key not in known:
            raise PolicyRefusal(
                f"DECISION_TIMESTAMP_NOT_IN_BARS: line {number} decides for {entry['t']!r}, which "
                f"is not a bar of these bars (they run {instants[0]} .. {instants[-1]}); a "
                "decision about a bar nobody supplied cannot be replayed against it")
        if key in series:
            raise PolicyRefusal(
                f"DUPLICATE_DECISION_TIMESTAMP: {entry['t']!r} is decided twice and the two "
                "decisions cannot both be the one that was made")
        series[key] = word
        counts[word] += 1
    if not series:
        raise PolicyRefusal(f"DECISIONS_UNREADABLE: {Path(path).name} carries no decisions")
    return series, counts


# --- running the replay ---------------------------------------------------------------------------

def run_worker(payload, worker_python, gym_fx, timeout=DEFAULT_TIMEOUT_SECONDS):
    if not worker_python:
        raise PolicyRefusal(
            "POLICY_INTERPRETER_NOT_CONFIGURED: set M5PHET_POLICY_PYTHON to an interpreter with "
            "gymnasium, backtrader and stable-baselines3; this harness will not run a simulation "
            "in the interface's environment")
    if not gym_fx or not Path(gym_fx, "gym_fx").is_dir():
        raise PolicyRefusal(
            f"GYM_FX_UNAVAILABLE: {market.GYM_FX_VARIABLE} does not name a gym-fx checkout "
            f"({gym_fx!r}), and the environment this sweep replays in lives there. Nothing is "
            "simulated in its absence")
    worker = str(Path(__file__).resolve().parent / "sweep_worker.py")
    done = subprocess.run([worker_python, worker], input=json.dumps(payload, allow_nan=False),
                          text=True, capture_output=True, timeout=timeout, cwd=gym_fx,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    line = [l for l in done.stdout.splitlines() if l.startswith("{")]
    if not line:
        raise PolicyRefusal(f"SWEEP_WORKER_FAILED ({done.returncode}): {done.stderr.strip()[-400:]}")
    answer = json.loads(line[-1])
    if "error" in answer:
        raise PolicyRefusal(str(answer["error"]))
    if done.returncode:
        raise PolicyRefusal(f"SWEEP_WORKER_FAILED ({done.returncode}): {done.stderr.strip()[-400:]}")
    return answer


def sweep(bars, decisions, *, bundle=None, env_config=None, gym_fx=None, worker_python=None,
          max_steps=None, include_fitted=True, environ=None):
    """Replay one decision series, the `flat` baseline and (when configured) the fitted policy.

    The three runs differ in their actions and in nothing else: the same bars, the same resolved
    configuration, a fresh environment each time. That is what makes the last line of the output a
    comparison rather than three numbers that happen to be printed together.
    """
    env = os.environ if environ is None else environ
    bundle = bundle or env.get("M5PHET_POLICY_BUNDLE")
    gym_fx = gym_fx or env.get(market.GYM_FX_VARIABLE)
    worker_python = worker_python or env.get("M5PHET_POLICY_PYTHON")

    manifest, document = read_bundle(bundle)
    config, provenance = resolve_environment(document, env_config)
    _header, instants = read_bars(bars, document, config.get("date_column", "DATE_TIME"))
    series, counts = read_decisions(decisions, instants)

    config = dict(config)
    config["input_data_file"] = str(Path(bars).resolve())
    steps = int(max_steps) if max_steps else len(instants)

    runs = [{"name": "candidate_decisions", "kind": "series",
             "actions": {key: DECISIONS[word] for key, word in series.items()}},
            {"name": "flat", "kind": "constant", "value": DECISIONS["flat"]}]
    fitted_absent = None
    if include_fitted:
        from .provider import PolicyProvider, state_ref_for

        provider = PolicyProvider(environ={**env, "M5PHET_POLICY_BUNDLE": str(bundle)})
        try:
            provider.load(state_ref_for(manifest))
            runs.append({"name": "fitted_sac", "kind": "fitted", "checkpoint": manifest["checkpoint"]})
        except PolicyRefusal as refusal:
            # The fitted column is a comparison, not a requirement. Its absence is reported with
            # the reason, because a table missing a column for an unstated reason reads as a table
            # whose column was zero.
            fitted_absent = str(refusal)

    answer = run_worker({"gym_fx": str(gym_fx), "env_config": config, "max_steps": steps,
                         "runs": runs}, worker_python, gym_fx)

    produced = answer["runs"]
    paths = {name: [step["t"] for step in result["steps"]] for name, result in produced.items()}
    # A decision the replay never reached is not an error -- the environment starts after the
    # first bar and stops where it stops -- but it is a difference between what was decided and
    # what was scored, and a table that hid it would overstate what the series was tried on.
    applied = set(paths.get("candidate_decisions") or ())
    unused = sorted(set(series) - applied)
    reference = paths.get("candidate_decisions") or next(iter(paths.values()), [])
    same_rows = all(path == reference for path in paths.values())
    return {
        "schema": SCHEMA,
        "execution_authorized": False,
        "reading": REWARD_READING,
        "position_reading": POSITION_READING,
        "policy_id": manifest["policy_id"],
        "provenance": manifest["provenance"],
        "bars": {"path": str(Path(bars).resolve()), "sha256": _sha256(bars),
                 "rows": len(instants), "first": instants[0], "last": instants[-1]},
        "decisions": {"path": str(Path(decisions).resolve()), "sha256": _sha256(decisions),
                      "count": len(series), "by_action": counts,
                      "applied_to_bars": len(applied),
                      "never_reached_by_the_replay": len(unused),
                      "never_reached_first": unused[:5],
                      "mapping": dict(DECISIONS),
                      "action_contract": config.get("continuous_action_contract",
                                                    "legacy_directional_v1"),
                      "mapping_reading": (
                          "each word is handed to the environment on its own continuous action "
                          "scale; the environment's action contract then decides what the value "
                          "does. Under legacy_directional_v1 a 0.0 is HOLD -- it does not close an "
                          "open position, which the environment's own protective orders do")},
        "environment": {**provenance,
                        "reward_plugin": config.get("reward_plugin"),
                        "strategy_plugin": config.get("strategy_plugin"),
                        "env_mode": config.get("env_mode"),
                        "simulation_engine": config.get("simulation_engine"),
                        "initial_cash": config.get("initial_cash"),
                        "position_size": config.get("position_size"),
                        "commission": config.get("commission"),
                        "continuous_action_threshold": config.get("continuous_action_threshold"),
                        "resolved_config_sha256": _digest(config),
                        "gym_fx": str(gym_fx),
                        "observation_contract_sha256": document.get("contract_sha256")},
        "runs": produced,
        "totals": {name: result["environment_return_training_reward_total"]
                   for name, result in produced.items()},
        "comparison": {"same_rows": same_rows,
                       "rows_compared": len(reference),
                       "verdict": "COMPARABLE" if same_rows else "NOT_COMPARABLE",
                       "why": (None if same_rows else
                               "the runs did not visit the same bars: " +
                               ", ".join(f"{name} {len(path)} step(s)" for name, path in paths.items())),
                       "fitted_sac_absent": fitted_absent},
        "not_measured": (
            "NO_NEW_MEASUREMENT of quality. This is a replay in a simulation under the "
            "environment's own training reward; no held-out evaluation, no naive reference on the "
            "same rows and no closure table is produced here, and `policy_profitability` stays "
            "refused"),
        "seconds": answer.get("seconds"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m agent_multi_m5phet.sweep",
        description="Replay a decision series as a candidate policy in the gym-fx simulation. "
                    "No broker, no order, no authority to execute anything.")
    parser.add_argument("--bars", required=True, help="CSV of bars carrying the fitted columns")
    parser.add_argument("--decisions", required=True,
                        help='one JSON object per line: {"t": <instant>, "action": "long|flat|short"}')
    parser.add_argument("--out", required=True, help="where the sweep is written")
    parser.add_argument("--bundle", default=None, help="the fitted bundle (default: M5PHET_POLICY_BUNDLE)")
    parser.add_argument("--env-config", default=None,
                        help="the campaign configuration to replay under (default: the one the "
                             "bundle's observation contract names)")
    parser.add_argument("--gym-fx", default=None, help="gym-fx checkout (default: M5PHET_GYM_FX)")
    parser.add_argument("--python", default=None,
                        help="interpreter to run the simulation in (default: M5PHET_POLICY_PYTHON)")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps")
    parser.add_argument("--no-fitted", action="store_true",
                        help="leave the fitted policy's own actions out of the comparison")
    args = parser.parse_args(argv)
    try:
        result = sweep(args.bars, args.decisions, bundle=args.bundle, env_config=args.env_config,
                       gym_fx=args.gym_fx, worker_python=args.python, max_steps=args.max_steps,
                       include_fitted=not args.no_fitted)
    except PolicyRefusal as refusal:
        print(json.dumps({"schema": SCHEMA, "execution_authorized": False,
                          "refused": str(refusal)}, indent=2), file=sys.stderr)
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(Path(args.out).resolve()),
                      "totals": result["totals"],
                      "steps": result["comparison"]["rows_compared"],
                      "verdict": result["comparison"]["verdict"],
                      "execution_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
