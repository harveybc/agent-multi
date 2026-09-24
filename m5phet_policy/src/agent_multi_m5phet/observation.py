"""Market data in, the policy's own observation out -- built by gym-fx, never guessed here.

Why this module is thin on purpose. The observation a policy was fitted on is a property of the
ENVIRONMENT that produced it, not of this interface. gym-fx owns that environment, so gym-fx owns
the construction: `gym_fx.observation_builder` calls the same preprocessor the env calls and
flattens the result the same way, and everything in this file is about reaching that builder and
reporting honestly when it cannot be reached. Nothing here reimplements a window, a normalization
or a feature order. If it did, the two would drift and the policy would answer the drift without
anyone noticing -- a wrong observation is not an error, it is just a different question.

Two pieces of operator configuration, both from the environment and never from a request:

    M5PHET_GYM_FX   a gym-fx checkout to import the builder from, when it is not installed in this
                    interpreter. The interface's environment is kept without gymnasium and
                    backtrader on purpose, and the builder is importable without either.

and one file inside the policy bundle:

    <bundle>/observation_contract.json   the feature configuration the checkpoint was fitted with

The bundle manifest declares an `observation_size` and nothing else about the observation's
meaning. A length is not a contract: 2724 floats in the wrong order are still 2724 floats. So the
market-data path stays closed until the operator declares WHICH columns, in WHICH order, over
WHICH window, with WHICH normalization -- and the declared length is checked against the manifest
before a single row is read.
"""

import json
import os
import sys
from pathlib import Path

from .refusal import PolicyRefusal

CONTRACT_FILENAME = "observation_contract.json"
CONTRACT_SCHEMA = "m5phet_observation_contract.v1"
GYM_FX_VARIABLE = "M5PHET_GYM_FX"

#: Top-level package names gym-fx shares with sibling repositories, agent-multi among them. Both
#: checkouts have an `app` package, so importing gym-fx from a working directory that is an
#: agent-multi checkout would resolve `app.observation_builder` inside the WRONG repository. The
#: import below therefore hides these names for the duration of the import and puts back whatever
#: was there. The symptom this prevents is a bare ModuleNotFoundError naming a module that plainly
#: does exist in the checkout the operator pointed at.
_SHARED_TOP_LEVEL_NAMES = ("app", "preprocessor_plugins", "gym_fx")

_BUILDER_CACHE = {}


# --- reaching the builder -------------------------------------------------------------------

def _import_plainly():
    import gym_fx.observation_builder as builder

    return builder


def _import_from_checkout(root):
    saved = {name: module for name, module in list(sys.modules.items())
             if name.split(".")[0] in _SHARED_TOP_LEVEL_NAMES}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, root)
    try:
        return _import_plainly()
    finally:
        sys.path.remove(root)
        # The imported module holds direct references to everything it needs, so the shared names
        # can be handed straight back to whoever had them. Leaving gym-fx's `app` in place instead
        # would break the next `import app` in this interpreter.
        for name in [n for n in sys.modules if n.split(".")[0] in _SHARED_TOP_LEVEL_NAMES]:
            del sys.modules[name]
        sys.modules.update(saved)


def load_builder(search_root=None):
    """Return `gym_fx.observation_builder`, or refuse by name. Never a substitute.

    The refusal is deliberately plain about what to do, because "gym-fx is missing" is an
    operator's problem with an operator's fix, and the raw-vector path keeps working meanwhile.
    """
    key = search_root or ""
    if key in _BUILDER_CACHE:
        return _BUILDER_CACHE[key]
    first = None
    try:
        builder = _import_plainly()
    except ImportError as exc:
        first = exc
        builder = None
    if builder is None and search_root:
        root = str(search_root)
        if not Path(root, "gym_fx").is_dir():
            raise PolicyRefusal(
                f"GYM_FX_UNAVAILABLE: {GYM_FX_VARIABLE} points at {root!r}, which is not a gym-fx "
                "checkout (no gym_fx/ directory there). The raw-vector path is unaffected")
        try:
            builder = _import_from_checkout(root)
        except ImportError as exc:
            raise PolicyRefusal(
                f"GYM_FX_UNAVAILABLE: the observation builder could not be imported from {root!r} "
                f"({exc}). The raw-vector path is unaffected") from exc
    if builder is None:
        raise PolicyRefusal(
            "GYM_FX_UNAVAILABLE: building an observation from market data needs gym-fx, which owns "
            f"the environment this policy was fitted in, and it is not importable here ({first}). "
            f"Install gym-fx in this interpreter or set {GYM_FX_VARIABLE} to a gym-fx checkout. "
            "Nothing is guessed in its absence and the raw-vector path is unaffected")
    _BUILDER_CACHE[key] = builder
    return builder


# --- the operator's declaration of what the policy observes ------------------------------------

def read_observation_contract(bundle):
    """The bundle's observation contract, or None. A malformed one reads as absent, like the manifest."""
    if not bundle:
        return None
    path = Path(bundle) / CONTRACT_FILENAME
    try:
        declared = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(declared, dict) or declared.get("schema") != CONTRACT_SCHEMA:
        return None
    environment = declared.get("environment")
    if not isinstance(environment, dict) or not environment.get("feature_columns"):
        return None
    return declared


def contract_of(declared, builder):
    try:
        return builder.ObservationContract.from_config(declared["environment"])
    except builder.ObservationRefusal as exc:
        raise PolicyRefusal(f"OBSERVATION_CONTRACT_INVALID: {exc}") from exc


def check_against_manifest(contract, manifest):
    """The declared feature configuration must produce the length the bundle promises.

    If it does not, one of the two is wrong and there is no way to tell which from here -- so
    neither is trusted. Serving the longer or the shorter of the two would be exactly the
    pad-or-trim failure the raw-vector path already refuses.
    """
    declared = int(manifest["observation_size"])
    produced = int(contract.expected_length)
    if produced != declared:
        raise PolicyRefusal(
            f"OBSERVATION_CONTRACT_DISAGREES_WITH_BUNDLE: the declared feature configuration builds "
            f"{produced} elements and the bundle's manifest declares {declared} for this policy. "
            "One of them does not describe this checkpoint, and this provider will not pick")


# --- the state the market data does not carry --------------------------------------------------

_AGENT_STATE_FIELDS = ("position", "equity", "initial_cash", "bar_index", "total_bars",
                       "entry_price", "holding_bars")


def agent_state_for(declared, builder, contract, supplied, last_price):
    """Build the AgentState, and say where every field came from.

    A policy that observes its own position and equity is asking a question about a situation, not
    only about a market. The situation is not in the CSV. So it comes either from the caller, or
    from a default the OPERATOR wrote into the bundle -- and whichever it was is reported in the
    payload, because a default nobody sees is indistinguishable from a fact.
    """
    if not contract.include_agent_state:
        return None, "NOT_OBSERVED_BY_THIS_POLICY", {}
    default = declared.get("agent_state_default")
    source = "SUPPLIED_WITH_THE_QUESTION"
    values = {}
    if isinstance(default, dict):
        values.update({key: default[key] for key in _AGENT_STATE_FIELDS if key in default})
        source = "DECLARED_DEFAULT_IN_THE_BUNDLE"
    if supplied is not None:
        if not isinstance(supplied, dict):
            raise PolicyRefusal(
                "AGENT_STATE_MUST_BE_A_MAPPING: agent_state names position, equity, initial_cash, "
                "bar_index and total_bars")
        unknown = sorted(set(supplied) - set(_AGENT_STATE_FIELDS))
        if unknown:
            raise PolicyRefusal(
                f"UNKNOWN_AGENT_STATE_FIELD: {unknown} is not part of this observation; the fields "
                f"are {list(_AGENT_STATE_FIELDS)}")
        values.update(supplied)
        source = ("CALLER_OVERRODE_THE_DECLARED_DEFAULT"
                  if isinstance(default, dict) else "SUPPLIED_WITH_THE_QUESTION")
    missing = [key for key in ("position", "equity", "initial_cash", "bar_index", "total_bars")
               if key not in values]
    if missing:
        raise PolicyRefusal(
            f"AGENT_STATE_REQUIRED: this policy observes its own state and {missing} were not "
            "supplied and are not declared in the bundle's observation contract. "
            + builder.AGENT_STATE_NOTE)
    try:
        state = builder.AgentState(
            position=int(values["position"]),
            equity=float(values["equity"]),
            initial_cash=float(values["initial_cash"]),
            price=float(last_price),
            bar_index=int(values["bar_index"]),
            total_bars=int(values["total_bars"]),
            entry_price=float(values.get("entry_price", 0.0) or 0.0),
            holding_bars=int(values.get("holding_bars", 0) or 0))
    except (builder.ObservationRefusal, TypeError, ValueError) as exc:
        raise PolicyRefusal(f"INVALID_AGENT_STATE: {exc}") from exc
    return state, source, {key: values[key] for key in _AGENT_STATE_FIELDS if key in values}


# --- the one thing this module does ------------------------------------------------------------

def build_from_rows(rows, declared, manifest, search_root=None):
    """Market data rows -> (observation as a list of floats, a declaration of how it was built).

    Every refusal gym-fx raises is passed through with its own reason code intact. Rewriting them
    here would cost the caller the one thing that makes a refusal useful: which rule was broken.
    """
    builder = load_builder(search_root)
    contract = contract_of(declared, builder)
    check_against_manifest(contract, manifest)

    supplied_state = None
    payload_rows = rows
    if isinstance(rows, dict):
        supplied_state = rows.get("agent_state")
        payload_rows = rows.get("rows")
        if payload_rows is None:
            raise PolicyRefusal(
                "MARKET_DATA_REQUIRED: a market-data object carries its bars under 'rows'")

    # CSV text is turned into rows here rather than inside the build, because the price of the last
    # bar is read before the build and reading it out of raw text would be a second parser.
    if isinstance(payload_rows, str):
        try:
            payload_rows = builder.rows_from_csv(payload_rows)
        except builder.ObservationRefusal as exc:
            raise PolicyRefusal(str(exc)) from exc

    # The agent-state block is priced against the most recent supplied bar; a policy that does not
    # observe agent state needs no price at all, so none is demanded of the caller.
    last_price = (_last_price(payload_rows, contract.price_column)
                  if contract.include_agent_state else None)
    state, source, values = agent_state_for(declared, builder, contract, supplied_state,
                                            last_price)
    try:
        built = builder.build_observation(
            payload_rows, contract, state,
            rows_start_the_series=bool(declared.get("rows_start_the_series", False)))
    except builder.ObservationRefusal as exc:
        raise PolicyRefusal(str(exc)) from exc

    declaration = built.declaration()
    declaration["agent_state_source"] = source
    declaration["agent_state_values"] = values
    declaration["price_at_the_last_row"] = (float(last_price) if last_price is not None else None)
    declaration["built_by"] = "gym_fx.observation_builder"
    declaration["policy_id"] = manifest["policy_id"]
    return built.as_list(), declaration


def _last_price(rows, price_column):
    """The close of the most recent supplied bar, which the agent-state block is priced against."""
    try:
        if hasattr(rows, "iloc"):
            return float(rows[price_column].iloc[-1])
        listed = list(rows)
        return float(listed[-1][price_column])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise PolicyRefusal(
            f"MISSING_COLUMNS: the price column {price_column!r} is not readable from the last row "
            "of the supplied data, and the agent-state block is priced against it") from exc


# --- telling market data apart from a raw vector ------------------------------------------------

def looks_like_rows(value):
    """True for a list of row objects -- the shape a parsed CSV attachment takes.

    Only the SHAPE is decided here. Whether those rows carry the right columns is gym-fx's
    judgement, and routing a table to the raw-vector path so it could be refused as "not a list of
    numbers" would tell the person to fix the wrong thing.
    """
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(row, dict) for row in value)


def looks_like_csv_text(value):
    """True for text that is a delimited table rather than prose.

    A header line with a separator and at least one further line. Prose has neither, and treating
    a sentence as a one-column CSV would answer a typo with a column-name complaint.
    """
    if not isinstance(value, str):
        return False
    lines = [line for line in value.splitlines() if line.strip()]
    return len(lines) >= 2 and "," in lines[0]


def gym_fx_root(environ=None):
    env = os.environ if environ is None else environ
    return env.get(GYM_FX_VARIABLE) or None
