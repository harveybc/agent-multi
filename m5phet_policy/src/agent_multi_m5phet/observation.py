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

    M5PHET_REPRESENTATION
                    a feature-eng checkout holding `feature_eng_m5phet.representation`, the module
                    that DECLARES `m5phet.representation.v1`. Only needed to read a spec, and only
                    when that package is not installed in this interpreter. A spec is never
                    validated by a second copy of those rules written here.

and one file inside the policy bundle:

    <bundle>/observation_contract.json   the feature configuration the checkpoint was fitted with

The bundle manifest declares an `observation_size` and nothing else about the observation's
meaning. A length is not a contract: 2724 floats in the wrong order are still 2724 floats. So the
market-data path stays closed until the operator declares WHICH columns, in WHICH order, over
WHICH window, with WHICH normalization -- and the declared length is checked against the manifest
before a single row is read.

**A representation spec may ask for the observation instead** (WP21a). `m5phet.representation.v1`
is the object the design job of WP06 emits: sampling, windows, lags, differencing, calendar clock,
features, exogenous columns. `observation_from_spec` builds the observation through the same
gym-fx construction from what that spec declares, and refuses `OBSERVATION_CONTRACT_MISMATCH` when
the spec does not reproduce the FITTED policy's contract -- naming the length and the columns
expected against produced. A policy answers only the representation it was fitted on: any other
spec would be answered, confidently, about data the policy never saw. `policy_representation_spec`
exports the fitted policy's own representation in that same shape, so the two can be compared, and
so that a person can start from what the policy actually reads instead of guessing at it.
"""

import json
import os
import sys
from pathlib import Path

from .refusal import PolicyRefusal

CONTRACT_FILENAME = "observation_contract.json"
CONTRACT_SCHEMA = "m5phet_observation_contract.v1"
GYM_FX_VARIABLE = "M5PHET_GYM_FX"
REPRESENTATION_VARIABLE = "M5PHET_REPRESENTATION"
REPRESENTATION_SCHEMA = "m5phet.representation.v1"

#: Top-level package names gym-fx shares with sibling repositories, agent-multi among them. Both
#: checkouts have an `app` package, so importing gym-fx from a working directory that is an
#: agent-multi checkout would resolve `app.observation_builder` inside the WRONG repository. The
#: import below therefore hides these names for the duration of the import and puts back whatever
#: was there. The symptom this prevents is a bare ModuleNotFoundError naming a module that plainly
#: does exist in the checkout the operator pointed at.
_SHARED_TOP_LEVEL_NAMES = ("app", "preprocessor_plugins", "gym_fx")

#: The same hazard for feature-eng, whose checkout also carries a top-level `app` package.
_SHARED_REPRESENTATION_NAMES = ("app", "feature_eng_m5phet")

_BUILDER_CACHE = {}
_REPRESENTATION_CACHE = {}


# --- reaching the builder -------------------------------------------------------------------

def _import_plainly():
    import gym_fx.observation_builder as builder

    return builder


def _import_from_checkout(root, importer=None, shared_names=_SHARED_TOP_LEVEL_NAMES):
    saved = {name: module for name, module in list(sys.modules.items())
             if name.split(".")[0] in shared_names}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, root)
    try:
        return (importer or _import_plainly)()
    finally:
        sys.path.remove(root)
        # The imported module holds direct references to everything it needs, so the shared names
        # can be handed straight back to whoever had them. Leaving gym-fx's `app` in place instead
        # would break the next `import app` in this interpreter.
        for name in [n for n in sys.modules if n.split(".")[0] in shared_names]:
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


# --- the representation a spec declares, and whether it is this policy's ------------------------
#
# WP21(a). `m5phet.representation.v1` (feature-eng `feature_eng_m5phet/representation.py`,
# `M5PHET/docs/REPRESENTATION_SPEC.md`) is the object that names what a model is allowed to see:
# sampling, windows, lags, differencing, calendar clock and columns, features, exogenous columns.
# A policy was fitted under exactly one of those, so a spec is either THIS policy's representation
# or it is a question for a different policy. There is no third case, and no partial credit: a
# vector built from another representation has the right length and the wrong meaning, and the
# policy answers it as confidently as the right one.
#
# What a spec decides here, and what it does not. The spec decides the window, the columns and the
# column the price series is read from. It says nothing about the env's normalization, its clip,
# its agent-state block or its position size -- `m5phet.representation.v1` has no key for any of
# them -- so those are taken from the FITTED contract and declared as such. That asymmetry is why
# the comparison below is a comparison of contracts rather than of specs: two specs can differ in
# annotations and build the same observation, and the only question that matters is whether the
# observation is the one this policy reads.

def _import_representation_plainly():
    import feature_eng_m5phet.representation as representation

    return representation


def load_representation_reader(search_root=None):
    """`feature_eng_m5phet.representation`, or None. Never a second copy of its rules.

    None is not a failure: the caller falls back to reading only the keys it actually uses and
    says so in the declaration, which is honest about what was and was not checked.
    """
    key = search_root or ""
    if key in _REPRESENTATION_CACHE:
        return _REPRESENTATION_CACHE[key]
    module = None
    try:
        module = _import_representation_plainly()
    except ImportError:
        module = None
    if module is None and search_root and Path(search_root, "feature_eng_m5phet").is_dir():
        try:
            module = _import_from_checkout(str(search_root), _import_representation_plainly,
                                           _SHARED_REPRESENTATION_NAMES)
        except ImportError:
            module = None
    _REPRESENTATION_CACHE[key] = module
    return module


def representation_root(environ=None):
    env = os.environ if environ is None else environ
    return env.get(REPRESENTATION_VARIABLE) or None


#: the keys of a spec this module reads. Everything else in `m5phet.representation.v1` is about
#: FITTING -- the holdout, the provenance, the annotations -- and changes no element of a vector.
_SPEC_KEYS_READ = ("schema", "windows", "lags", "differencing", "target", "features", "exogenous")


def validate_representation(spec, search_root=None):
    """Read one spec, or refuse it by name. Returns (spec, what validated it).

    The declaring module validates when it is reachable, because it owns the vocabulary, the
    clocks and every refusal code the spec's own documentation promises. When it is not, the keys
    this module reads are checked here and the declaration says exactly that -- a spec that passed
    a partial reading must never be reported as one that passed the full one.
    """
    if not isinstance(spec, dict):
        raise PolicyRefusal(
            "REPRESENTATION_SPEC_REQUIRED: a representation is a JSON object of schema "
            f"{REPRESENTATION_SCHEMA!r}, and {type(spec).__name__} is not one")
    if spec.get("schema") != REPRESENTATION_SCHEMA:
        raise PolicyRefusal(
            f"REPRESENTATION_SPEC_INVALID: WRONG_SCHEMA: the representation declares "
            f"{spec.get('schema')!r} and this path reads {REPRESENTATION_SCHEMA!r}")
    reader = load_representation_reader(search_root)
    if reader is not None:
        try:
            reader.validate_spec(spec)
        except reader.SpecError as exc:
            raise PolicyRefusal(f"REPRESENTATION_SPEC_INVALID: {exc}") from exc
        return spec, "feature_eng_m5phet.representation.validate_spec"
    missing = [key for key in _SPEC_KEYS_READ if key not in spec and key != "exogenous"]
    if missing:
        raise PolicyRefusal(
            f"REPRESENTATION_SPEC_INVALID: MISSING_KEY: the representation is missing {missing}, "
            f"which this path reads to build an observation. The module that declares "
            f"{REPRESENTATION_SCHEMA!r} is not importable here, so only the keys used were "
            f"checked; install feature-eng or set {REPRESENTATION_VARIABLE} for the full reading")
    return spec, "keys used by this path only (feature_eng_m5phet.representation not importable)"


def spec_columns(spec):
    """The feature columns a spec declares, in the order the observation lays them out.

    `features` first, then `exogenous`: the closed vocabulary feature-eng builds, then the raw
    columns of the caller's own file. The order is part of the contract -- the policy's first
    layer is indexed by it -- so it is stated here once rather than inferred anywhere.
    """
    return list(spec.get("features") or []) + list(spec.get("exogenous") or [])


def contract_from_spec(spec, fitted, builder):
    """The observation contract a representation spec describes, or a refusal by name.

    Everything the spec declares is taken from the spec. Everything the env decides and the spec
    has no key for -- scaling, its window, the clip, the price window, the agent-state block, the
    position size -- is taken from `fitted`, because inventing it here would be a third opinion
    about an observation that already has two.
    """
    windows = list(spec.get("windows") or [])
    if len(windows) != 1:
        raise PolicyRefusal(
            f"REPRESENTATION_NOT_OBSERVABLE: the representation declares {len(windows)} windows "
            f"{windows} and this environment observes exactly one; keeping the longest would drop "
            "the others silently and build a window nobody declared")
    if spec.get("lags"):
        raise PolicyRefusal(
            f"REPRESENTATION_NOT_OBSERVABLE: the representation declares individual lags "
            f"{spec['lags']}, and the env's observation is a contiguous window plus the agent's "
            "own state; there is no block for a lag and one would have to be invented")
    if int((spec.get("differencing") or {}).get("order", 0)) != 0:
        raise PolicyRefusal(
            "REPRESENTATION_NOT_OBSERVABLE: the representation differences the target, and the "
            "env reads the columns of the file as they stand; a difference taken here would be a "
            "transform the policy was not fitted under")
    transform = (spec.get("target") or {}).get("transform")
    if transform != "level":
        raise PolicyRefusal(
            f"REPRESENTATION_NOT_OBSERVABLE: the representation models the target under "
            f"{transform!r}; the env's price window is the level of its price column, and the "
            "returns block beside it is the env's own construction, not this transform")
    columns = spec_columns(spec)
    config = dict(fitted.to_config())
    config.update({"window_size": windows[0], "feature_columns": columns,
                   "price_column": (spec.get("target") or {}).get("column", fitted.price_column)})
    # A binary column the spec does not carry cannot stay declared: `from_config` refuses a binary
    # name that is not a feature, and that refusal would hide the difference the caller must see.
    config["feature_binary_columns"] = [name for name in fitted.feature_binary_columns
                                        if name in set(columns)]
    try:
        return builder.ObservationContract.from_config(config)
    except builder.ObservationRefusal as exc:
        raise PolicyRefusal(f"REPRESENTATION_NOT_OBSERVABLE: {exc}") from exc


def check_spec_against_contract(derived, fitted):
    """The derived contract must be the fitted one, element for element, or nothing is served."""
    differences = derived.differences_from(fitted)
    if not differences:
        return
    raise PolicyRefusal(
        "OBSERVATION_CONTRACT_MISMATCH: the representation builds "
        f"{derived.expected_length} elements from {len(derived.feature_columns)} feature columns "
        f"and this policy was fitted on {fitted.expected_length} elements from "
        f"{len(fitted.feature_columns)} columns -- " + "; ".join(differences)
        + ". A policy answers only the representation it was fitted on; the vector this spec "
          "describes would be answered as confidently as the right one")


def policy_representation_spec(document, *, step_seconds=None, timezone="UTC", clock="receipt",
                               provenance=None):
    """The FITTED policy's own representation, written as `m5phet.representation.v1`.

    `document` is the bundle's observation contract (`observation_contract.json`), because that is
    the only place the policy's representation exists: the manifest declares a length, and a
    length is not a representation. The point of this export is comparison -- feed it back to
    `observation_from_spec` and it must reproduce the builder element for element -- and, for a
    person, a starting point that is what the policy reads rather than a guess at it.

    Three keys `m5phet.representation.v1` requires decide nothing about an observation: the
    holdout, the availability clock over an empty column list, and the timezone of a series whose
    calendar features are empty. They are declared, listed in `not_decided`, and excluded from
    every comparison this module makes. Nothing else is filled in: an unparseable timeframe is
    refused rather than defaulted, because the sampling step is a real property of the data.
    """
    if not isinstance(document, dict) or not isinstance(document.get("environment"), dict):
        raise PolicyRefusal(
            "OBSERVATION_CONTRACT_UNAVAILABLE: a policy's representation is exported from its "
            "bundle's observation contract, which declares the columns, the order and the window; "
            "a manifest declares a length and a length is not a representation")
    environment = document["environment"]
    step = step_seconds if step_seconds is not None else _seconds_of(document.get("timeframe"))
    if step is None:
        raise PolicyRefusal(
            f"REPRESENTATION_SAMPLING_UNKNOWN: the bundle declares timeframe "
            f"{document.get('timeframe')!r}, which this exporter cannot read as a sampling step; "
            "pass step_seconds. The step is a property of the data and is not defaulted here")
    columns = [str(name) for name in environment.get("feature_columns") or ()]
    price = str(environment.get("price_column", "CLOSE"))
    stated = str(document.get("provenance") or "")
    if provenance is None:
        provenance = next((word for word in ("DEVELOPMENT", "GOVERNED", "PRODUCTION")
                           if stated.upper().startswith(word)), "DEVELOPMENT")
    return {
        "schema": REPRESENTATION_SCHEMA,
        "sampling": {"step_seconds": int(step), "timezone": timezone},
        "target": {"column": price, "transform": "level"},
        "windows": [int(environment["window_size"])],
        "lags": [],
        "differencing": {"order": 0},
        "calendar": {"clock": clock, "columns": []},
        # feature-eng's closed vocabulary names features feature-eng builds. A policy's columns are
        # the columns of the campaign's own model-ready export, so they are declared where the spec
        # declares open names -- and `exogenous` says, in the spec's own words, that the file holds
        # them and no vocabulary can check them.
        "features": [],
        "exogenous": [name for name in columns if name != price],
        "holdout": {"fraction": 0.2},
        "provenance": provenance,
        "fitted_state_ref": (f"policy:{document['checkpoint_sha256']}"
                             if document.get("checkpoint_sha256") else None),
        "candidate_id": f"fitted:{document.get('policy_id')}",
        "why": {"windows": f"the fitted observation contract declares window_size "
                           f"{environment.get('window_size')}",
                "exogenous": f"the {len(columns)} columns this policy was fitted on, in the fitted "
                             f"order; feature-eng's vocabulary does not declare them and cannot "
                             f"check them",
                "target": f"the env reads the level of {price!r} for its price window"},
        "not_decided": {
            "holdout": "a fitted policy's observation has no holdout; the schema requires the key, "
                       "this value decides nothing here and no comparison reads it",
            "calendar.clock": "this representation carries no calendar column, so no availability "
                              "clock applies to anything; the key is required and is not a claim",
            "sampling.timezone": "the bundle declares a timeframe and not a zone; with no calendar "
                                 "feature declared, nothing in this representation reads it"},
    }


def _seconds_of(timeframe):
    """`"4h"` -> 14400. Unreadable -> None, which the caller refuses rather than defaults."""
    text = str(timeframe or "").strip().lower()
    units = {"s": 1, "m": 60, "min": 60, "h": 3600, "d": 86400, "w": 604800}
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            head = text[: -len(suffix)] or "1"
            if head.isdigit() and int(head) > 0:
                return int(head) * units[suffix]
    return None


def fitted_contract_of(contract, builder):
    """Accept either a gym-fx contract or the bundle's declared contract document.

    The provider holds the document, a test holds the contract, and making one of them convert
    before calling would put the conversion in two places.
    """
    if isinstance(contract, builder.ObservationContract):
        return contract, None
    if isinstance(contract, dict) and isinstance(contract.get("environment"), dict):
        return contract_of(contract, builder), contract
    if isinstance(contract, dict):
        try:
            return builder.ObservationContract.from_config(contract), None
        except builder.ObservationRefusal as exc:
            raise PolicyRefusal(f"OBSERVATION_CONTRACT_INVALID: {exc}") from exc
    raise PolicyRefusal(
        "OBSERVATION_CONTRACT_UNAVAILABLE: the fitted policy's contract is required to judge a "
        "representation; pass the bundle's observation contract or a gym-fx ObservationContract")


def observation_from_spec(rows, spec, contract, manifest=None, search_root=None):
    """Market data + a representation spec -> the observation, or a refusal by name.

    The build is the same one `build_from_rows` performs -- gym-fx's, not this module's. What the
    spec adds is a question asked before any row is read: is this the representation the policy
    was fitted on? If it is, the answer is identical to the fitted path, element for element, and
    the declaration says which spec was honoured. If it is not, nothing is built.
    """
    builder = load_builder(search_root)
    fitted, declared = fitted_contract_of(contract, builder)
    if manifest is not None:
        check_against_manifest(fitted, manifest)
    validated, validated_by = validate_representation(spec, representation_root())
    derived = contract_from_spec(validated, fitted, builder)
    check_spec_against_contract(derived, fitted)

    observation, declaration = _build_with_contract(rows, declared or {}, derived, builder,
                                                    manifest)
    reader = load_representation_reader(representation_root())
    declaration["representation"] = {
        "schema": REPRESENTATION_SCHEMA,
        "spec_id": reader.spec_id(validated) if reader is not None else None,
        "candidate_id": validated.get("candidate_id"),
        "validated_by": validated_by,
        "window": validated["windows"][0],
        "feature_columns": len(spec_columns(validated)),
        "price_column": validated["target"]["column"],
        "taken_from_the_fitted_contract": ["feature_scaling", "feature_scaling_window",
                                           "feature_clip", "feature_binary_columns",
                                           "include_price_window", "include_agent_state",
                                           "agent_state_contract", "position_size"],
        "reading": ("this representation was checked against the policy's fitted observation "
                    "contract and reproduces it; the keys listed above are the env's and the "
                    "representation schema has no field for them"),
    }
    return observation, declaration


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
    return _build_with_contract(rows, declared, contract, builder, manifest)


def _build_with_contract(rows, declared, contract, builder, manifest=None):
    """The build itself, once the contract to build under has been settled.

    Shared by the fitted path and the representation-spec path so that "the same construction" is
    a fact about the code and not a claim in a docstring.
    """
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
    declaration["policy_id"] = manifest["policy_id"] if manifest else None
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
