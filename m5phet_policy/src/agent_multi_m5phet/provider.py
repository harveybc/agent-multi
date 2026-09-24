"""The M5PHET policy provider: an existing fitted policy proposes an action, and nothing else happens.

This wraps a policy that was already trained. It does not train, it does not size a position, it does not reach a broker,
and it never returns anything a caller could mistake for an instruction to trade: `execution_authorized` is False in every
payload it produces, and the runtime rejects the result if it is not.

The policy engine lives in its own interpreter. Stable-Baselines3 and Torch do not belong in the interface's environment, so
inference runs in a subprocess against the interpreter the operator configured, exactly as the forecast provider does. The
subprocess receives the observation and returns the action; it is never given a prompt, a path from a user, or a shell.

Configuration, all from the operator's environment and never from a request:

    M5PHET_POLICY_PYTHON   interpreter that has stable-baselines3 installed
    M5PHET_POLICY_BUNDLE   directory holding manifest.json, which names the checkpoint and its digest

A bundle names a checkpoint by absolute path and records its sha256. The digest is verified before the policy is loaded, so
a checkpoint that changed on disk is refused rather than silently served under the identity of the one that was reviewed.

A caller may supply either the observation vector itself, or the market data it is built from. The second is what a person
actually has, and the first is what nobody can type. The vector path is unchanged and needs nothing installed; the market
data path builds the observation through gym-fx, which owns the environment the policy was fitted in, and refuses plainly
when gym-fx is not reachable rather than falling back to a construction guessed here. See `observation.py`.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import observation as market
from .refusal import PolicyRefusal

NAME = "trading_policy"
FAMILY = "policy"
OUTPUT_KIND = "policy_action"
SUPPORTED = ({"operation": "infer", "family": FAMILY, "output_kind": OUTPUT_KIND},)

#: a deterministic policy returns one action; it carries no distribution, and calling that "uncertainty" would invent one
UNCERTAINTY = "NONE_DETERMINISTIC_POLICY"

DEFAULT_TIMEOUT_SECONDS = 120


def _digest_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_manifest(bundle):
    """The operator's declaration of which fitted policy this provider may serve."""
    if not bundle:
        return None
    path = Path(bundle) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    required = ("schema", "policy_id", "checkpoint", "checkpoint_sha256", "observation_size",
                "action_size", "action_low", "action_high", "unit", "action_space", "provenance")
    if not isinstance(manifest, dict) or any(manifest.get(field) in (None, "") for field in required):
        return None
    if manifest["schema"] != "m5phet_policy_bundle.v1":
        return None
    return manifest


def state_ref_for(manifest):
    return f"policy:{manifest['checkpoint_sha256']}"


class PolicyProvider:
    """One fitted policy, one declared combination, one subprocess boundary."""

    name = NAME

    def __init__(self, environ=None, runner=None):
        env = os.environ if environ is None else environ
        self.bundle = env.get("M5PHET_POLICY_BUNDLE")
        self.worker_python = env.get("M5PHET_POLICY_PYTHON")
        self.timeout = int(env.get("M5PHET_POLICY_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
        self.manifest = read_manifest(self.bundle)
        # The bundle's observation contract is what makes market data usable. It is read here, but
        # nothing is imported from gym-fx until a request actually needs it: a provider that has
        # only ever been asked for a raw vector must not depend on an install it never uses.
        self.observation_contract = market.read_observation_contract(self.bundle)
        self.gym_fx_root = market.gym_fx_root(env)
        self._runner = runner or self._subprocess
        self.last_call = None

    # --- what the runtime checks before anything is loaded -------------------------------------------------------
    def capabilities(self):
        manifest = self.manifest
        return {"provider": self.name,
                "operations": ["infer"],
                "families": [FAMILY],
                "output_kinds": [OUTPUT_KIND],
                "uncertainty_methods": [UNCERTAINTY],
                "supported": [dict(entry) for entry in SUPPORTED],
                "known_states": [state_ref_for(manifest)] if manifest else [],
                "backend": "stable_baselines3_cpu_subprocess" if self.worker_python else "not_configured",
                "policy_id": manifest["policy_id"] if manifest else None,
                "observation_size": manifest["observation_size"] if manifest else None,
                "action_space": manifest["action_space"] if manifest else None,
                "provenance": manifest["provenance"] if manifest else None,
                "market_data": self.market_data_readiness(),
                "reading": ("this provider proposes the action an ALREADY FITTED policy returns for one observation. It is "
                            "not a recommendation, not a position size and not an order; nothing here can authorize one")}

    def market_data_readiness(self):
        """Whether market data can be turned into this policy's observation, and if not, why not.

        Declared rather than discovered at request time, so a person can see before they attach a file that this policy
        needs, say, 256 rows of 83 named columns -- and so the answer to "why did it refuse my CSV" is not a surprise."""
        manifest = self.manifest
        if manifest is None:
            return {"supported": False, "why": "no operator-declared bundle is configured"}
        declared = self.observation_contract
        if declared is None:
            return {"supported": False,
                    "why": (f"this bundle declares an observation SIZE and not an observation CONTRACT; add "
                            f"{market.CONTRACT_FILENAME} naming the feature columns, their order, the window and the "
                            "normalization this policy was fitted with. A length alone cannot say what the numbers mean"),
                    "expected_file": str(Path(self.bundle) / market.CONTRACT_FILENAME)}
        try:
            builder = market.load_builder(self.gym_fx_root)
            contract = market.contract_of(declared, builder)
            market.check_against_manifest(contract, manifest)
        except PolicyRefusal as refusal:
            return {"supported": False, "why": str(refusal)}
        return {"supported": True,
                "required_rows": contract.required_rows,
                "observation_length": contract.expected_length,
                "window_size": contract.window_size,
                "feature_count": len(contract.feature_columns),
                "price_column": contract.price_column,
                "feature_scaling": contract.feature_scaling,
                "contract_sha256": contract.digest,
                "asset": declared.get("asset"),
                "timeframe": declared.get("timeframe"),
                "built_by": "gym_fx.observation_builder",
                "reading": ("attach a CSV, or a JSON list of row objects, whose columns are the ones this policy was "
                            "fitted on, ending at the bar you are asking about. The observation is built by the "
                            "environment's own code; it is never padded, trimmed or zero-filled to fit")}

    def known_states(self):
        return self.capabilities()["known_states"]

    # --- the fitted state ----------------------------------------------------------------------------------------
    def load(self, state_ref):
        manifest = self.manifest
        if manifest is None:
            raise PolicyRefusal("POLICY_BUNDLE_UNAVAILABLE: no operator-declared bundle is configured")
        if state_ref != state_ref_for(manifest):
            raise PolicyRefusal(f"UNKNOWN_FITTED_STATE: {state_ref!r} is not this provider's declared policy")
        checkpoint = Path(manifest["checkpoint"])
        if not checkpoint.is_file():
            raise PolicyRefusal(f"CHECKPOINT_MISSING: {checkpoint.name} is not on disk")
        observed = _digest_file(checkpoint)
        if observed != manifest["checkpoint_sha256"]:
            raise PolicyRefusal(
                f"CHECKPOINT_CHANGED: the file hashes to {observed[:12]} and the bundle declares "
                f"{manifest['checkpoint_sha256'][:12]}; a policy that changed is not the policy that was reviewed")
        return {"state_ref": state_ref, "digest": manifest["checkpoint_sha256"],
                "model_sha256": manifest["checkpoint_sha256"], "task_id": manifest["policy_id"],
                "compatible_task_ids": [manifest["policy_id"]],
                "identity": {"policy_id": manifest["policy_id"], "provenance": manifest["provenance"],
                             "observation_size": manifest["observation_size"],
                             "action_space": manifest["action_space"]}}

    # --- inference -----------------------------------------------------------------------------------------------
    def _subprocess(self, payload):
        if not self.worker_python:
            raise PolicyRefusal("POLICY_INTERPRETER_NOT_CONFIGURED: set M5PHET_POLICY_PYTHON to an interpreter with "
                                "stable-baselines3; this provider will not load a policy in the interface's environment")
        worker = str(Path(__file__).resolve().parent / "worker.py")
        done = subprocess.run([self.worker_python, worker], input=json.dumps(payload, allow_nan=False),
                              text=True, capture_output=True, timeout=self.timeout,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        line = [l for l in done.stdout.splitlines() if l.startswith("{")]
        if done.returncode or not line:
            raise PolicyRefusal(f"POLICY_WORKER_FAILED ({done.returncode}): {done.stderr.strip()[-200:]}")
        return json.loads(line[-1])

    def infer(self, request, state):
        manifest = self.manifest
        schema = request.get("output_schema") or {}
        requested = list(schema.get("targets") or [])
        inputs = request.get("inputs") or {}
        observation = inputs.get("observation")
        problem = None
        if not requested:
            problem = "POLICY_TARGET_REQUIRED: name the action this request is asking for"
        elif not isinstance(observation, list) or not observation:
            problem = "OBSERVATION_REQUIRED: supply the observation vector this policy was fitted on"
        elif len(observation) != manifest["observation_size"]:
            problem = (f"OBSERVATION_SIZE_MISMATCH: this policy observes {manifest['observation_size']} values and "
                       f"{len(observation)} were supplied; padding or trimming would be a different observation")
        elif any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in observation):
            problem = "OBSERVATION_MUST_BE_FINITE_NUMBERS: a policy observation is numeric, and a boolean is not a number"
        if problem is not None:
            return {"outputs": {name: {"status": "INVALID_INPUT", "why": problem} for name in (requested or ["action"])}}
        answer = self._runner({"checkpoint": manifest["checkpoint"], "observation": [float(v) for v in observation],
                               "deterministic": True})
        self.last_call = {"seconds": answer.get("seconds"), "policy_id": manifest["policy_id"]}
        action = answer.get("action")
        if not isinstance(action, list) or len(action) != manifest["action_size"]:
            return {"outputs": {name: {"status": "PROVIDER_ERROR",
                                       "why": "the policy returned an action of the wrong shape"} for name in requested}}
        low, high = manifest["action_low"], manifest["action_high"]
        clipped = [v for v in action if not low - 1e-9 <= v <= high + 1e-9]
        # If the observation was built from market data, the build travels with the action. A vector of 2724 floats is
        # unfalsifiable on its own: it looks the same whether it came from the right 256 rows or from a zero-fill.
        build = inputs.get("observation_build")
        payload = {"action": [float(v) for v in action],
                   "unit": manifest["unit"],
                   "action_space": manifest["action_space"],
                   "policy_id": manifest["policy_id"],
                   "deterministic": True,
                   "out_of_declared_range": clipped,
                   "execution_authorized": False,
                   "observation_source": "BUILT_FROM_MARKET_DATA" if isinstance(build, dict) else "SUPPLIED_AS_A_VECTOR",
                   "observation_build": copy.deepcopy(build) if isinstance(build, dict) else None,
                   "reading": ("the action this fitted policy returns for this observation. It is a proposal from a model, "
                               "not advice, not a position size and not an order")}
        return {"outputs": {name: {"status": "OK", "uncertainty": UNCERTAINTY, "payload": copy.deepcopy(payload)}
                            for name in requested},
                "population": {"observations": 1, "observation_sha256": digest(observation)}}

    # --- the chat adapter ----------------------------------------------------------------------------------------
    def chat_slots(self):
        """The vocabulary a person's words may be resolved against: exactly what this bundle declares, and nothing more.

        A bundle names one fitted policy, so `policy_id` is the only thing here with an enumerable set of values. The
        observation is not a choice -- it is a vector of numbers supplied with the question -- and determinism is not
        offered as an option, so neither is declared. An open slot cannot be validated against anything, so declaring one
        would hand the interpreter a field it could fill with a name this provider has never heard of; a bundle-less
        provider therefore declares nothing at all."""
        manifest = self.manifest
        if manifest is None:
            return []
        policy_id = manifest["policy_id"]
        spoken = " ".join(part for part in policy_id.replace("-", "_").split("_") if part)
        aliases = ["policy", "model", "agent", "politica", "pol\u00edtica", "modelo"]
        if spoken and spoken != policy_id:
            aliases.append(spoken)
        return [{"name": "policy_id", "type": "string", "allowed": [policy_id],
                 "aliases": {policy_id: aliases}, "number_hints": []}]

    def chat_request(self, prompt, data, config, parameters=None):
        """A question plus one observation becomes a typed request. The prompt selects nothing it could get wrong: this
        provider has exactly one supported question, so a prose instruction cannot widen what it does.

        `parameters` carries what the workbench resolved from the person's words. A policy named there that is not this
        bundle's is refused BY NAME: serving the only policy available under a name nobody asked for would answer a
        different question, and the person would have no way to tell."""
        manifest = self.manifest
        if manifest is None:
            raise PolicyRefusal("POLICY_BUNDLE_UNAVAILABLE: no operator-declared bundle is configured")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise PolicyRefusal("PARAMETERS_MUST_BE_A_MAPPING: resolved parameters arrive as a name/value mapping")
            unknown = sorted(set(parameters) - {"policy_id"})
            if unknown:
                raise PolicyRefusal(f"UNDECLARED_PARAMETER: this provider declares policy_id and was given {unknown}")
            named = parameters.get("policy_id")
            if named is not None and named != manifest["policy_id"]:
                raise PolicyRefusal(
                    f"UNKNOWN_POLICY: {named!r} is not this bundle's fitted policy, which is "
                    f"{manifest['policy_id']!r}; the one policy available is not a substitute for the one named")
        observation, build = self._observation_from(data)
        state = config.get("state") or state_ref_for(manifest)
        inputs = {"observation": observation, "question": prompt}
        if build is not None:
            inputs["observation_build"] = build
        return {"schema_version": "m5phet.task.draft2",
                "request_id": "chat:" + digest([prompt, observation])[:32],
                "task_id": manifest["policy_id"],
                "operation": "infer", "family": FAMILY, "output_kind": OUTPUT_KIND,
                "as_of": config.get("as_of"),
                "provider_ref": NAME,
                "fitted_state_ref": state,
                "output_schema": {"targets": ["action"]},
                "inputs": inputs,
                "execution_constraints": {"partial_results": False}}

    def _observation_from(self, data):
        """Resolve what was attached into (observation, how it was built).

        Two inputs are accepted and they are told apart by SHAPE, never by the prompt. A list of numbers is the
        observation itself and takes the path it has always taken -- nothing about it changed, and it works with no
        gym-fx installed. A table of named rows, which is what an attached CSV becomes once the workbench has parsed
        it, is market data and is built into an observation by the environment's own code.

        Routing on shape matters for the refusals. A table sent down the vector path would come back as "not a list of
        numbers", and the person would go looking for a formatting mistake instead of reading that their file is 40
        rows short of the window the policy needs."""
        market_data = None
        supplied_state = None
        observation = data

        if isinstance(data, dict):
            if "observation" in data:
                observation = data.get("observation")
            elif "rows" in data:
                market_data, supplied_state = data.get("rows"), data.get("agent_state")
                observation = None
            else:
                observation = None
        elif market.looks_like_rows(data):
            if self._rows_carry_the_fitted_columns(data):
                market_data = data
                observation = None
            elif len(data) == 1:
                # Not this policy's market data. A SINGLE row may still be a carrier for a vector, which is how a
                # one-line CSV of numbers arrives, so that reading is kept. It stops at one row: reading the first row
                # of a many-row table as the whole observation is the helpful guess this provider exists to avoid, and
                # a table with many rows is market data by every reading except that one.
                row = data[0]
                try:
                    observation = (json.loads(next(iter(row.values()))) if len(row) == 1
                                   else [float(v) for v in row.values()])
                except (TypeError, ValueError):
                    observation = None
                if not isinstance(observation, list):
                    raise PolicyRefusal(self._unusable_table_refusal(data))
            else:
                raise PolicyRefusal(self._unusable_table_refusal(data))
        elif isinstance(data, str):
            try:
                observation = json.loads(data)
            except ValueError:
                # prose is not an observation, and guessing numbers out of it would invent the input
                observation = None
            if market.looks_like_rows(observation):
                market_data, observation = observation, None
            elif observation is None and market.looks_like_csv_text(data):
                market_data = data

        if market_data is not None:
            payload = ({"rows": market_data, "agent_state": supplied_state}
                       if supplied_state is not None else market_data)
            return market.build_from_rows(payload, self._required_contract(), self.manifest, self.gym_fx_root)

        if not isinstance(observation, list):
            raise PolicyRefusal("OBSERVATION_REQUIRED: attach a JSON list of numbers, or a CSV row of numbers, as the "
                                "observation this policy was fitted on -- or a table of market data with the columns "
                                "this policy was fitted on, and it will be built for you")
        return observation, None

    def _required_contract(self):
        declared = self.observation_contract
        if declared is None:
            readiness = self.market_data_readiness()
            raise PolicyRefusal("OBSERVATION_CONTRACT_UNAVAILABLE: " + readiness["why"])
        return declared

    def _rows_carry_the_fitted_columns(self, rows):
        """True when the table's columns are the ones this policy was fitted on.

        Read from the contract alone, with nothing imported: the question is whether this looks like an attempt to
        supply market data, and that has to be answerable before deciding which refusal the caller deserves."""
        declared = self.observation_contract
        if declared is None:
            return False
        wanted = set(declared["environment"].get("feature_columns") or ())
        return bool(wanted) and wanted.issubset(set(rows[0]))

    def _unusable_table_refusal(self, rows):
        readiness = self.market_data_readiness()
        if readiness.get("supported"):
            declared = self.observation_contract["environment"].get("feature_columns") or []
            missing = [name for name in declared if name not in set(rows[0])]
            return (f"MISSING_COLUMNS: this table has {len(rows[0])} columns and is missing {len(missing)} of the "
                    f"{len(declared)} this policy was fitted on, the first being {missing[:5]}")
        return ("OBSERVATION_REQUIRED: this looks like a table of market data, and this provider cannot build an "
                "observation from market data here: " + readiness["why"])

    def chat_examples(self):
        manifest = self.manifest
        if manifest is None:
            return []
        size = manifest["observation_size"]
        examples = [{"title": "DEVELOPMENT: fitted policy, one observation, proposed action",
                     "prompt": "What action does this fitted policy propose for this observation?",
                     "data": json.dumps([0.0] * size),
                     "config": {"input": "json", "provider": NAME, "family": FAMILY, "output_kind": OUTPUT_KIND,
                                "state": state_ref_for(manifest)}}]
        readiness = self.market_data_readiness()
        if readiness.get("supported"):
            # Deliberately not a pasteable vector: the example a person can act on is the one that says what FILE to
            # attach, since the market-data path exists precisely because nobody can type the other one.
            examples.append({
                "title": (f"DEVELOPMENT: market data in, proposed action out "
                          f"({readiness['required_rows']} rows of "
                          f"{readiness['feature_count']} fitted columns)"),
                "prompt": "What action does this fitted policy propose for these bars?",
                "data": (f"attach a CSV whose last {readiness['required_rows']} rows end at the bar you are asking "
                         f"about, carrying the {readiness['feature_count']} feature columns this policy was fitted on "
                         f"plus {readiness['price_column']}. Fewer rows is refused, not padded"),
                "config": {"input": "csv", "provider": NAME, "family": FAMILY, "output_kind": OUTPUT_KIND,
                           "state": state_ref_for(manifest)}})
        return examples
