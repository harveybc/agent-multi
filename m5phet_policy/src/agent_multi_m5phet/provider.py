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

Under the unified question envelope (`m5phet.questions`) this provider is the `rl` area and answers two question types
from the same fitted checkpoint, through the same path: `next_action`, the actor's action for the observation in the
state, and `value_estimation`, the twin critics' estimate of discounted return at that action. Neither carries a number
the checkpoint does not hold. The actor's Gaussian is reported as the actor's distribution, not as a probability of being
right; the critic's value is reported as the critic's estimate under the training reward, not as a realised profit; and
where the engine returns neither, the field is absent with its reason, or the question is refused as NOT_ESTIMABLE.
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

try:                                                                   # the envelope is optional for the older paths
    from m5phet import questions as envelope
except ImportError:                                                    # pragma: no cover - exercised only without m5phet
    envelope = None

NAME = "trading_policy"
FAMILY = "policy"
OUTPUT_KIND = "policy_action"
SUPPORTED = ({"operation": "infer", "family": FAMILY, "output_kind": OUTPUT_KIND},)

#: a deterministic policy returns one action; it carries no distribution, and calling that "uncertainty" would invent one
UNCERTAINTY = "NONE_DETERMINISTIC_POLICY"

DEFAULT_TIMEOUT_SECONDS = 120

#: the question types this provider answers under `m5phet.questions`, and what the worker must evaluate for each
AREA = "rl"
QUESTION_TYPES = {"next_action": {"required": [], "optional": []},
                  "value_estimation": {"required": [], "optional": []}}
_EVALUATIONS_FOR = {"next_action": ("actor_distribution",), "value_estimation": ("critic",)}

CONFIDENCE_ABSENT = ("not emitted: a SAC actor returns one action and a Gaussian over its pre-squash value. The Gaussian's "
                     "spread is reported under action_distribution; it describes the actor, and no number here says how "
                     "likely the action is to be right")
DISTRIBUTION_READING = ("the actor's own distribution for this observation: a Gaussian over the pre-squash action, with "
                        "the reported action at tanh(mean). It is the policy's stochastic spread, NOT a probability of "
                        "the action being right and NOT a distribution over outcomes")
VALUE_READING = ("the critic's estimate of discounted return under the training reward, not a realised profit. "
                 "expected_return is the minimum of the twin Q critics at the actor's action, as SAC itself uses it; "
                 "uncertainty_bounds are the two critics' minimum and maximum, which is their disagreement and nothing "
                 "more")


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
        evaluate = [name for name in (inputs.get("evaluate") or []) if isinstance(name, str)]
        call = {"checkpoint": manifest["checkpoint"], "observation": [float(v) for v in observation],
                "deterministic": True}
        if evaluate:
            call["evaluate"] = evaluate
        answer = self._runner(call)
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
        # What was asked of the engine beyond the action travels back only if the engine returned it. A missing reading
        # is reported as missing by whoever asked for it; nothing is synthesised from the action to fill the gap.
        for name in evaluate:
            if isinstance(answer.get(name), dict):
                payload[name] = copy.deepcopy(answer[name])
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

    def chat_request(self, prompt, data, config, parameters=None, spec=None):
        """A question plus one observation becomes a typed request. The prompt selects nothing it could get wrong: this
        provider has exactly one supported question, so a prose instruction cannot widen what it does.

        `parameters` carries what the workbench resolved from the person's words. A policy named there that is not this
        bundle's is refused BY NAME: serving the only policy available under a name nobody asked for would answer a
        different question, and the person would have no way to tell.

        `spec` is optional and only reaches the market-data path: a `m5phet.representation.v1` object saying which
        representation the attached bars are to be read under. Absent, the bars are read under the fitted policy's own
        contract and nothing about this call changes. Present, the spec must reproduce that contract or the request is
        refused `OBSERVATION_CONTRACT_MISMATCH` -- a policy answers only the representation it was fitted on. It may
        also arrive inside the attachment itself, as `{"rows": ..., "spec": ...}`, which is how a question envelope
        carries it."""
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
        observation, build = self._observation_from(data, spec)
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

    def _observation_from(self, data, spec=None):
        """Resolve what was attached into (observation, how it was built).

        Two inputs are accepted and they are told apart by SHAPE, never by the prompt. A list of numbers is the
        observation itself and takes the path it has always taken -- nothing about it changed, and it works with no
        gym-fx installed. A table of named rows, which is what an attached CSV becomes once the workbench has parsed
        it, is market data and is built into an observation by the environment's own code.

        Routing on shape matters for the refusals. A table sent down the vector path would come back as "not a list of
        numbers", and the person would go looking for a formatting mistake instead of reading that their file is 40
        rows short of the window the policy needs.

        A representation spec, when one is supplied, changes nothing about that routing: it changes which contract the
        market data is read under, and only after that contract has been proved to be this policy's own."""
        market_data = None
        supplied_state = None
        observation = data

        if isinstance(data, dict):
            if "observation" in data:
                observation = data.get("observation")
            elif "rows" in data:
                market_data, supplied_state = data.get("rows"), data.get("agent_state")
                spec = data.get("spec", spec)
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
            if spec is not None:
                return market.observation_from_spec(payload, spec, self._required_contract(), self.manifest,
                                                   self.gym_fx_root)
            return market.build_from_rows(payload, self._required_contract(), self.manifest, self.gym_fx_root)

        if spec is not None:
            # A vector was supplied, not rows. There is nothing to read under a representation, and accepting the spec
            # would let a receipt say the observation was built under it when it was typed by hand.
            raise PolicyRefusal(
                "REPRESENTATION_SPEC_NEEDS_MARKET_DATA: a representation says how to READ rows into an observation, "
                "and what was supplied is the observation itself; attach the bars, or drop the spec")

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

    # --- the question envelope: `m5phet.questions`, area "rl" --------------------------------------------------------
    area = AREA

    def question_types(self):
        return {name: {"required": list(spec["required"]), "optional": list(spec["optional"])}
                for name, spec in QUESTION_TYPES.items()}

    def answer_questions(self, state, questions, data, as_of):
        """Answer named questions about one state, each on its own, from the same engine call.

        The state carries `current_observation` -- a raw vector, a table of the fitted columns, or CSV text -- and may
        name `policy_id`. Both go down `chat_request` exactly as a chat does: a policy named that is not this bundle's is
        refused by name, and an observation of the wrong length is refused rather than padded. One subprocess call then
        evaluates what the questions need, and each answer takes only its own part of what came back."""
        if envelope is None:
            raise PolicyRefusal("QUESTION_ENVELOPE_UNAVAILABLE: m5phet.questions is not importable here")
        state = state if isinstance(state, dict) else {}
        observation = state.get("current_observation", data)
        parameters = {"policy_id": state["policy_id"]} if state.get("policy_id") is not None else None
        # `spec` is the state's optional declaration of WHICH representation the supplied rows are; absent, the
        # fitted policy's own contract reads them, exactly as before.
        spec = state.get("spec")
        instructions = next((q.get("instructions") for q in questions.values()
                             if isinstance(q.get("instructions"), str)), "") or ""
        try:
            if observation is None:
                raise PolicyRefusal("OBSERVATION_REQUIRED: the state names no current_observation; supply the vector "
                                    "this policy was fitted on, or a table of the fitted columns ending at the bar "
                                    "being asked about")
            request = self.chat_request(instructions, observation, {"as_of": as_of, "state": ""}, parameters,
                                        spec=spec)
            fitted = self.load(request["fitted_state_ref"])
        except PolicyRefusal as refusal:
            return {name: envelope.refusal(envelope.STATE_REQUIRED, str(refusal), q["type"])
                    for name, q in questions.items()}
        wanted = sorted({e for q in questions.values() for e in _EVALUATIONS_FOR.get(q["type"], ())})
        request["inputs"]["evaluate"] = wanted
        try:
            result = self.infer(request, fitted)
        except PolicyRefusal as refusal:
            return {name: envelope.refusal(envelope.PROVIDER_ERROR, str(refusal), q["type"])
                    for name, q in questions.items()}
        output = result["outputs"]["action"]
        if output.get("status") != "OK":
            kind = envelope.STATE_REQUIRED if output.get("status") == "INVALID_INPUT" else envelope.PROVIDER_ERROR
            return {name: envelope.refusal(kind, output.get("why", output.get("status")), q["type"])
                    for name, q in questions.items()}
        payload = output["payload"]
        out = {"__state_ref__": fitted["state_ref"]}
        for name, question in questions.items():
            out[name] = self._answer_one(question["type"], payload)
        return out

    def _identity(self, payload):
        return {"policy_id": payload["policy_id"], "execution_authorized": False,
                "observation_source": payload["observation_source"],
                "observation_build": copy.deepcopy(payload["observation_build"])}

    def _answer_one(self, kind, payload):
        if kind == "next_action":
            answer = {"type": kind, "action": list(payload["action"]), "unit": payload["unit"],
                      "action_space": payload["action_space"], "deterministic": True,
                      "out_of_declared_range": list(payload["out_of_declared_range"]),
                      "confidence": None, "confidence_absent_because": CONFIDENCE_ABSENT,
                      "reading": payload["reading"], **self._identity(payload)}
            distribution = payload.get("actor_distribution")
            if isinstance(distribution, dict):
                answer["action_distribution"] = {**copy.deepcopy(distribution), "reading": DISTRIBUTION_READING}
            else:
                answer["action_distribution"] = None
                answer["action_distribution_absent_because"] = ("the engine returned the action and no actor "
                                                                "distribution; none is derived from one action")
            return answer
        if kind == "value_estimation":
            critic = payload.get("critic")
            values = critic.get("q_values") if isinstance(critic, dict) else None
            if not isinstance(values, list) or not values or \
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
                return envelope.refusal(envelope.NOT_ESTIMABLE,
                                        "the engine returned no critic evaluation for this observation; a return is "
                                        "estimated from the checkpoint's Q critics or not at all", kind)
            values = [float(v) for v in values]
            return {"type": kind, "expected_return": min(values), "uncertainty_bounds": [min(values), max(values)],
                    "critic_values": values, "n_critics": critic.get("n_critics", len(values)),
                    "evaluated_at_action": list(payload["action"]), "discount_gamma": critic.get("gamma"),
                    "estimator": "min_of_twin_q_critics", "reading": VALUE_READING, **self._identity(payload)}
        return envelope.refusal(envelope.UNSUPPORTED_QUESTION_TYPE, f"this provider does not answer {kind!r}", kind)

    def chat_examples(self):
        manifest = self.manifest
        if manifest is None:
            return []
        size = manifest["observation_size"]
        # Two examples, two inputs, two actions from the same policy (Retsu measured -0.0200 here and 0.0591 on the
        # bars, 2026-09-24). The titles say so, because a person who pastes one and a person who pastes the other are
        # not asking the same thing and must not read the answers as one number.
        examples = [{"title": "DEVELOPMENT: an all-zero observation vector; its action is this input's, not the "
                              "bars example's",
                     "prompt": "What action does this fitted policy propose for this observation?",
                     "reading": "a synthetic observation of zeros; the same policy answers the market-data example "
                                "with a different action because the input differs",
                     "data": json.dumps([0.0] * size),
                     "config": {"input": "json", "provider": NAME, "family": FAMILY, "output_kind": OUTPUT_KIND,
                                "state": state_ref_for(manifest)}}]
        readiness = self.market_data_readiness()
        if readiness.get("supported"):
            # An example is something a person clicks and runs. Describing what to attach is documentation, and shipping
            # it AS the example means the first thing anyone tries is refused -- for the very reason the example exists.
            # So this one carries real bars when the operator's sample is reachable, and is simply absent when it is not.
            bars = self._example_bars(readiness)
            if bars:
                examples.append({
                    "title": (f"DEVELOPMENT: market data in, proposed action out "
                              f"({readiness['required_rows']} rows of {readiness['feature_count']} fitted columns); "
                              f"a different input from the zero-vector example, so a different action"),
                    "prompt": "What action does this fitted policy propose for these bars?",
                    "reading": "the observation is built from these bars by gym-fx's own construction; the action is "
                               "a target position fraction of the policy's action scale, not an order",
                    "data": bars,
                    "config": {"input": "csv", "provider": NAME, "family": FAMILY, "output_kind": OUTPUT_KIND,
                               "state": state_ref_for(manifest)}})
        return examples

    def _example_bars(self, readiness):
        """The tail of the operator's declared sample, as CSV text, or nothing.

        `M5PHET_POLICY_SAMPLE` names a file of bars carrying the fitted columns. It is the operator's declaration, never
        a path from a request, and its absence removes the example rather than producing one that cannot run."""
        import csv
        import io
        import os

        sample = os.environ.get("M5PHET_POLICY_SAMPLE")
        if not sample or not Path(sample).is_file():
            return None
        needed = int(readiness["required_rows"])
        try:
            with Path(sample).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        except OSError:
            return None
        if len(rows) < needed + 1:
            return None
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(rows[0])
        writer.writerows(rows[-needed:])
        return buffer.getvalue()
