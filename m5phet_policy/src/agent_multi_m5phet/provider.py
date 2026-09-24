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
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

NAME = "trading_policy"
FAMILY = "policy"
OUTPUT_KIND = "policy_action"
SUPPORTED = ({"operation": "infer", "family": FAMILY, "output_kind": OUTPUT_KIND},)

#: a deterministic policy returns one action; it carries no distribution, and calling that "uncertainty" would invent one
UNCERTAINTY = "NONE_DETERMINISTIC_POLICY"

DEFAULT_TIMEOUT_SECONDS = 120


class PolicyRefusal(ValueError):
    """A refusal that names itself. Nothing here substitutes a default action for a refusal."""


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
                "reading": ("this provider proposes the action an ALREADY FITTED policy returns for one observation. It is "
                            "not a recommendation, not a position size and not an order; nothing here can authorize one")}

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
        payload = {"action": [float(v) for v in action],
                   "unit": manifest["unit"],
                   "action_space": manifest["action_space"],
                   "policy_id": manifest["policy_id"],
                   "deterministic": True,
                   "out_of_declared_range": clipped,
                   "execution_authorized": False,
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
        observation = data
        if isinstance(data, dict):
            observation = data.get("observation")
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            row = data[0]
            try:
                observation = (json.loads(next(iter(row.values()))) if len(row) == 1
                               else [float(v) for v in row.values()])
            except (TypeError, ValueError):
                observation = None
        if isinstance(observation, str):
            try:
                observation = json.loads(observation)
            except ValueError:
                # prose is not an observation, and guessing numbers out of it would invent the input
                observation = None
        if not isinstance(observation, list):
            raise PolicyRefusal("OBSERVATION_REQUIRED: attach a JSON list of numbers, or a CSV row of numbers, as the "
                                "observation this policy was fitted on")
        state = config.get("state") or state_ref_for(manifest)
        return {"schema_version": "m5phet.task.draft2",
                "request_id": "chat:" + digest([prompt, observation])[:32],
                "task_id": manifest["policy_id"],
                "operation": "infer", "family": FAMILY, "output_kind": OUTPUT_KIND,
                "as_of": config.get("as_of"),
                "provider_ref": NAME,
                "fitted_state_ref": state,
                "output_schema": {"targets": ["action"]},
                "inputs": {"observation": observation, "question": prompt},
                "execution_constraints": {"partial_results": False}}

    def chat_examples(self):
        manifest = self.manifest
        if manifest is None:
            return []
        size = manifest["observation_size"]
        return [{"title": "DEVELOPMENT: fitted policy, one observation, proposed action",
                 "prompt": "What action does this fitted policy propose for this observation?",
                 "data": json.dumps([0.0] * size),
                 "config": {"input": "json", "provider": NAME, "family": FAMILY, "output_kind": OUTPUT_KIND,
                            "state": state_ref_for(manifest)}}]
