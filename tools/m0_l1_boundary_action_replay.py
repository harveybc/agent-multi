#!/usr/bin/env python3
"""Same-artifact threshold replay diagnostic (WP2, order 2026-08-11 §6).

QUESTION. At the easy->normal boundary the ladder switches the action
threshold from 0.0 to 0.1. Does that discontinuity merely EXPOSE a
policy that has already collapsed (near-constant output), or does it
actively CAUSE the mapping collapse of a policy whose output still
carries real variation — all of it inside the new deadband?

CANONICAL OBSERVATION STREAM — FLAT-HOLD REPLAY. For each split the
env is built exactly as the pipeline builds a split evaluation env
(``PipelinePlugin._split_csv`` + ``_make_split_env`` with the agent's
``wrap_env``, ``solvency_mode`` forced to ``normal_realistic`` exactly
as ``_eval_on_split`` forces it), from the D0 arm config materialized
by ``tools.m0_l1_mechanism_ladder.materialize_arm_config`` — so the
train-tail and inner-validation windows are hash-identical to the
ladder's. The env is then advanced by stepping a FORCED HOLD action
(``zeros(action_space.shape)``) every step: gym-fx coerces an exact
zero to HOLD under every threshold in [0, 1), so no position ever
opens and agent-state stays flat. At each step, BEFORE stepping, the
loaded SAC policy runs deterministically on the current observation
and the raw action is recorded. The observation stream is therefore a
pure function of (data slice, config, seed) — identical for both
artifacts — and each artifact yields exactly one raw action vector
per split. Thresholds 0.0 and 0.1 are applied ANALYTICALLY to that
same vector (no second rollout); the emitted action-vector sha256
proves both threshold arms consumed identical input.

THRESHOLD MAPPING (gym-fx ``GymFxEnv._coerce_action`` parity):
  threshold == 0.0:  a > 0 -> long,  a < 0 -> short,  a == 0 -> hold
  threshold  > 0.0:  a >= t -> long, a <= -t -> short, else hold

NEAR-UNIQUE TOLERANCE (declared, derived from dtype precision):
  tolerance = numpy.finfo(numpy.float32).eps * max(1.0, max(|a|))
i.e. one float32 unit-in-the-last-place at the observed action scale
(floored at scale 1.0, the action-space bound). Two actions closer
than this are indistinguishable at float32 precision. Near-unique
count clusters the sorted raw actions: a new cluster starts whenever
the gap to the previous value exceeds the tolerance.

OUTCOME RULE (derived per split from the emitted facts, then joined):
  crossings           := count(|a| >= 0.1)          (long+short at 0.1)
  variation_retained  := (max(a) - min(a)) > tolerance
  1. crossings > 0                  -> NO_THRESHOLD_COLLAPSE
     (some deterministic actions survive the 0.1 deadband; this holds
     even for a constant policy sitting above the threshold — the
     threshold is not what silences it).
  2. else if variation_retained     ->
     THRESHOLD_CAUSES_MAPPING_COLLAPSE_WITH_VARIATION_RETAINED
     (real float32-distinguishable variation exists, yet every action
     lies inside the 0.1 deadband: all HOLD at 0.1 while threshold
     0.0 maps non-hold — the discontinuity, not the policy, erases
     the mapping).
  3. else                           -> THRESHOLD_EXPOSES_PREEXISTING_COLLAPSE
     (the raw output is already near-constant at dtype precision and
     below the deadband; the threshold only exposes the collapse).
Overall: both splits must agree; if they disagree, or any split is
empty/capped, or any pinned input hash mismatches, the result is
REPLAY_INCONCLUSIVE with the exact refusals listed.

DISCIPLINE. No learning, no selection, read-only on every input; the
sealed collection root is never written. Diagnosis only — never
decision-eligible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools import m0_l1_mechanism_ladder as ladder  # noqa: E402
from tools.l1_factorial_screen import atomic_write_json  # noqa: E402

OUTPUT_SCHEMA = "agent_multi.m0_l1_boundary_action_replay.v1"

# The ladder contract is the ONLY data/config source (WP2). The pinned
# sha is enforced unless --contract explicitly overrides (tests only).
PINNED_CONTRACT_SHA256 = (
    "c26fbed45581c2a2322162f30497b1be32fd631067953c4da5d618cf4ed4e604")

# Subject artifacts — verified fail-closed before any model load.
ARTIFACTS = {
    "d2_post_easy": {
        "path": ("/home/harveybc/.local/share/agent-multi/"
                 "ladder_collection_97c0bb29_20260811/sealed/"
                 "97c0bb29e82dfea3/D2_BOUNDARY_ONLY/"
                 "attempt-8266c6c8ac0bfb12-01/model.post_easy.zip"),
        "sha256": ("2620b7227f201560da7a017eb479350a1e92dc9148c804f6f"
                   "4acf37efaedbc26"),
        "role": ("D2_BOUNDARY_ONLY post-easy artifact (sealed ladder "
                 "copy): the trained phase-1 policy handed across the "
                 "boundary"),
    },
    "anchor": {
        "path": ("~/.local/share/agent-multi/"
                 "eth_curriculum_decision_20260807_v2/seed101/"
                 "anchor_seed101.zip"),
        "sha256": ("cb27375c663819333aa4442d5817c657b19ad0e8b61395b04"
                   "c5e6212217fbc62"),
        "role": "D0 anchor control: the mature source anchor, seed 101",
    },
}

THRESHOLD_EASY = 0.0
THRESHOLD_NORMAL = 0.1
ABS_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)
MAX_REPLAY_STEPS = 1_000_000
LADDER_ARM = "D0_M0_EXACT"

# Output label -> the pipeline _split_csv key it replays.
REPLAY_SPLITS = (("train_tail", "train_tail"), ("inner_validation", "val"))

OUTCOME_EXPOSES = "THRESHOLD_EXPOSES_PREEXISTING_COLLAPSE"
OUTCOME_CAUSES = (
    "THRESHOLD_CAUSES_MAPPING_COLLAPSE_WITH_VARIATION_RETAINED")
OUTCOME_NONE = "NO_THRESHOLD_COLLAPSE"
OUTCOME_INCONCLUSIVE = "REPLAY_INCONCLUSIVE"

TOLERANCE_FORMULA = (
    "numpy.finfo(numpy.float32).eps * max(1.0, max(abs(raw_actions)))")
OBS_SERIALIZATION = (
    "sha256(b'obs_manifest.v1|shape=<shape>|dtype=float32|' + "
    "concat(per-step observation as C-order float32 bytes) + "
    "b'|count=<steps>')")
ACTION_SERIALIZATION = (
    "sha256(b'action_vector.v1|dtype=float32|count=<n>|' + "
    "raw action vector as C-order float32 bytes)")

EXIT_CLASS = {
    OUTCOME_EXPOSES: 0,
    OUTCOME_CAUSES: 0,
    OUTCOME_NONE: 0,
    OUTCOME_INCONCLUSIVE: 4,
}


# ---------------------------------------------------------------------------
# pure, socket-free primitives (unit-tested)
# ---------------------------------------------------------------------------

def verify_sha256(path: Path, expected: str, label: str) -> str | None:
    """Return an exact refusal string on mismatch/absence, else None."""
    path = Path(path)
    if not path.is_file():
        return f"{label}: file missing at {path}"
    observed = sysid.sha_file(path)
    if observed != expected:
        return (f"{label}: sha256 mismatch at {path} — expected "
                f"{expected}, observed {observed}")
    return None


def flat_hold_action(action_space) -> np.ndarray:
    """The forced HOLD step input: zeros over the continuous Box.

    gym-fx coerces an exact zero to HOLD under every legal threshold
    (threshold 0.0 maps only strictly nonzero values to a side), so a
    zero action can never open a position.
    """
    shape = getattr(action_space, "shape", None)
    if not shape:
        raise ValueError(
            "flat-hold replay requires a continuous Box action space "
            f"with a nonempty shape; got {action_space!r}")
    dtype = getattr(action_space, "dtype", None) or np.float32
    return np.zeros(tuple(shape), dtype=dtype)


def flat_hold_replay(env, predict_raw, *, seed: int,
                     max_steps: int = MAX_REPLAY_STEPS) -> dict:
    """Advance ``env`` on forced-hold actions; record the policy's raw
    deterministic action on each pre-step observation.

    ``predict_raw(obs)`` must return the raw continuous action (any
    array-like; the first element is the scalar action, exactly as the
    pipeline's ``_raw_action_value`` reads it).
    """
    forced = flat_hold_action(env.action_space)
    obs, _info = env.reset(seed=seed)
    hasher = hashlib.sha256()
    obs_shape: tuple | None = None
    raws: list[np.float32] = []
    terminated = truncated = capped = False
    steps = 0
    while True:
        arr = np.ascontiguousarray(np.asarray(obs, dtype=np.float32))
        if obs_shape is None:
            obs_shape = arr.shape
            hasher.update(
                f"obs_manifest.v1|shape={obs_shape}|dtype=float32|"
                .encode())
        elif arr.shape != obs_shape:
            raise ValueError(
                f"observation shape drifted mid-episode: {arr.shape} "
                f"!= {obs_shape}")
        hasher.update(arr.tobytes())
        raw = predict_raw(obs)
        raws.append(np.float32(np.asarray(raw).reshape(-1)[0]))
        obs, _reward, terminated, truncated, _info = env.step(
            forced.copy())
        steps += 1
        if terminated or truncated:
            break
        if steps >= max_steps:
            capped = True
            break
    hasher.update(f"|count={steps}".encode())
    return {
        "observation_count": steps,
        "observation_shape": list(obs_shape or ()),
        "observation_manifest_sha256": hasher.hexdigest(),
        "raw_actions": np.asarray(raws, dtype=np.float32),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps_capped": bool(capped),
    }


def action_vector_sha256(raw: np.ndarray) -> str:
    vec = np.ascontiguousarray(np.asarray(raw, dtype=np.float32)
                               .reshape(-1))
    hasher = hashlib.sha256()
    hasher.update(
        f"action_vector.v1|dtype=float32|count={vec.size}|".encode())
    hasher.update(vec.tobytes())
    return hasher.hexdigest()


def near_unique_tolerance(raw: np.ndarray) -> dict:
    """Declared dtype-precision tolerance (formula in the output)."""
    vec = np.asarray(raw, dtype=np.float64).reshape(-1)
    eps = float(np.finfo(np.float32).eps)
    scale = float(np.max(np.abs(vec))) if vec.size else 0.0
    return {
        "formula": TOLERANCE_FORMULA,
        "float32_eps": eps,
        "observed_scale_max_abs": scale,
        "value": eps * max(1.0, scale),
    }


def near_unique_count(raw: np.ndarray, tolerance: float) -> int:
    """Cluster count of sorted raw actions with gap > tolerance."""
    vec = np.sort(np.asarray(raw, dtype=np.float64).reshape(-1))
    if vec.size == 0:
        return 0
    return 1 + int(np.count_nonzero(np.diff(vec) > float(tolerance)))


def map_actions(raw: np.ndarray, threshold: float) -> dict:
    """Long/short/hold counts under gym-fx ``_coerce_action`` semantics."""
    vec = np.asarray(raw, dtype=np.float64).reshape(-1)
    thr = float(threshold)
    if thr == 0.0:
        long_n = int(np.count_nonzero(vec > 0.0))
        short_n = int(np.count_nonzero(vec < 0.0))
    else:
        long_n = int(np.count_nonzero(vec >= thr))
        short_n = int(np.count_nonzero(vec <= -thr))
    return {
        "threshold": thr,
        "long": long_n,
        "short": short_n,
        "hold": int(vec.size - long_n - short_n),
        "non_hold": long_n + short_n,
    }


def split_facts(raw: np.ndarray, *, observation_count: int,
                observation_manifest_sha256: str) -> dict:
    """Every emitted per-split fact the outcome rule derives from."""
    vec = np.asarray(raw, dtype=np.float64).reshape(-1)
    count = int(vec.size)
    tol = near_unique_tolerance(raw)
    abs_vec = np.abs(vec)
    if count:
        quantiles = {
            f"q{q:.2f}": float(np.quantile(abs_vec, q))
            for q in ABS_QUANTILES
        }
        q25, q75 = (float(np.quantile(vec, 0.25)),
                    float(np.quantile(vec, 0.75)))
        stats = {
            "min": float(vec.min()),
            "max": float(vec.max()),
            "mean": float(vec.mean()),
            "std": float(vec.std()),
            "interquartile_range": q75 - q25,
            "q25": q25,
            "q75": q75,
        }
    else:
        quantiles = {}
        stats = {"min": None, "max": None, "mean": None, "std": None,
                 "interquartile_range": None, "q25": None, "q75": None}
    count_gt_zero = int(np.count_nonzero(abs_vec > 0.0))
    count_ge_normal = int(
        np.count_nonzero(abs_vec >= THRESHOLD_NORMAL))
    variation_retained = bool(
        count and (stats["max"] - stats["min"]) > tol["value"])
    return {
        "observation_count": int(observation_count),
        "observation_manifest_sha256": observation_manifest_sha256,
        "raw_actions": {
            **stats,
            "count": count,
            "abs_quantiles": quantiles,
            "abs_quantiles_declared": list(ABS_QUANTILES),
            "unique_count_exact": int(np.unique(vec).size),
            "near_unique_count": near_unique_count(raw, tol["value"]),
            "near_unique_tolerance": tol,
        },
        "count_abs_gt_zero": count_gt_zero,
        "fraction_abs_gt_zero": (count_gt_zero / count) if count else None,
        "count_abs_ge_normal_threshold": count_ge_normal,
        "fraction_abs_ge_normal_threshold": (
            (count_ge_normal / count) if count else None),
        "action_vector_sha256": action_vector_sha256(raw),
        "mapped_actions": {
            "threshold_0.0": map_actions(raw, THRESHOLD_EASY),
            "threshold_0.1": map_actions(raw, THRESHOLD_NORMAL),
        },
        "variation_retained": variation_retained,
    }


def classify_split(facts: dict) -> str:
    """The documented outcome rule, derived only from emitted facts."""
    if int(facts["count_abs_ge_normal_threshold"]) > 0:
        return OUTCOME_NONE
    if bool(facts["variation_retained"]):
        return OUTCOME_CAUSES
    return OUTCOME_EXPOSES


def classify_overall(split_outcomes: dict) -> tuple[str, list]:
    """Join per-split outcomes fail-closed (both must agree)."""
    if not split_outcomes:
        return OUTCOME_INCONCLUSIVE, ["no split produced a replay"]
    distinct = sorted(set(split_outcomes.values()))
    if len(distinct) == 1:
        return distinct[0], []
    detail = ", ".join(f"{name}={outcome}"
                       for name, outcome in sorted(split_outcomes.items()))
    return OUTCOME_INCONCLUSIVE, [f"split outcomes disagree: {detail}"]


def refusal_result(refusals: list, **context) -> dict:
    """Typed inconclusive payload carrying the exact refusals."""
    return {
        "outcome": OUTCOME_INCONCLUSIVE,
        "refusals": list(refusals),
        **context,
    }


# ---------------------------------------------------------------------------
# the real replay (heavy imports stay lazy; unit tests never enter here)
# ---------------------------------------------------------------------------

def _torch_facts() -> dict:
    facts = {"torch_cuda_available": None, "torch_cuda_device_count": None,
             "torch_cuda_device_name": None}
    try:
        import torch
        facts["torch_cuda_available"] = bool(torch.cuda.is_available())
        facts["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            facts["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return facts


def _csv_row_count(path: Path) -> int:
    with Path(path).open("rb") as handle:
        rows = sum(1 for _ in handle)
    return max(0, rows - 1)  # header


def run_replay(artifact_key: str, *, device: str,
               contract_path: Path | None = None) -> dict:
    """One artifact, both splits, one payload. Read-only on inputs."""
    import tempfile

    started = datetime.now(timezone.utc)
    spec = ARTIFACTS[artifact_key]
    artifact_path = Path(spec["path"]).expanduser()
    contract_pinned = contract_path is None
    resolved_contract = (Path(contract_path) if contract_path
                         else ladder.CONTRACT_PATH)

    payload: dict = {
        "schema": OUTPUT_SCHEMA,
        "tool": "tools/m0_l1_boundary_action_replay.py",
        "question": ("does the normal-phase action threshold 0.1 expose "
                     "an already-collapsed policy or cause mapping "
                     "collapse while variation is retained?"),
        "generated_utc": started.isoformat(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_requested": device,
        "torch_facts": _torch_facts(),
        "artifact": {
            "key": artifact_key,
            "path": str(artifact_path),
            "role": spec["role"],
            "sha256_expected": spec["sha256"],
            "sha256_observed": None,
        },
        "contract": {
            "path": str(resolved_contract),
            "sha256_pinned": PINNED_CONTRACT_SHA256,
            "sha256_observed": None,
            "pinned": contract_pinned,
        },
        "code_identity": None,
        "replay_design": {
            "canonical_observation_stream": (
                "flat-hold replay: env advanced on forced zero actions "
                "(HOLD under every threshold); policy queried "
                "deterministically on each pre-step observation"),
            "arm_config": LADDER_ARM,
            "eval_solvency_mode": "normal_realistic",
            "thresholds_applied_analytically": [THRESHOLD_EASY,
                                                THRESHOLD_NORMAL],
            "threshold_mapping": (
                "gym-fx _coerce_action parity: t==0: a>0 long, a<0 "
                "short, a==0 hold; t>0: a>=t long, a<=-t short, else "
                "hold"),
            "observation_serialization": OBS_SERIALIZATION,
            "action_vector_serialization": ACTION_SERIALIZATION,
            "near_unique_tolerance_formula": TOLERANCE_FORMULA,
            "abs_quantiles_declared": list(ABS_QUANTILES),
        },
        "splits": {},
        "split_outcomes": {},
        "outcome": None,
        "refusals": [],
        "declarations": {
            "evidence_class": "mechanism_diagnostic",
            "no_learning": True,
            "no_selection": True,
            "read_only_inputs": True,
            "decision_eligible": False,
            "live_promotion_eligible": False,
        },
    }
    refusals: list[str] = []

    # 1. pinned-input verification — refuse typed BEFORE any model load.
    if resolved_contract.is_file():
        payload["contract"]["sha256_observed"] = sysid.sha_file(
            resolved_contract)
    if contract_pinned:
        refusal = verify_sha256(resolved_contract, PINNED_CONTRACT_SHA256,
                                "ladder contract")
        if refusal:
            refusals.append(refusal)
    elif not resolved_contract.is_file():
        refusals.append(
            f"ladder contract: file missing at {resolved_contract}")
    refusal = verify_sha256(artifact_path, spec["sha256"],
                            f"artifact {artifact_key}")
    if refusal:
        refusals.append(refusal)
    else:
        payload["artifact"]["sha256_observed"] = spec["sha256"]
    if refusals:
        payload.update(refusal_result(refusals))
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        payload["elapsed_seconds"] = (
            datetime.now(timezone.utc) - started).total_seconds()
        return payload

    try:
        contract = ladder.load_contract(resolved_contract)
        payload["code_identity"] = ladder.source_identities()

        # 2. D0 config through the ladder's ONLY materialization path
        # (fail-closed base-config/manifest/CSV/nested-union checks).
        with tempfile.TemporaryDirectory(
                prefix="boundary_replay_") as scratch:
            config = ladder.materialize_arm_config(
                contract, LADDER_ARM, Path(scratch))
            identity = config.pop("_identity")
            eval_config = dict(config)
            eval_config["solvency_mode"] = "normal_realistic"
            eval_config["load_model"] = str(artifact_path)
            eval_config["quiet_mode"] = True
            eval_config["return_trace_dir"] = None
            payload["config_identity"] = identity
            payload["resolved_eval_config_sha256"] = (
                sysid.resolved_config_sha256(eval_config))
            payload["replay_design"]["eval_seed"] = int(
                eval_config["eval_seed"])

            from pipeline_plugins.rl_pipeline_with_validation import (
                PipelinePlugin as ValidationPipeline)
            from pipeline_plugins._observation_contract import (
                validate_observation_contract)

            validate_observation_contract(eval_config)
            pipeline = ValidationPipeline(eval_config)
            paths = pipeline._split_csv(eval_config)
            env_plugin_name = eval_config.get("env_plugin", "gym_fx_env")
            agent = ladder._agent_plugin(
                identity["plugins"]["agent_plugin"])

            from stable_baselines3 import SAC

            model = None
            for label, key in REPLAY_SPLITS:
                csv_path = paths[key]
                plug, env = pipeline._make_split_env(
                    env_plugin_name, eval_config, csv_path, agent)
                try:
                    if model is None:
                        # sac_agent.Plugin.load parity plus the explicit
                        # device binding this diagnostic pins per fleet
                        # GPU (SAC.load(path, env=env) is the plugin's
                        # exact call).
                        agent._require_continuous(env)
                        model = SAC.load(str(artifact_path), env=env,
                                         device=device)
                    replay = flat_hold_replay(
                        env,
                        lambda obs: agent.predict(
                            model, obs, deterministic=True),
                        seed=int(eval_config["eval_seed"]))
                finally:
                    try:
                        plug.close()
                    except Exception:
                        pass
                facts = split_facts(
                    replay["raw_actions"],
                    observation_count=replay["observation_count"],
                    observation_manifest_sha256=replay[
                        "observation_manifest_sha256"])
                facts.update({
                    "split_key": key,
                    "split_csv_path": str(csv_path),
                    "split_csv_sha256": sysid.sha_file(Path(csv_path)),
                    "split_csv_rows": _csv_row_count(Path(csv_path)),
                    "observation_shape": replay["observation_shape"],
                    "termination": {
                        "terminated": replay["terminated"],
                        "truncated": replay["truncated"],
                        "steps_capped": replay["steps_capped"],
                    },
                })
                if replay["observation_count"] <= 0:
                    refusals.append(
                        f"split {label}: empty replay (0 observations)")
                if replay["steps_capped"]:
                    refusals.append(
                        f"split {label}: replay hit the "
                        f"{MAX_REPLAY_STEPS}-step cap before episode end")
                payload["splits"][label] = facts
                if not refusals:
                    payload["split_outcomes"][label] = classify_split(
                        facts)
    except Exception as exc:  # noqa: BLE001 — typed inconclusive
        refusals.append(f"{type(exc).__name__}: {exc}")

    if refusals:
        payload.update(refusal_result(refusals))
    else:
        outcome, join_refusals = classify_overall(
            payload["split_outcomes"])
        payload["outcome"] = outcome
        payload["refusals"] = join_refusals
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    payload["elapsed_seconds"] = (
        datetime.now(timezone.utc) - started).total_seconds()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True,
                        choices=sorted(ARTIFACTS))
    parser.add_argument("--device", required=True,
                        choices=("cuda", "cpu"))
    parser.add_argument("--output", required=True, type=Path,
                        help="JSON result path (never inside a sealed "
                             "collection root)")
    parser.add_argument("--contract", type=Path, default=None,
                        help="contract override FOR TESTS ONLY; the "
                             "default enforces the pinned ladder "
                             "contract sha256")
    args = parser.parse_args()

    payload = run_replay(args.artifact, device=args.device,
                         contract_path=args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "outcome": payload["outcome"],
        "artifact": args.artifact,
        "split_outcomes": payload.get("split_outcomes"),
        "refusals": payload.get("refusals"),
        "output": str(args.output),
    }, default=str), flush=True)
    return EXIT_CLASS.get(payload["outcome"],
                          EXIT_CLASS[OUTCOME_INCONCLUSIVE])


if __name__ == "__main__":
    sys.exit(main())
