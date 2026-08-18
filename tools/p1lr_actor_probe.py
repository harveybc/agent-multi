#!/usr/bin/env python3
"""WP-C (finding AUD-F1-20260817-277): CPU read-only checkpoint probe.

A rollout trace cannot separate "the policy reacted to its input" from
"the input moved and the policy followed something else", because a
rollout only ever visits the states the market handed it. This probe
measures the checkpoint itself.

**Evidence contract, per the 2026-08-17 order.** Twenty arbitrary
synthetic vectors are a DIAGNOSTIC, never a global proof of input
independence. A promotable verdict requires a REAL-ROLE observation
batch (``--observations``) produced by the pipeline's own preprocessor
and bound to its contract hash; the synthetic sweep only supplements
it. Without real observations the probe still runs and reports, and the
result is explicitly typed non-promotable.

Probes:

1. real-role observation sensitivity, with the identical-observation
   and row-permutation controls the classifier demands;
2. synthetic dissimilar-input sweep (diagnostic only);
3. repeated stochastic draws from ONE state;
4. actor mean/log-std;
5. critic Q1/Q2 over an action grid;
6. parameter deltas reported SEPARATELY for actor, critic and target
   critic — the whole policy state is not an "actor delta".

Refuses: sealed artifacts by path, output inside a campaign identity,
invalid draw/grid counts, and any non-finite probe output. Hashes each
model before and after use. Exits non-zero unless at least one model
was probed completely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _policy_behavior as pb  # noqa: E402

SCHEMA = "agent_multi.p1lr_actor_probe.v2"
SEALED_MARKERS = ("sealed_test", "sealed", "2025")
#: A campaign identity root is 16 lowercase hex characters.
IDENTITY_DIR_LENGTH = 16
MAX_DRAWS = 100_000
MAX_GRID = 10_000


class ProbeRefusal(RuntimeError):
    """Typed refusal; the probe fails closed."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_not_sealed(path: Path) -> None:
    parts = [str(path).lower()] + [p.lower() for p in Path(path).parts]
    for marker in SEALED_MARKERS:
        if any(marker in part for part in parts):
            raise ProbeRefusal(
                f"REFUSED_SEALED_ARTIFACT: {path} matches {marker!r} in "
                "its resolved path")


def assert_output_outside_identity(out: Path) -> None:
    """Refuse to write inside a campaign identity: a 16-hex directory
    component is an experiment identity root."""
    for part in out.resolve().parts:
        if len(part) == IDENTITY_DIR_LENGTH and \
                all(c in "0123456789abcdef" for c in part):
            raise ProbeRefusal(
                f"REFUSED_OUTPUT_INSIDE_IDENTITY: {out} contains the "
                f"campaign identity component {part!r}; a diagnostic "
                "never writes into the identity it measures")


def _validate_count(name: str, value: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ProbeRefusal(f"REFUSED_INVALID_{name.upper()}: "
                           f"{value!r} is not an integer") from error
    if number < 2 or number > maximum:
        raise ProbeRefusal(
            f"REFUSED_INVALID_{name.upper()}: {number} is outside "
            f"[2, {maximum}]")
    return number


def _finite_array(name: str, values):
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ProbeRefusal(
            f"REFUSED_NONFINITE_OUTPUT: {name} contains NaN or Inf; a "
            "probe never reports a non-finite measurement")
    return arr


def _stats(name: str, values) -> dict:
    import numpy as np
    arr = _finite_array(name, values).reshape(-1)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()), "max": float(arr.max()),
        "mean": float(arr.mean()), "std": float(arr.std()),
        "spread": float(arr.max() - arr.min()),
        "unique_count": int(np.unique(np.round(arr, 12)).size),
    }


def build_probe_observations(dim: int, seed: int = 0):
    """Deliberately dissimilar synthetic observations. DIAGNOSTIC ONLY:
    if the actor cannot tell THESE apart it is input-independent, but
    telling them apart does NOT prove it discriminates real states."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = [np.zeros(dim), np.ones(dim), -np.ones(dim),
            np.full(dim, 3.0), np.full(dim, -3.0),
            np.linspace(-3.0, 3.0, dim), np.linspace(3.0, -3.0, dim),
            rng.normal(0.0, 1.0, dim), rng.normal(0.0, 3.0, dim),
            rng.uniform(-5.0, 5.0, dim)]
    rows += [rng.normal(0.0, 1.0, dim) for _ in range(10)]
    return np.stack(rows).astype("float32")


def load_observations(path: Path, dim: int):
    """Real-role observation batch produced by the pipeline, never
    re-derived here — a second preprocessor would be a second source of
    truth."""
    import numpy as np
    assert_not_sealed(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.loadtxt(path, delimiter=",", ndmin=2)
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ProbeRefusal(
            f"REFUSED_OBSERVATION_SHAPE: {path} has shape {arr.shape}, "
            f"expected (rows, {dim})")
    _finite_array("observations", arr)
    return arr


def _parameter_deltas(model, base_model) -> dict:
    """Actor, critic and target-critic deltas reported SEPARATELY."""
    import torch
    groups = {"actor": "actor.", "critic": "critic.",
              "critic_target": "critic_target."}
    out: dict = {}
    with torch.no_grad():
        now = dict(model.policy.state_dict())
        before = dict(base_model.policy.state_dict())
        for label, prefix in groups.items():
            total, moved, l2 = 0, 0, 0.0
            for key, tensor_now in now.items():
                if not key.startswith(prefix):
                    continue
                tensor_before = before.get(key)
                if tensor_before is None or \
                        tensor_now.shape != tensor_before.shape:
                    continue
                diff = (tensor_now - tensor_before).float()
                total += diff.numel()
                moved += int((diff.abs() > 0).sum().item())
                l2 += float((diff ** 2).sum().item())
            out[label] = {
                "parameters_compared": total,
                "parameters_changed": moved,
                "changed_fraction": (moved / total) if total else None,
                "l2_norm": math.sqrt(l2),
            }
    return out


def probe(model_path: Path, *, genesis_path: Path | None,
          observations_path: Path | None,
          observation_contract_sha256: str | None,
          role: str | None,
          threshold: float, draws: int, grid: int) -> dict:
    import numpy as np
    import torch
    from stable_baselines3 import SAC

    model_path = Path(model_path).resolve()
    assert_not_sealed(model_path)
    sha_before = _sha_file(model_path)
    model = SAC.load(str(model_path), device="cpu")
    dim = int(np.prod(model.observation_space.shape))

    # 1. real-role observation sensitivity (the only promotable path)
    observation_evidence: dict = {"available": False}
    behavior: dict | None = None
    if observations_path is not None:
        real = load_observations(Path(observations_path).resolve(), dim)
        det, _ = model.predict(real, deterministic=True)
        det = _finite_array("real deterministic actions", det
                            ).reshape(len(real), -1)[:, 0]
        repeated_input = np.repeat(real[:1], 3, axis=0)
        rep, _ = model.predict(repeated_input, deterministic=True)
        rep = _finite_array("identical-observation control", rep
                            ).reshape(3, -1)[:, 0]
        order = np.random.default_rng(0).permutation(len(real))
        perm, _ = model.predict(real[order], deterministic=True)
        perm = _finite_array("row-permutation control", perm
                             ).reshape(len(real), -1)[:, 0]
        evidence = {
            "model_sha256": sha_before,
            "observation_contract_sha256": observation_contract_sha256,
            "observation_rows": int(len(real)),
            "role": role,
            "observations_file": str(observations_path),
        }
        try:
            behavior = pb.classify_with_observation_evidence(
                det.tolist(), threshold=threshold,
                observation_evidence=evidence,
                repeated_observation_actions=rep.tolist(),
                permuted_observation_actions=perm.tolist())
            observation_evidence = {"available": True, **evidence}
        except pb.PolicyBehaviorError as error:
            observation_evidence = {"available": False,
                                    "refusal": str(error), **evidence}

    # 2. synthetic sweep — DIAGNOSTIC ONLY
    synthetic = build_probe_observations(dim)
    syn, _ = model.predict(synthetic, deterministic=True)
    syn = _finite_array("synthetic actions", syn
                        ).reshape(len(synthetic), -1)[:, 0]
    syn_stats = _stats("synthetic actions", syn)

    # 3. stochastic draws from ONE state
    one = np.repeat(synthetic[:1], draws, axis=0)
    sto, _ = model.predict(one, deterministic=False)
    sto_stats = _stats("stochastic draws",
                       np.asarray(sto).reshape(draws, -1)[:, 0])

    # 4/5. actor distribution and critic response
    actor_stats: dict = {}
    critic_stats: dict = {}
    with torch.no_grad():
        tensor = torch.as_tensor(synthetic, dtype=torch.float32)
        try:
            mean_actions, log_std, _ = \
                model.policy.actor.get_action_dist_params(tensor)
            actor_stats = {
                "mean_action": _stats("actor mean",
                                      mean_actions.cpu().numpy()),
                "log_std": _stats("actor log_std",
                                  log_std.cpu().numpy()),
                "implied_sigma_mean": float(
                    torch.exp(log_std).mean().cpu().numpy()),
            }
        except ProbeRefusal:
            raise
        except Exception as error:
            actor_stats = {"unavailable": str(error)}
        try:
            base = tensor[:1].repeat(grid, 1)
            sweep = torch.linspace(-1.0, 1.0, grid).reshape(grid, 1)
            q_values = model.policy.critic(base, sweep)
            critic_stats = {
                f"q{i + 1}": _stats(f"critic q{i + 1}",
                                    q.cpu().numpy().reshape(-1))
                for i, q in enumerate(q_values)}
            critic_stats["action_grid"] = {"points": grid,
                                           "min": -1.0, "max": 1.0}
        except ProbeRefusal:
            raise
        except Exception as error:
            critic_stats = {"unavailable": str(error)}

    # 6. separated parameter deltas
    deltas: dict = {"available": False}
    if genesis_path is not None:
        genesis_path = Path(genesis_path).resolve()
        assert_not_sealed(genesis_path)
        if genesis_path.is_file():
            base_model = SAC.load(str(genesis_path), device="cpu")
            deltas = {"available": True,
                      "genesis_file": str(genesis_path),
                      "genesis_sha256": _sha_file(genesis_path),
                      "groups": _parameter_deltas(model, base_model)}

    sha_after = _sha_file(model_path)
    if sha_after != sha_before:
        raise ProbeRefusal(
            f"REFUSED_MODEL_CHANGED_DURING_USE: {model_path} hashed "
            f"{sha_before} before and {sha_after} after")

    return {
        "model_file": str(model_path),
        "model_sha256_before": sha_before,
        "model_sha256_after": sha_after,
        "observation_dim": dim,
        "threshold": float(threshold),
        "real_role_observation_evidence": observation_evidence,
        "behavior": behavior,
        "promotable_as_learned_activity": bool(
            behavior and behavior.get("promotable_as_learned_activity")),
        "synthetic_sweep": {
            "diagnostic_only": True,
            "note": ("synthetic vectors cannot prove input dependence; "
                     "identical output across them proves independence"),
            "probe_observations": len(synthetic),
            "deterministic_action": syn_stats,
            "input_independent": syn_stats.get("spread", 0.0) <= 1e-9,
        },
        "stochastic_same_state": sto_stats,
        "actor": actor_stats,
        "critic": critic_stats,
        "parameter_delta_from_genesis": deltas,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WP-C CPU checkpoint probe for finding 277")
    parser.add_argument("--model", required=True, type=Path, nargs="+")
    parser.add_argument("--genesis", type=Path, default=None)
    parser.add_argument("--observations", type=Path, default=None,
                        help="real-role observation batch (.npy/.csv) "
                             "produced by the pipeline; without it the "
                             "result is non-promotable")
    parser.add_argument("--observation-contract-sha256", default=None)
    parser.add_argument("--role", default=None)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--grid", type=int, default=41)
    args = parser.parse_args(argv)

    try:
        out = args.out.expanduser()
        assert_output_outside_identity(out)
        draws = _validate_count("draws", args.draws, MAX_DRAWS)
        grid = _validate_count("grid", args.grid, MAX_GRID)
    except ProbeRefusal as error:
        print(json.dumps({"outcome": "REFUSED", "detail": str(error)}))
        return 2

    results, refusals = [], []
    for model_path in args.model:
        try:
            results.append(probe(
                model_path.expanduser(),
                genesis_path=args.genesis,
                observations_path=args.observations,
                observation_contract_sha256=args.observation_contract_sha256,
                role=args.role, threshold=args.threshold,
                draws=draws, grid=grid))
        except Exception as error:
            refusals.append({"model": str(model_path),
                             "error": f"{type(error).__name__}: {error}"})

    report = {
        "schema": SCHEMA,
        "hostname": socket.gethostname(),
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "probes": results, "refusals": refusals,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps({
        "outcome": "PROBED" if results else "NO_MODEL_PROBED",
        "out": str(out), "digest": _sha_file(out),
        "probed": len(results), "refused": len(refusals),
        "promotable": [r["promotable_as_learned_activity"]
                       for r in results]}))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
