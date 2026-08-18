#!/usr/bin/env python3
"""WP0 checkpoint probe (finding AUD-F1-20260817-277), CPU/read-only.

The return traces answer "what did the policy DO on this rollout". They
cannot answer "does the policy REACT to its input at all", because a
rollout only ever visits the states the market handed it. This probe
answers the second question directly from the checkpoint's weights.

Probes, none of which need the trading environment:

1. **Input-independence.** Feed the actor a batch of deliberately
   extreme, mutually dissimilar observations (zeros, ±1, ±3, random
   normal, random uniform, per-feature ramps). A policy whose
   deterministic output is identical across inputs this different is
   input-independent — conclusive evidence of behavioral constancy that
   no rollout can give, since a rollout cannot rule out "the states
   were all alike".
2. **Stochastic draws from the SAME state** (WP0/WP2 q4): repeated
   sampling at one observation, so exploration noise can be compared
   against the deterministic collapse.
3. **Actor distribution parameters**: mean and log-std statistics — a
   very negative log-std is a policy that has annealed to determinism.
4. **Critic response over an action grid** (WP0): Q1/Q2 variation as
   the action sweeps [-1, 1] at fixed observation. A flat critic cannot
   teach an actor which action is better.
5. **Parameter delta from genesis** (WP0/WP2 q3): did the weights move
   while behavior stayed constant?

Never writes into an experiment root; never opens sealed 2025; loads on
CPU with ``device="cpu"``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "agent_multi.p1lr_actor_probe.v1"
SEALED_MARKERS = ("sealed_test", "sealed", "2025")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_not_sealed(path: Path) -> None:
    low = str(path).lower()
    for marker in SEALED_MARKERS:
        if marker in low:
            raise RuntimeError(
                f"REFUSED_SEALED_ARTIFACT: {path} matches {marker!r}")


def _stats(values) -> dict:
    import numpy as np
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "spread": float(arr.max() - arr.min()),
        "unique_count": int(np.unique(np.round(arr, 12)).size),
    }


def build_probe_observations(dim: int, seed: int = 0):
    """Deliberately dissimilar observations. If the actor cannot tell
    THESE apart, it cannot tell market states apart."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = [
        np.zeros(dim),
        np.ones(dim),
        -np.ones(dim),
        np.full(dim, 3.0),
        np.full(dim, -3.0),
        np.linspace(-3.0, 3.0, dim),
        np.linspace(3.0, -3.0, dim),
        rng.normal(0.0, 1.0, dim),
        rng.normal(0.0, 3.0, dim),
        rng.uniform(-5.0, 5.0, dim),
    ]
    for _ in range(10):
        rows.append(rng.normal(0.0, 1.0, dim))
    return np.stack(rows).astype(np.float32)


def probe(model_path: Path, *, genesis_path: Path | None,
          draws: int, grid: int) -> dict:
    import numpy as np
    import torch
    from stable_baselines3 import SAC

    _assert_not_sealed(model_path)
    model = SAC.load(str(model_path), device="cpu")
    obs_space = model.observation_space
    dim = int(np.prod(obs_space.shape))
    observations = build_probe_observations(dim)

    # 1. input-independence: deterministic action per observation
    deterministic, _ = model.predict(observations, deterministic=True)
    det = np.asarray(deterministic, dtype=float).reshape(len(observations), -1)[:, 0]

    # 2. stochastic draws from ONE state
    one = np.repeat(observations[:1], draws, axis=0)
    stochastic, _ = model.predict(one, deterministic=False)
    sto = np.asarray(stochastic, dtype=float).reshape(draws, -1)[:, 0]

    # 3/4. actor distribution and critic response over an action grid
    actor_stats: dict = {}
    critic_stats: dict = {}
    with torch.no_grad():
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        try:
            mean_actions, log_std, _ = model.policy.actor.get_action_dist_params(tensor)
            actor_stats = {
                "mean_action": _stats(mean_actions.cpu().numpy()),
                "log_std": _stats(log_std.cpu().numpy()),
                "implied_sigma_mean": float(
                    torch.exp(log_std).mean().cpu().numpy()),
            }
        except Exception as error:  # pragma: no cover - shape variance
            actor_stats = {"unavailable": str(error)}
        try:
            base = tensor[:1].repeat(grid, 1)
            sweep = torch.linspace(-1.0, 1.0, grid).reshape(grid, 1)
            q_values = model.policy.critic(base, sweep)
            q_arrays = [q.cpu().numpy().reshape(-1) for q in q_values]
            critic_stats = {
                f"q{i + 1}": _stats(q) for i, q in enumerate(q_arrays)}
            critic_stats["action_grid"] = {
                "points": grid, "min": -1.0, "max": 1.0}
        except Exception as error:  # pragma: no cover
            critic_stats = {"unavailable": str(error)}

    # 5. parameter delta from genesis
    delta: dict = {"available": False}
    if genesis_path is not None and genesis_path.is_file():
        _assert_not_sealed(genesis_path)
        base_model = SAC.load(str(genesis_path), device="cpu")
        total, moved, l2 = 0, 0, 0.0
        with torch.no_grad():
            now = dict(model.policy.state_dict())
            before = dict(base_model.policy.state_dict())
            for key, tensor_now in now.items():
                tensor_before = before.get(key)
                if tensor_before is None or \
                        tensor_now.shape != tensor_before.shape:
                    continue
                diff = (tensor_now - tensor_before).float()
                total += diff.numel()
                moved += int((diff.abs() > 0).sum().item())
                l2 += float((diff ** 2).sum().item())
        delta = {
            "available": True,
            "genesis_file": str(genesis_path),
            "genesis_sha256": _sha_file(genesis_path),
            "parameters_compared": total,
            "parameters_changed": moved,
            "changed_fraction": (moved / total) if total else None,
            "l2_norm": math.sqrt(l2),
        }

    det_stats = _stats(det)
    return {
        "model_file": str(model_path),
        "model_sha256": _sha_file(model_path),
        "observation_dim": dim,
        "input_independence": {
            "probe_observations": len(observations),
            "deterministic_action": det_stats,
            "input_independent": det_stats.get("spread", 0.0) <= 1e-9,
            "note": ("identical deterministic output across deliberately "
                     "dissimilar observations means the mapping ignores "
                     "its input"),
        },
        "stochastic_same_state": _stats(sto),
        "actor": actor_stats,
        "critic": critic_stats,
        "parameter_delta_from_genesis": delta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WP0 CPU checkpoint probe for finding 277")
    parser.add_argument("--model", required=True, type=Path, nargs="+")
    parser.add_argument("--genesis", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--grid", type=int, default=41)
    args = parser.parse_args(argv)

    results, refusals = [], []
    for model_path in args.model:
        try:
            results.append(probe(model_path.expanduser(),
                                 genesis_path=(args.genesis.expanduser()
                                               if args.genesis else None),
                                 draws=args.draws, grid=args.grid))
        except Exception as error:
            refusals.append({"model": str(model_path),
                             "error": f"{type(error).__name__}: {error}"})

    report = {
        "schema": SCHEMA,
        "hostname": socket.gethostname(),
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "probes": results,
        "refusals": refusals,
    }
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "outcome": "PROBED", "out": str(out),
        "probed": len(results), "refused": len(refusals),
        "input_independent": [
            r["input_independence"]["input_independent"] for r in results],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
