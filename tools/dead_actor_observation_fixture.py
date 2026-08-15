#!/usr/bin/env python3
"""CPU before/after evidence for AUD-P1LR-20260815-235.

Claim under test: ``include_price_window: true`` — 32 raw ETH closes plus
their 32 raw diffs appended to an otherwise rolling-z-scored, +-10-clipped
2656-dim feature block — is what kills the SAC actor's first hidden layer,
and removing it keeps the actor alive.

The fixture is deliberately small, deterministic and CPU-only, and it
reuses the REAL pieces so the result is not an analogy:

  * the real observation builder, ``feature_window_preprocessor`` from
    gym-fx, on the real pinned ETHUSD 4h dataset;
  * the real feature set and scaling of the held-fixed base config
    (83 columns, ``rolling_zscore``, window 32, clip 10);
  * a real Stable-Baselines3 SAC actor with the real architecture
    (``net_arch=[256, 256]``, ``MlpPolicy``) and the real seed handling.

Only the environment is a stand-in: a replay env that walks the same real
observations and pays ``action * next-bar log return``. That keeps a
genuine gradient signal flowing without the cost of a full gym-fx trading
rollout. The arms necessarily have different first-layer input shapes, so
this is a bounded system-level mechanism screen, not a same-weight causal
proof. The sealed-artifact replay supplies that separate proof.

Two measurements are reported per arm:

``at_init``
    before any gradient step.  This already separates the arms, and it is
    the important half of the evidence: with the raw price window the
    sign of a unit's pre-activation is fixed by the always-positive price
    level, so a large share of units is dead for EVERY observation from
    initialisation — and a dead ReLU has exactly zero gradient, so those
    units can never come back.  No learning rate reaches them.

``after_training``
    after a bounded number of real SAC updates.

Usage::

    python tools/dead_actor_observation_fixture.py            # both arms
    python tools/dead_actor_observation_fixture.py --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _actor_liveness as _liveness  # noqa: E402

#: The held-fixed base config the L1/L2 programs bind.
BASE_CONFIG = REPO / "examples/results/project3_ethusdt_4h_sac_train_val_test_v2/config_out.json"

#: The held-fixed dataset. The ladder contract pins it by absolute path
#: and sha256, so read the pin rather than hardcoding a location that is
#: wrong in a worktree.
_LADDER_CONTRACT = (
    REPO / "examples/config/phase_3_eth_sac_dynamics/m0_l1_mechanism_ladder_v1.json")


def _pinned_dataset() -> Path:
    candidates = []
    try:
        pin = json.loads(_LADDER_CONTRACT.read_text())["common"]["data"]
        candidates.append(Path(str(pin["path"])).expanduser())
    except (OSError, KeyError, ValueError):
        pass
    candidates.append(
        REPO / "examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


DATA_PATH = _pinned_dataset()


def _expected_dataset_sha256() -> str | None:
    try:
        return str(json.loads(_LADDER_CONTRACT.read_text())["common"][
            "data"]["sha256"])
    except (OSError, KeyError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

#: A contiguous, deterministic slice well inside the dataset so every
#: observation has a full 256-row causal scaling history.
DEFAULT_START_ROW = 12_000
DEFAULT_OBSERVATIONS = 1_024
DEFAULT_UPDATES = 3_000
DEFAULT_SEED = 101

ARMS = ("with_raw_price_window", "feature_only")


def _load_base_config() -> Dict[str, Any]:
    config = json.loads(BASE_CONFIG.read_text())
    # Present in the preprocessor's defaults but absent from the archived
    # config; the contract's value is 10.
    config.setdefault("feature_clip", 10.0)
    return config


def build_observations(*, include_price_window: bool, start_row: int,
                       count: int) -> tuple:
    """Real observations, flattened exactly as a Dict space flattens.

    Returns ``(observations, next_returns, price_stats)``.
    """
    import pandas as pd
    from preprocessor_plugins.feature_window_preprocessor import Plugin

    base = _load_base_config()
    frame = pd.read_csv(DATA_PATH)
    config = {
        "window_size": int(base["window_size"]),
        "price_column": base.get("price_column", "CLOSE"),
        "feature_columns": list(base["feature_columns"]),
        "feature_binary_columns": list(base.get("feature_binary_columns")
                                       or []),
        "feature_scaling": base["feature_scaling"],
        "feature_scaling_window": int(base["feature_scaling_window"]),
        "feature_clip": float(base["feature_clip"]),
        "include_price_window": bool(include_price_window),
        "include_agent_state": bool(base.get("include_agent_state", True)),
        "position_size": float(base.get("position_size", 1.0) or 1.0),
    }
    plugin = Plugin()
    closes = frame[config["price_column"]].astype(float).to_numpy()

    rows: List[np.ndarray] = []
    returns: List[float] = []
    price_values: List[float] = []
    total_bars = int(len(frame))
    for offset in range(count):
        step = int(start_row + offset)
        bridge = {
            "initial_cash": 10_000.0,
            "equity": 10_000.0,
            "price": float(closes[step]),
            "position": 0,
            "bar_index": step,
            "total_bars": total_bars,
        }
        obs = plugin.make_observation(data=frame, step=step,
                                      bridge_state=bridge, config=config)
        # gymnasium's FlattenObservation concatenates a Dict space in
        # sorted key order; mirror that so the vector is the one the
        # policy would really see.
        flat = np.concatenate([np.asarray(obs[key], dtype=np.float32).ravel()
                               for key in sorted(obs)])
        rows.append(flat)
        if include_price_window:
            price_values.append(float(np.abs(obs["prices"]).mean()))
        nxt = float(closes[min(step + 1, len(closes) - 1)])
        cur = float(closes[step])
        returns.append(float(np.log(max(nxt, 1e-9) / max(cur, 1e-9))))
    observations = np.stack(rows)
    stats = {
        "observation_dim": int(observations.shape[1]),
        "abs_max": float(np.abs(observations).max()),
        "abs_mean": float(np.abs(observations).mean()),
        "raw_price_block_abs_mean": (float(np.mean(price_values))
                                     if price_values else None),
    }
    return observations, np.asarray(returns, dtype=np.float32), stats


import gymnasium as gym  # noqa: E402


class _ObservationReplayEnv(gym.Env):
    """Minimal Gymnasium env replaying real observations.

    Identical between the two arms except for the observation vector, so
    it can never be the explanation for a difference between them.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(self, observations: np.ndarray, returns: np.ndarray) -> None:
        super().__init__()
        self._observations = observations
        self._returns = returns
        self._index = 0
        dim = int(observations.shape[1])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.render_mode = None
        self.spec = None

    def reset(self, *, seed: int | None = None, options: Any = None):
        self._index = 0
        return self._observations[0], {}

    def step(self, action):
        reward = float(np.asarray(action).reshape(-1)[0]
                       * self._returns[self._index]) * 100.0
        self._index += 1
        terminated = False
        truncated = self._index >= len(self._observations) - 1
        obs = self._observations[min(self._index,
                                     len(self._observations) - 1)]
        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        return None


def _build_actor(env, *, seed: int):
    from stable_baselines3 import SAC

    base = _load_base_config()
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-5,
        buffer_size=20_000,
        learning_starts=256,
        batch_size=int(base.get("batch_size", 256)),
        tau=float(base.get("tau", 0.005)),
        gamma=float(base.get("gamma", 0.99)),
        train_freq=1,
        gradient_steps=1,
        ent_coef=base.get("ent_coef", 0.2),
        policy_kwargs={"net_arch": list(base.get("net_arch", [256, 256]))},
        verbose=0,
        seed=int(seed),
        device="cpu",
    )
    model.set_random_seed(int(seed))
    return model


def _deterministic_actions(model, observations: np.ndarray) -> np.ndarray:
    actions, _ = model.predict(observations, deterministic=True)
    return np.asarray(actions, dtype=np.float32).reshape(len(observations), -1)[:, 0]


def _live_mask(model, observations: np.ndarray) -> np.ndarray:
    """Per-unit live mask of the actor's first layer on this batch."""
    weight, bias, _ = _liveness.first_layer_parameters(model)
    pre = np.asarray(observations, dtype=np.float64) @ weight.T + bias
    return (pre > 0.0).any(axis=0)


def _measure(model, observations: np.ndarray, *, label: str,
             arm: str) -> Dict[str, Any]:
    actions = _deterministic_actions(model, observations)
    facts = _liveness.actor_liveness_facts(
        model=model,
        observations=observations,
        actions=actions,
        split=arm,
        phase=label,
    )
    facts["action_std"] = float(np.std(actions.astype(np.float64)))
    facts["action_unique_count"] = int(np.unique(actions).size)
    facts["action_min"] = float(np.min(actions))
    facts["action_max"] = float(np.max(actions))
    return facts


def run_arm(*, include_price_window: bool, start_row: int, count: int,
            updates: int, seed: int) -> Dict[str, Any]:
    arm = ARMS[0] if include_price_window else ARMS[1]
    observations, returns, stats = build_observations(
        include_price_window=include_price_window,
        start_row=start_row, count=count)
    env = _ObservationReplayEnv(observations, returns)
    model = _build_actor(env, seed=seed)
    at_init = _measure(model, observations, label="at_init", arm=arm)
    mask_init = _live_mask(model, observations)
    if updates > 0:
        model.learn(total_timesteps=int(updates), log_interval=10 ** 9)
    after = _measure(model, observations, label="after_training", arm=arm)
    mask_after = _live_mask(model, observations)

    dead_init = ~mask_init
    revived = int(np.count_nonzero(dead_init & mask_after))
    died = int(np.count_nonzero(mask_init & ~mask_after))
    return {
        "arm": arm,
        "include_price_window": bool(include_price_window),
        "observation_stats": stats,
        "sac_updates": int(updates),
        "at_init": at_init,
        "after_training": after,
        # The zero-gradient claim, measured rather than asserted: a unit
        # that fires on no observation receives exactly zero gradient, so
        # it can never come back. ``revived_units`` is expected to be 0.
        "zero_gradient_evidence": {
            "dead_at_init": int(np.count_nonzero(dead_init)),
            "revived_units": revived,
            "newly_dead_units": died,
            "dead_at_init_still_dead": int(
                np.count_nonzero(dead_init & ~mask_after)),
            "claim": "units dead at initialisation receive zero gradient "
                     "and never revive; revived_units must be 0",
        },
    }


def _row(arm: Dict[str, Any], phase: str) -> str:
    facts = arm[phase]
    return (
        f"  {phase:<15} "
        f"live={facts['live_unit_count']:>3}/{facts['first_layer_units']:<3} "
        f"({facts['live_unit_fraction']:.4f})  "
        f"varying={facts['varying_unit_count']:>3}  "
        f"pre_mean={facts['preactivation_mean']:>+12.4g}  "
        f"action_std={facts['action_std']:.3e}  "
        f"uniq={facts['action_unique_count']:>4}  "
        f"{facts['classification']}"
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-row", type=int, default=DEFAULT_START_ROW)
    parser.add_argument("--observations", type=int,
                        default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    if not DATA_PATH.is_file():
        print(f"REFUSED: pinned dataset missing: {DATA_PATH}")
        return 4
    expected_dataset_sha256 = _expected_dataset_sha256()
    observed_dataset_sha256 = _sha256(DATA_PATH)
    if (not expected_dataset_sha256
            or observed_dataset_sha256 != expected_dataset_sha256):
        print("REFUSED: dataset does not match the ladder contract: "
              f"expected={expected_dataset_sha256} "
              f"observed={observed_dataset_sha256}")
        return 4

    import torch

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)
    np.random.seed(args.seed)

    arms = []
    for include in (True, False):
        arms.append(run_arm(include_price_window=include,
                            start_row=args.start_row,
                            count=args.observations,
                            updates=args.updates,
                            seed=args.seed))

    print("AUD-P1LR-20260815-235 — raw price window vs feature-only "
          "observation")
    print(f"dataset={DATA_PATH.name} rows[{args.start_row}:"
          f"{args.start_row + args.observations}] seed={args.seed} "
          f"sac_updates={args.updates} device=cpu")
    for arm in arms:
        stats = arm["observation_stats"]
        print(f"\n{arm['arm']}  (include_price_window="
              f"{arm['include_price_window']})")
        print(f"  observation_dim={stats['observation_dim']} "
              f"abs_max={stats['abs_max']:.4g} "
              f"abs_mean={stats['abs_mean']:.4g} "
              f"raw_price_block_abs_mean="
              f"{stats['raw_price_block_abs_mean']}")
        print(_row(arm, "at_init"))
        print(_row(arm, "after_training"))
        zero = arm["zero_gradient_evidence"]
        print(f"  zero-gradient    dead_at_init={zero['dead_at_init']:>3}  "
              f"revived={zero['revived_units']:>3}  "
              f"newly_dead={zero['newly_dead_units']:>3}  "
              f"dead_at_init_still_dead="
              f"{zero['dead_at_init_still_dead']:>3}")

    payload = {
        "schema": "agent_multi.dead_actor_observation_fixture.v1",
        "finding": "AUD-P1LR-20260815-235",
        "dataset": str(DATA_PATH),
        "dataset_sha256": observed_dataset_sha256,
        "base_config": str(BASE_CONFIG),
        "start_row": int(args.start_row),
        "observations": int(args.observations),
        "sac_updates": int(args.updates),
        "seed": int(args.seed),
        "device": "cpu",
        "arms": arms,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
