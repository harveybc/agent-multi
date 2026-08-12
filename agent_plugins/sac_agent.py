"""
sac_agent.py — stable-baselines3 SAC wrapper as an agent plugin.

SAC uses a continuous action space. gym-fx exposes `action_space_mode` —
when set to "continuous", the env returns Box(-1, +1, shape=(1,)) and
BTBridgeStrategy thresholds at ±0.33 to {-1, 0, +1} (per plan decision,
apples-to-apples with discrete baselines).

SAC also needs a flat obs; we wrap the env with FlattenObservation in build.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ._progress_callback import make_progress_callback


_WEIGHT_TRANSFER_REPLAY_CAPACITY = 1


def _load_sac_source_for_weight_transfer(SAC, path: str, *, device: str):
    """Load only the source policy without recreating its training buffer."""
    return SAC.load(
        path,
        device=device,
        custom_objects={"buffer_size": _WEIGHT_TRANSFER_REPLAY_CAPACITY},
    )


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_tensor_hash(policy) -> str:
    """Deterministic digest of every parameter tensor in state-dict
    order. Two policies hash equal iff their tensors are byte-equal."""
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _policy_component_distances(source_policy, target_policy) -> Dict[str, float]:
    """Mean absolute difference per SAC component (actor / critic /
    critic_target). Zero everywhere proves an exact transfer."""
    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for name, source_tensor in source_state.items():
        component = name.split(".")[0]
        diff = (
            (source_tensor.detach().cpu() - target_state[name].detach().cpu())
            .abs()
            .sum()
            .item()
        )
        sums[component] = sums.get(component, 0.0) + diff
        counts[component] = counts.get(component, 0) + source_tensor.numel()
    return {
        component: (sums[component] / counts[component] if counts[component] else 0.0)
        for component in sums
    }


def _transfer_expanded_policy_state(
    source_state,
    target_state,
    *,
    source_observation_dim: int,
    target_observation_dim: int,
    action_dim: int,
):
    """Copy SAC policy tensors while adding neutral observation columns."""
    if target_observation_dim < source_observation_dim:
        raise ValueError("warm start cannot remove observation dimensions")
    transferred = {}
    expanded_keys: list[str] = []
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            transferred[key] = target_value
            continue
        if tuple(source_value.shape) == tuple(target_value.shape):
            transferred[key] = source_value
            continue
        if (
            source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
        ):
            source_width = int(source_value.shape[1])
            target_width = int(target_value.shape[1])
            replacement = target_value.clone()
            if (
                source_width == source_observation_dim
                and target_width == target_observation_dim
            ):
                replacement.zero_()
                replacement[:, :source_observation_dim] = source_value
            elif (
                source_width == source_observation_dim + action_dim
                and target_width == target_observation_dim + action_dim
            ):
                replacement.zero_()
                replacement[:, :source_observation_dim] = source_value[
                    :, :source_observation_dim
                ]
                replacement[:, -action_dim:] = source_value[:, -action_dim:]
            else:
                raise ValueError(
                    f"unsupported warm-start tensor expansion for {key}: "
                    f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
                )
            transferred[key] = replacement
            expanded_keys.append(key)
            continue
        raise ValueError(
            f"incompatible warm-start tensor {key}: "
            f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
        )
    return transferred, expanded_keys


class Plugin:
    plugin_params: Dict[str, Any] = {
        "total_timesteps": 10_000,
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 1_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto",
        "use_sde": False,
        "net_arch": (256, 256),
        "device": "auto",
        "agent_verbose": 0,
        "train_seed": 0,
        "training_progress_file": None,
        "progress_file": None,
        "progress_update_interval_steps": 1000,
        "ga_fitness_dd_lambda": 1.0,
        "oracle_behavior_pretrain_enabled": False,
        "oracle_behavior_labels_file": None,
        "oracle_behavior_pretrain_epochs": 3,
        "oracle_behavior_pretrain_batch_size": 512,
        "oracle_behavior_pretrain_hold_fraction": 0.10,
        "oracle_behavior_pretrain_max_samples": 0,
        "oracle_behavior_pretrain_clip_grad_norm": 1.0,
    }

    _PARAM_KEYS = tuple(plugin_params.keys())

    plugin_debug_vars: List[str] = [
        "total_timesteps", "learning_rate", "batch_size", "buffer_size",
        "gamma", "tau", "use_sde", "net_arch", "device", "train_seed",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self._PARAM_KEYS:
                self.params[k] = v

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

    def build(self, env, config: Dict[str, Any]):
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise ImportError("stable-baselines3 is required for sac_agent") from exc

        self._require_continuous(env)
        p = self._resolve(config)
        policy_kwargs = {"net_arch": list(p["net_arch"])}
        seed = int(p["train_seed"])
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=float(p["learning_rate"]),
            buffer_size=int(p["buffer_size"]),
            learning_starts=int(p["learning_starts"]),
            batch_size=int(p["batch_size"]),
            tau=float(p["tau"]),
            gamma=float(p["gamma"]),
            train_freq=int(p["train_freq"]),
            gradient_steps=int(p["gradient_steps"]),
            ent_coef=p["ent_coef"],
            target_update_interval=int(p["target_update_interval"]),
            target_entropy=p["target_entropy"],
            use_sde=bool(p["use_sde"]),
            policy_kwargs=policy_kwargs,
            verbose=int(p["agent_verbose"]),
            seed=seed,
            device=str(p["device"]),
        )
        # Explicitly re-seed after construction so gSDE exploration noise,
        # torch, numpy, env, and action_space all share this seed (SB3's
        # constructor-time seeding does not always reseed gSDE on SAC).
        model.set_random_seed(seed)
        if bool(p["use_sde"]):
            try:
                model.policy.reset_noise()
            except Exception:
                pass
        return model

    def train(self, model, config: Dict[str, Any]):
        p = self._resolve(config)
        total_timesteps = int(p["total_timesteps"])
        callback = make_progress_callback({**config, **p}, total_timesteps)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        return model

    def pretrain_behavior(self, model, env, config: Dict[str, Any]) -> Dict[str, Any]:
        """Supervised actor warm-up from train-only oracle actions.

        The oracle labels are not used as live features and validation/test
        labels are never consumed here.  This only nudges SAC's actor before
        the normal RL loop starts; critic/replay learning remains unchanged.
        """
        p = self._resolve(config)
        if not bool(config.get("oracle_behavior_pretrain_enabled", p["oracle_behavior_pretrain_enabled"])):
            return {"enabled": False}
        labels_file = config.get("oracle_behavior_labels_file", p["oracle_behavior_labels_file"])
        if not labels_file:
            raise ValueError("oracle_behavior_pretrain_enabled requires oracle_behavior_labels_file")
        labels_path = Path(str(labels_file))
        if not labels_path.exists():
            raise FileNotFoundError(f"oracle_behavior_labels_file not found: {labels_path}")

        import csv
        import numpy as np
        import torch as th

        labels: list[tuple[float, float]] = []
        with labels_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    action = float(row.get("oracle_action", 0.0) or 0.0)
                    confidence = float(row.get("oracle_confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    action, confidence = 0.0, 0.0
                action = max(-1.0, min(1.0, action))
                labels.append((action, max(0.0, confidence)))
        if not labels:
            raise ValueError(f"oracle_behavior_labels_file has no labels: {labels_path}")

        seed = int(config.get("train_seed", p["train_seed"]))
        rng = np.random.default_rng(seed)
        hold_fraction = float(
            config.get("oracle_behavior_pretrain_hold_fraction", p["oracle_behavior_pretrain_hold_fraction"])
        )
        max_samples = int(
            config.get("oracle_behavior_pretrain_max_samples", p["oracle_behavior_pretrain_max_samples"])
            or 0
        )
        observations: list[np.ndarray] = []
        targets: list[float] = []
        weights: list[float] = []

        obs, info = env.reset(seed=seed)
        done = False
        steps = 0
        while not done and steps < len(labels):
            bar_index = int(info.get("bar_index", steps) or steps)
            if 0 <= bar_index < len(labels):
                target_action, confidence = labels[bar_index]
                include = target_action != 0.0 or rng.random() < hold_fraction
                if include:
                    observations.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                    targets.append(target_action)
                    weights.append(1.0 if target_action == 0.0 else max(1.0, min(10.0, confidence)))
                    if max_samples > 0 and len(observations) >= max_samples:
                        break
            neutral_action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, _reward, terminated, truncated, info = env.step(neutral_action)
            done = bool(terminated or truncated)
            steps += 1

        if not observations:
            raise ValueError(f"oracle_behavior_pretrain collected zero samples from {labels_path}")

        obs_array = np.stack(observations).astype(np.float32)
        action_dim = int(np.prod(env.action_space.shape))
        target_array = np.repeat(np.asarray(targets, dtype=np.float32).reshape(-1, 1), action_dim, axis=1)
        weight_array = np.asarray(weights, dtype=np.float32).reshape(-1, 1)

        device = getattr(model, "device", "cpu")
        batch_size = int(config.get("oracle_behavior_pretrain_batch_size", p["oracle_behavior_pretrain_batch_size"]))
        epochs = int(config.get("oracle_behavior_pretrain_epochs", p["oracle_behavior_pretrain_epochs"]))
        clip_norm = float(
            config.get("oracle_behavior_pretrain_clip_grad_norm", p["oracle_behavior_pretrain_clip_grad_norm"])
            or 0.0
        )
        model.policy.set_training_mode(True)
        losses: list[float] = []
        indices = np.arange(len(obs_array))
        for _epoch in range(max(1, epochs)):
            rng.shuffle(indices)
            for start in range(0, len(indices), max(1, batch_size)):
                batch_idx = indices[start : start + max(1, batch_size)]
                obs_tensor = th.as_tensor(obs_array[batch_idx], device=device)
                target_tensor = th.as_tensor(target_array[batch_idx], device=device)
                weight_tensor = th.as_tensor(weight_array[batch_idx], device=device)
                pred = model.actor(obs_tensor, deterministic=True)
                loss = (((pred - target_tensor) ** 2) * weight_tensor).mean()
                model.actor.optimizer.zero_grad()
                loss.backward()
                if clip_norm > 0:
                    th.nn.utils.clip_grad_norm_(model.actor.parameters(), clip_norm)
                model.actor.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        nonzero = sum(1 for value in targets if value != 0.0)
        summary = {
            "enabled": True,
            "labels_file": str(labels_path),
            "samples": len(observations),
            "nonzero_samples": nonzero,
            "hold_samples": len(observations) - nonzero,
            "epochs": max(1, epochs),
            "batch_size": max(1, batch_size),
            "loss_initial": losses[0] if losses else None,
            "loss_final": losses[-1] if losses else None,
            "mean_loss": sum(losses) / len(losses) if losses else None,
        }
        if not config.get("quiet_mode"):
            print(
                "[oracle_behavior_pretrain] "
                f"samples={summary['samples']} nonzero={summary['nonzero_samples']} "
                f"loss={summary['loss_initial']}->{summary['loss_final']}",
                flush=True,
            )
        return summary

    def predict(self, model, obs, deterministic: bool = True):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    def save(self, model, path: str) -> None:
        model.save(path)

    def load(self, path: str, env):
        from stable_baselines3 import SAC
        self._require_continuous(env)
        return SAC.load(path, env=env)

    def load_for_training(self, path: str, env, config: Dict[str, Any]):
        """Warm-start weights in a model built from the candidate's genes.

        Reusing ``SAC.load(..., custom_objects=...)`` is unsafe when a
        candidate changes entropy mode: Stable-Baselines may try to restore an
        optimizer that the source archive never had.  A warm start promises
        trained weights, not stale optimizer moments, so construct the target
        model from the candidate and transfer the complete policy state.
        """
        from stable_baselines3 import SAC

        self._require_continuous(env)
        resolved = self._resolve(config)
        source = _load_sac_source_for_weight_transfer(
            SAC,
            path,
            device=str(resolved["device"]),
        )
        target_config = dict(config)
        requested_entropy = target_config.get("ent_coef", resolved["ent_coef"])
        if requested_entropy == "auto" and not isinstance(source.ent_coef, str):
            target_config["ent_coef"] = f"auto_{float(source.ent_coef):g}"

        target = self.build(env, target_config)
        source_policy_hash = _policy_tensor_hash(source.policy)
        target_hash_before = _policy_tensor_hash(target.policy)
        target.policy.load_state_dict(source.policy.state_dict(), strict=True)
        if (
            getattr(source, "log_ent_coef", None) is not None
            and getattr(target, "log_ent_coef", None) is not None
            and source.log_ent_coef.shape == target.log_ent_coef.shape
        ):
            target.log_ent_coef.data.copy_(source.log_ent_coef.data)
        # M0 boundary evidence (SAC inner-curriculum order §7): the
        # transfer must be provable from facts, not asserted — tensor
        # hash equality after the copy, component distances, and the
        # exact optimizer learning rates the fine-tune will actually use.
        target_hash_after = _policy_tensor_hash(target.policy)
        target.warm_start_transfer_evidence = {
            "source_model": str(Path(path).resolve()),
            "source_artifact_sha256": _file_sha256(
                Path(path) if Path(path).exists()
                else Path(str(path) + ".zip")
            ),
            "source_policy_tensor_hash": source_policy_hash,
            "target_policy_tensor_hash_before_transfer": target_hash_before,
            "target_policy_tensor_hash_after_transfer": target_hash_after,
            "policy_hash_matches_source_after_transfer": (
                target_hash_after == source_policy_hash
            ),
            "component_l1_distances_after_transfer": (
                _policy_component_distances(source.policy, target.policy)
            ),
            "source_entropy_mode": (
                "automatic"
                if getattr(source, "ent_coef_optimizer", None) is not None
                else "fixed"
            ),
            "source_entropy_value": (
                None if isinstance(source.ent_coef, str)
                else float(source.ent_coef)
            ),
            "target_entropy_mode": (
                "automatic"
                if getattr(target, "ent_coef_optimizer", None) is not None
                else "fixed"
            ),
            "target_entropy_value": (
                None if isinstance(target.ent_coef, str)
                else float(target.ent_coef)
            ),
            "target_actor_optimizer_lr": float(
                target.actor.optimizer.param_groups[0]["lr"]
            ),
            "target_critic_optimizer_lr": float(
                target.critic.optimizer.param_groups[0]["lr"]
            ),
            "optimizer_state_transferred": False,
            "replay_transitions_transferred": 0,
            "source_replay_capacity_for_weight_transfer": int(
                getattr(source, "buffer_size", _WEIGHT_TRANSFER_REPLAY_CAPACITY)
            ),
            "target_replay_capacity": int(target.buffer_size),
            "replay_size_at_boundary": int(
                getattr(
                    getattr(target, "replay_buffer", None), "size",
                    lambda: 0,
                )()
                if getattr(target, "replay_buffer", None) is not None
                else 0
            ),
            "policy_state_transferred": True,
        }
        return target

    def load_with_observation_expansion(
        self,
        path: str,
        env,
        config: Dict[str, Any],
    ):
        """Warm-start SAC after appending visible execution-cost observations."""
        import numpy as np
        from stable_baselines3 import SAC

        self._require_continuous(env)
        source = _load_sac_source_for_weight_transfer(
            SAC,
            path,
            device=str(self._resolve(config)["device"]),
        )
        target = self.build(env, config)
        source_observation_dim = int(np.prod(source.observation_space.shape))
        target_observation_dim = int(np.prod(target.observation_space.shape))
        action_dim = int(np.prod(target.action_space.shape))
        transferred, expanded_keys = _transfer_expanded_policy_state(
            source.policy.state_dict(),
            target.policy.state_dict(),
            source_observation_dim=source_observation_dim,
            target_observation_dim=target_observation_dim,
            action_dim=action_dim,
        )
        target.policy.load_state_dict(transferred, strict=True)
        if (
            getattr(source, "log_ent_coef", None) is not None
            and getattr(target, "log_ent_coef", None) is not None
            and source.log_ent_coef.shape == target.log_ent_coef.shape
        ):
            target.log_ent_coef.data.copy_(source.log_ent_coef.data)
        target.warm_start_transfer_evidence = {
            "source_model": str(Path(path).resolve()),
            "source_observation_dim": source_observation_dim,
            "target_observation_dim": target_observation_dim,
            "added_observation_dimensions": (
                target_observation_dim - source_observation_dim
            ),
            "expanded_policy_tensors": expanded_keys,
            "new_observation_weights_initialized_to": 0.0,
            "source_replay_capacity_for_weight_transfer": int(
                getattr(source, "buffer_size", _WEIGHT_TRANSFER_REPLAY_CAPACITY)
            ),
            "target_replay_capacity": int(target.buffer_size),
            "optimizer_state_transferred": False,
            "replay_transitions_transferred": 0,
        }
        return target

    def wrap_env(self, env, config: Dict[str, Any] | None = None):
        """Called by the pipeline before build/load — SAC needs flat obs."""
        return self._flatten_if_dict(env)

    def fitness(self, summary: Dict[str, Any], config: Dict[str, Any]) -> float:
        total_return = float(summary.get("total_return") or 0.0)
        max_dd = summary.get("max_drawdown_pct") or 0.0
        try:
            max_dd = float(max_dd)
        except (TypeError, ValueError):
            max_dd = 0.0
        lam = float(config.get("ga_fitness_dd_lambda", self.params["ga_fitness_dd_lambda"]))
        return total_return - lam * (max_dd / 100.0)

    def hparam_schema(self) -> List[Tuple[str, float, float, str]]:
        return [
            ("learning_rate", 1e-5, 1e-3, "float"),
            ("batch_size", 64, 512, "int"),
            ("gamma", 0.9, 0.999, "float"),
            ("tau", 1e-3, 5e-2, "float"),
            ("train_freq", 1, 8, "int"),
            ("gradient_steps", 1, 8, "int"),
        ]

    def _resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.params)
        for k in self.params:
            if k in config and config[k] is not None:
                merged[k] = config[k]
        return merged

    @staticmethod
    def _flatten_if_dict(env):
        try:
            from gymnasium import spaces
            from gymnasium.wrappers import FlattenObservation
        except ImportError:
            return env
        obs_space = getattr(env, "observation_space", None)
        if isinstance(obs_space, spaces.Dict):
            return FlattenObservation(env)
        return env

    @staticmethod
    def _require_continuous(env) -> None:
        try:
            from gymnasium import spaces
        except ImportError:
            return
        act = getattr(env, "action_space", None)
        if act is not None and not isinstance(act, spaces.Box):
            raise ValueError(
                "sac_agent requires a continuous Box action_space. "
                "Set 'action_space_mode': 'continuous' in the env config."
            )
