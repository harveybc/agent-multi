"""Policy bundle: SB3 zip + the facts a fresh process needs to reproduce decisions.

A policy zip alone restores weights. It does not say which observation layout
the policy was trained on, how features were normalized, or how a continuous
output becomes an order. The bundle records all of that and the loader refuses
an env whose contract differs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

BUNDLE_SCHEMA = "rl_temporal.policy_bundle.v1"
DISCRETE_MAPPING = {"0": "hold", "1": "long", "2": "short"}

_NORMALIZATION_KEYS = ("feature_scaling", "feature_scaling_window", "feature_clip", "include_price_window",
                       "include_agent_state", "agent_state_contract", "window_size", "feature_columns",
                       "feature_binary_columns", "preprocessor_plugin")


def action_mapping_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(cfg.get("action_space_mode", "discrete")).lower()
    if mode == "continuous":
        return {"action_space_mode": "continuous", "space": "Box(-1, 1, (1,))",
                "continuous_action_threshold": float(cfg.get("continuous_action_threshold", 0.33)),
                "continuous_action_contract": cfg.get("continuous_action_contract", "legacy_directional_v1"),
                "effective_actions": {">= +thr": "long", "<= -thr": "short", "else": "hold"},
                "note": "the env's own tested threshold mapping; not a discrete-SAC"}
    return {"action_space_mode": "discrete", "space": "Discrete(3)", "mapping": dict(DISCRETE_MAPPING)}


def normalization_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: cfg.get(k) for k in _NORMALIZATION_KEYS}
    out["window"] = cfg.get("window_size")                      # bars per observation
    out["bar_period"] = cfg.get("bar_period") or cfg.get("timeframe") or cfg.get("bar") or "UNDECLARED"
    out["other_frequency_streams"] = cfg.get("other_frequency_streams", [])  # none: one stream, one bar grid
    return out


def _versions() -> Dict[str, str]:
    import stable_baselines3
    import torch

    out = {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__}
    try:
        import gymnasium

        out["gymnasium"] = gymnasium.__version__
    except ImportError:
        pass
    return out


def _env_config(env) -> Dict[str, Any]:
    return dict(getattr(env.unwrapped, "config", {}) or {})


def _representation(model) -> Dict[str, Any]:
    from .donor_contract import component_identity
    from .modular_torch import ModularTemporalExtractor, modular_config_sha256, parameter_accounting
    from .observation_layout import ObservationLayout

    if hasattr(model, "actor"):          # SAC: the actor owns (or shares) the extractor
        ext = model.actor.features_extractor
    elif hasattr(model, "q_net"):        # DQN: the online network owns it
        ext = model.q_net.features_extractor
    else:
        ext = getattr(model.policy, "features_extractor", None)
    if isinstance(ext, ModularTemporalExtractor):
        return {"kind": "modular_temporal", "layout_digest": ext.layout.digest,
                "layout": ext.layout.declaration(), "modular_config_sha256": modular_config_sha256(ext.modular_config),
                "regimes": dict(ext.regimes), "donor": ext.donor, "identity": component_identity(ext.encoder),
                "parameters": parameter_accounting(ext)}
    space = model.observation_space
    return {"kind": "native_flat", "layout_digest": None, "layout": None,
            "modular_config_sha256": None, "regimes": None, "donor": None,
            "extractor": type(ext).__name__, "observation_dim": int(np.prod(space.shape))}


def save_policy_bundle(model, directory, *, env_config: Dict[str, Any], arm: str,
                       tolerance_abs: float = 1e-6, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    zip_path = d / "policy.zip"
    model.save(zip_path)
    doc = {"schema": BUNDLE_SCHEMA, "arm": arm, "algorithm": type(model).__name__,
           "policy_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
           "action_mapping": action_mapping_from_config(env_config),
           "normalization": normalization_from_config(env_config),
           "representation": _representation(model), "versions": _versions(),
           "tolerance": {"device": str(model.device), "abs": float(tolerance_abs)},
           "num_timesteps": int(model.num_timesteps),
           # M05 shadow intake contract: never an execution claim, evidence declared explicitly
           "execution_authorized": False,
           "evidence": {"research_validated": False, "live_inference_eligible": False,
                        "live_execution_eligible": False},
           **(extra or {})}
    (d / "env_config.json").write_text(json.dumps(env_config, indent=2, sort_keys=True, default=str) + "\n")
    (d / "bundle.json").write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    return doc


def _check(name: str, saved: Any, live: Any) -> None:
    if saved != live:
        raise ValueError(f"{name} mismatch: bundle {saved!r} vs env {live!r}")


def load_policy_bundle(directory, env, *, device: str = "cpu"):
    d = Path(directory)
    doc = json.loads((d / "bundle.json").read_text())
    if doc.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported bundle schema {doc.get('schema')!r}")
    zip_path = d / "policy.zip"
    if hashlib.sha256(zip_path.read_bytes()).hexdigest() != doc["policy_zip_sha256"]:
        raise ValueError("policy.zip sha256 mismatch")
    cfg = _env_config(env)
    _check("action_mapping", doc["action_mapping"], action_mapping_from_config(cfg))
    _check("normalization", doc["normalization"], json.loads(json.dumps(normalization_from_config(cfg), default=str)))
    if doc["representation"].get("layout_digest"):
        from .observation_layout import ObservationLayout

        live = ObservationLayout.from_space(env.unwrapped.observation_space,
                                           feature_order=doc["representation"]["layout"]["feature_order"])
        _check("layout_digest", doc["representation"]["layout_digest"], live.digest)
    if doc["algorithm"] == "SAC":
        from stable_baselines3 import SAC as Algo
    elif doc["algorithm"] == "DQN":
        from stable_baselines3 import DQN as Algo
    else:
        raise ValueError(doc["algorithm"])
    return Algo.load(zip_path, env=env, device=device)


def decisions(model, observations: np.ndarray) -> List[float]:
    """Deterministic decisions on a fixed batch, as plain floats (ints for Discrete)."""
    out = []
    for obs in np.asarray(observations, dtype=np.float32):
        action, _ = model.predict(obs, deterministic=True)
        a = np.asarray(action).reshape(-1)
        out.append(float(a[0]) if a.size == 1 else [float(v) for v in a])
    return out
