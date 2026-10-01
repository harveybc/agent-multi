"""Plug a representation into the installed SAC/DQN agent plugins.

The agent plugins keep building SB3 models exactly as before. When a config
carries a ``representation`` block, ``policy_kwargs_for`` turns it into SB3
``policy_kwargs`` (features extractor class + kwargs, and for SAC the explicit
``share_features_extractor``). Without the block nothing changes: that is the
native baseline (SB3 ``FlattenExtractor``).

``representation`` block (JSON-able)::

    {"kind": "modular_temporal",             # or "native_flat"
     "modular_config": {...},                # predictor.modular.v1
     "feature_order": [...],                 # must equal env feature_columns
     "regimes": {"branch_0": "R0", ..., "core": "R0"},
     "donor": null | "<donor dir>",
     "share_features_extractor": true}       # SAC only; explicit
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .observation_layout import ObservationLayout

KINDS = ("native_flat", "modular_temporal")


def policy_kwargs_for(env, config: Dict[str, Any], *, algorithm: str) -> Dict[str, Any]:
    rep = config.get("representation")
    if not rep:
        return {}
    kind = rep.get("kind")
    if kind not in KINDS:
        raise ValueError(f"representation.kind must be one of {KINDS}; got {kind!r}")
    out: Dict[str, Any] = {}
    if algorithm == "SAC":
        if "share_features_extractor" not in rep:
            raise ValueError("SAC representation must declare share_features_extractor explicitly")
        out["share_features_extractor"] = bool(rep["share_features_extractor"])
    if kind == "native_flat":
        from .modular_torch import NativeFlatExtractor

        out["features_extractor_class"] = NativeFlatExtractor
        return out
    from .modular_torch import ModularTemporalExtractor

    feature_order = rep.get("feature_order") or config.get("feature_columns")
    if not feature_order:
        raise ValueError("modular_temporal representation needs feature_order (or env feature_columns)")
    if config.get("feature_columns") and list(config["feature_columns"]) != list(feature_order):
        raise ValueError("representation.feature_order differs from env feature_columns")
    layout = ObservationLayout.from_space(env.unwrapped.observation_space, feature_order=feature_order)
    out["features_extractor_class"] = ModularTemporalExtractor
    out["features_extractor_kwargs"] = {"layout": layout, "modular_config": rep["modular_config"],
                                        "regimes": rep.get("regimes"), "donor": rep.get("donor")}
    return out


def build_model(algorithm: str, env, *, modular_config: Dict[str, Any], feature_order,
                share_features_extractor: bool = True, regimes: Optional[Dict[str, str]] = None,
                donor: Optional[str] = None, device: str = "cpu", seed: int = 0, net_arch=(32, 32),
                **algo_params: Any):
    """Build SAC or DQN through the existing agent plugins with the modular extractor."""
    if algorithm == "SAC":
        from agent_plugins.sac_agent import Plugin
    elif algorithm == "DQN":
        from agent_plugins.dqn_agent import Plugin
    else:
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    donor = getattr(donor, "path", donor)  # a Donor object or its directory
    config: Dict[str, Any] = {
        "representation": {"kind": "modular_temporal", "modular_config": modular_config,
                           "feature_order": list(feature_order), "regimes": regimes, "donor": donor,
                           "share_features_extractor": bool(share_features_extractor)},
        "device": device, "train_seed": int(seed), "net_arch": list(net_arch),
    }
    config.update(algo_params)
    return Plugin().build(env, config)
