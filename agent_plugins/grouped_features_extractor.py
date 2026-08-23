"""SB3 extractor that preserves temporal and semantic observation structure."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent_plugins.component_config import deep_merge_strict
from app.plugin_loader import load_plugin


def _resolved_component(
    group: str,
    spec: dict[str, Any],
    *,
    path: str,
    metadata_keys: frozenset[str] = frozenset(),
):
    allowed = {"plugin", "params"} | set(metadata_keys)
    if set(spec) - allowed:
        raise ValueError(f"unknown {path} keys: {sorted(set(spec) - allowed)}")
    name = str(spec.get("plugin") or "").strip()
    if not name:
        raise ValueError(f"{path}.plugin is required")
    plugin_class, _ = load_plugin(group, name)
    params = deep_merge_strict(
        plugin_class.plugin_params, spec.get("params"), path=f"{path}.params"
    )
    return name, plugin_class, params


def build_grouped_extractor_class():
    """Import torch/SB3 lazily so ordinary CLI discovery stays lightweight."""
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class GroupedFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, architecture: dict[str, Any]):
            if not hasattr(observation_space, "spaces") or "features" not in observation_space.spaces:
                raise ValueError("grouped extractor requires a Dict observation containing 'features'")
            defaults = {
                "schema": "agent_multi.grouped_features.v1",
                "feature_columns": [],
                "branches": [],
                "state_keys": [],
                "state_branch": {"plugin": "mlp_branch", "params": None},
                "fusion": {"plugin": "gated_fusion", "params": None},
                "share_features_extractor": False,
            }
            cfg = deep_merge_strict(defaults, architecture, path="feature_extractor")
            if cfg["schema"] != defaults["schema"]:
                raise ValueError("unsupported feature extractor schema")
            feature_columns = list(cfg["feature_columns"])
            feature_shape = observation_space.spaces["features"].shape
            if len(feature_shape) != 2 or feature_shape[1] != len(feature_columns):
                raise ValueError("feature_columns must exactly describe observation_space['features']")
            if not cfg["branches"]:
                raise ValueError("feature_extractor.branches must not be empty")

            modules: list[nn.Module] = []
            indices: list[list[int]] = []
            branch_dims: list[int] = []
            claimed: set[str] = set()
            for number, branch in enumerate(cfg["branches"]):
                allowed = {"name", "features", "plugin", "params"}
                if set(branch) - allowed:
                    raise ValueError(f"unknown feature_extractor.branches[{number}] keys")
                names = list(branch.get("features") or [])
                if not names or any(name not in feature_columns for name in names):
                    raise ValueError(f"branch {number} contains missing or unknown features")
                overlap = claimed.intersection(names)
                if overlap:
                    raise ValueError(f"features assigned to multiple branches: {sorted(overlap)}")
                claimed.update(names)
                _, plugin_class, params = _resolved_component(
                    "feature_branch.plugins",
                    branch,
                    path=f"feature_extractor.branches[{number}]",
                    metadata_keys=frozenset({"name", "features"}),
                )
                module, output_dim = plugin_class.build(len(names), int(feature_shape[0]), params)
                modules.append(module)
                indices.append([feature_columns.index(name) for name in names])
                branch_dims.append(int(output_dim))

            missing = set(feature_columns) - claimed
            if missing:
                raise ValueError(f"feature columns without a semantic branch: {sorted(missing)}")

            state_keys = list(cfg["state_keys"])
            if state_keys:
                absent = [key for key in state_keys if key not in observation_space.spaces]
                if absent:
                    raise ValueError(f"state_keys absent from observation space: {absent}")
                _, plugin_class, params = _resolved_component(
                    "feature_branch.plugins", cfg["state_branch"], path="feature_extractor.state_branch"
                )
                state_width = sum(int(observation_space.spaces[key].shape[0]) for key in state_keys)
                module, output_dim = plugin_class.build(state_width, 1, params)
                modules.append(module)
                branch_dims.append(int(output_dim))

            _, fusion_class, fusion_params = _resolved_component(
                "feature_fusion.plugins", cfg["fusion"], path="feature_extractor.fusion"
            )
            fusion, features_dim = fusion_class.build(branch_dims, fusion_params)
            super().__init__(observation_space, features_dim)
            self.temporal_branches = nn.ModuleList(modules[: len(indices)])
            self.feature_indices = deepcopy(indices)
            self.state_keys = state_keys
            self.state_branch = modules[-1] if state_keys else None
            self.fusion = fusion
            self.effective_architecture = cfg

        def forward(self, observations):
            features = observations["features"].float()
            encoded = [
                branch(features[:, :, indices])
                for branch, indices in zip(self.temporal_branches, self.feature_indices)
            ]
            if self.state_branch is not None:
                state = torch.cat([observations[key].float() for key in self.state_keys], dim=-1)
                encoded.append(self.state_branch(state.unsqueeze(1)))
            return self.fusion(encoded)

    return GroupedFeaturesExtractor
