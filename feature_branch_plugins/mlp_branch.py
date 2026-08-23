from __future__ import annotations

from typing import Any


class Plugin:
    plugin_params = {"hidden_dims": [64], "output_dim": 32, "activation": "relu"}

    @staticmethod
    def build(input_channels: int, window_size: int, config: dict[str, Any]):
        import torch.nn as nn

        activation = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(
            str(config["activation"]).lower()
        )
        if activation is None:
            raise ValueError("mlp_branch activation must be relu, gelu, or silu")
        dims = [input_channels * window_size, *map(int, config["hidden_dims"]), int(config["output_dim"])]
        if any(dim < 1 for dim in dims):
            raise ValueError("mlp_branch dimensions must be positive")
        layers: list[nn.Module] = [nn.Flatten()]
        for left, right in zip(dims, dims[1:]):
            layers.extend((nn.Linear(left, right), activation()))
        return nn.Sequential(*layers), dims[-1]

