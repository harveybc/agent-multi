from __future__ import annotations

from typing import Any


class Plugin:
    plugin_params = {
        "channels": [64, 64],
        "kernel_size": 3,
        "dilation_base": 2,
        "dropout": 0.0,
        "activation": "relu",
    }

    @staticmethod
    def build(input_channels: int, window_size: int, config: dict[str, Any]):
        import torch.nn as nn

        from feature_branch_plugins._topology import (
            require_dropout, require_int_list, require_positive_int)
        channels = require_int_list(config, "channels")
        kernel = require_positive_int(config, "kernel_size")
        dilation_base = require_positive_int(config, "dilation_base")
        require_dropout(config)
        activation = {"relu": nn.ReLU, "gelu": nn.GELU}.get(str(config["activation"]).lower())
        if activation is None:
            raise ValueError("tcn_branch activation must be relu or gelu")

        class CausalBlock(nn.Module):
            def __init__(self, left: int, right: int, dilation: int):
                super().__init__()
                self.trim = (kernel - 1) * dilation
                self.conv = nn.Conv1d(left, right, kernel, padding=self.trim, dilation=dilation)
                self.norm = nn.LayerNorm(right)
                self.act = activation()
                self.drop = nn.Dropout(float(config["dropout"]))
                self.skip = nn.Identity() if left == right else nn.Conv1d(left, right, 1)

            def forward(self, values):
                residual = self.skip(values)
                out = self.conv(values)
                if self.trim:
                    out = out[..., :-self.trim]
                out = self.norm(out.transpose(1, 2)).transpose(1, 2)
                return self.act(self.drop(out) + residual)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                blocks = []
                left = input_channels
                for index, right in enumerate(channels):
                    blocks.append(CausalBlock(left, right, dilation_base ** index))
                    left = right
                self.blocks = nn.Sequential(*blocks)

            def forward(self, values):
                return self.blocks(values.transpose(1, 2))[..., -1]

        return Encoder(), channels[-1]

