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

        channels = [int(value) for value in config["channels"]]
        kernel = int(config["kernel_size"])
        dilation_base = int(config["dilation_base"])
        if not channels or min(channels) < 1 or kernel < 1 or dilation_base < 1:
            raise ValueError("invalid tcn_branch topology")
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

