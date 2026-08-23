from __future__ import annotations

from typing import Any


class _LastHiddenGRU:
    @staticmethod
    def module(input_channels: int, config: dict[str, Any]):
        import torch.nn as nn

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(
                    input_channels,
                    int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    batch_first=True,
                    dropout=float(config["dropout"]) if int(config["num_layers"]) > 1 else 0.0,
                    bidirectional=bool(config["bidirectional"]),
                )

            def forward(self, values):
                _, hidden = self.gru(values)
                directions = 2 if self.gru.bidirectional else 1
                hidden = hidden.view(self.gru.num_layers, directions, values.shape[0], self.gru.hidden_size)
                return hidden[-1].transpose(0, 1).reshape(values.shape[0], -1)

        return Encoder()


class Plugin:
    plugin_params = {
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
    }

    @staticmethod
    def build(input_channels: int, window_size: int, config: dict[str, Any]):
        hidden = int(config["hidden_size"])
        layers = int(config["num_layers"])
        if hidden < 1 or layers < 1:
            raise ValueError("gru_branch hidden_size and num_layers must be positive")
        return _LastHiddenGRU.module(input_channels, config), hidden * (2 if config["bidirectional"] else 1)

