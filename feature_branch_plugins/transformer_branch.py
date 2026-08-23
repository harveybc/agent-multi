from __future__ import annotations

from typing import Any


class Plugin:
    plugin_params = {
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "feedforward_dim": 128,
        "dropout": 0.0,
        "pool": "mean",
    }

    @staticmethod
    def build(input_channels: int, window_size: int, config: dict[str, Any]):
        import torch
        import torch.nn as nn

        model_dim = int(config["model_dim"])
        heads = int(config["num_heads"])
        if model_dim < 1 or heads < 1 or model_dim % heads:
            raise ValueError("transformer model_dim must be positive and divisible by num_heads")
        if str(config["pool"]) not in {"mean", "last"}:
            raise ValueError("transformer pool must be mean or last")

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(input_channels, model_dim)
                self.position = nn.Parameter(torch.zeros(1, window_size, model_dim))
                layer = nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=heads,
                    dim_feedforward=int(config["feedforward_dim"]),
                    dropout=float(config["dropout"]),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, int(config["num_layers"]), enable_nested_tensor=False)

            def forward(self, values):
                encoded = self.encoder(self.projection(values) + self.position[:, : values.shape[1]])
                return encoded.mean(dim=1) if config["pool"] == "mean" else encoded[:, -1]

        return Encoder(), model_dim

