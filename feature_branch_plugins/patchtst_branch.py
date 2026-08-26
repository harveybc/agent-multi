"""PatchTST-style branch (Data-First order §3): channel-independent
patching + Transformer encoder with a CAUSAL mask over patch tokens.
Contract: build(input_channels, window_size, config) -> (module, dim);
module maps (B, T, F) -> (B, dim) using only in-window (past) data."""
from __future__ import annotations

from typing import Any


class Plugin:
    plugin_params = {
        "patch_len": 8,
        "stride": 8,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "ff_mult": 2,
        "dropout": 0.0,
    }

    @staticmethod
    def build(input_channels: int, window_size: int, config: dict[str, Any]):
        import torch
        import torch.nn as nn

        patch_len = int(config["patch_len"])
        stride = int(config["stride"])
        d_model = int(config["d_model"])
        n_heads = int(config["n_heads"])
        n_layers = int(config["n_layers"])
        if patch_len < 1 or stride < 1 or patch_len > window_size:
            raise ValueError("invalid patchtst_branch patching")
        if d_model % n_heads:
            raise ValueError("patchtst_branch d_model must divide n_heads")
        n_patches = 1 + (window_size - patch_len) // stride

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(patch_len, d_model)
                self.pos = nn.Parameter(
                    torch.zeros(1, n_patches, d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model, n_heads, d_model * int(config["ff_mult"]),
                    float(config["dropout"]), batch_first=True,
                    norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, n_layers)
                self.head = nn.Linear(d_model * input_channels, d_model)
                mask = torch.triu(
                    torch.full((n_patches, n_patches), float("-inf")),
                    diagonal=1)
                self.register_buffer("causal_mask", mask)

            def forward(self, values):  # (B, T, F)
                b, t, f = values.shape
                x = values.permute(0, 2, 1)          # (B, F, T)
                patches = x.unfold(-1, patch_len, stride)  # (B,F,P,L)
                tok = self.embed(patches) + self.pos  # (B, F, P, D)
                tok = tok.reshape(b * f, -1, d_model)
                enc = self.encoder(tok, mask=self.causal_mask)
                last = enc[:, -1, :].reshape(b, f * d_model)
                return self.head(last)

        return Encoder(), d_model
