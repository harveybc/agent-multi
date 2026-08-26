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

        from feature_branch_plugins._topology import (
            require_dropout, require_heads_divide, require_patch_coverage,
            require_positive_int)
        patch_len = require_positive_int(config, "patch_len")
        stride = require_positive_int(config, "stride")
        d_model, n_heads = require_heads_divide(config, "d_model",
                                                "n_heads")
        n_layers = require_positive_int(config, "n_layers")
        require_positive_int(config, "ff_mult")
        require_dropout(config)
        n_patches = require_patch_coverage(window_size, patch_len,
                                           stride)
        # DATA-SOTA-331: patches are ENDPOINT-ANCHORED — the offset
        # drops the OLDEST remainder bars so the final patch always
        # ends at the newest observation. Zero-start unfold silently
        # blinded the model to the newest bars when
        # (window - patch_len) % stride != 0 (auditor counterexample).
        tail_offset = (window_size - patch_len) % stride

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
                if tail_offset:
                    x = x[..., tail_offset:]
                patches = x.unfold(-1, patch_len, stride)  # (B,F,P,L)
                tok = self.embed(patches) + self.pos  # (B, F, P, D)
                tok = tok.reshape(b * f, -1, d_model)
                enc = self.encoder(tok, mask=self.causal_mask)
                last = enc[:, -1, :].reshape(b, f * d_model)
                return self.head(last)

        return Encoder(), d_model
